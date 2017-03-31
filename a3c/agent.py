import tensorflow as tf
import numpy as np

class Agent():
    def __init__(self, session, action_size, model='mnih',
        optimizer=tf.train.AdamOptimizer(1e-4)):

        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session

        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            if model == 'mnih':
                self.state, self.policy, self.value = self.build_model(84, 84, 4)
            else:
                # Assume we wanted a feedforward neural network
                self.state, self.policy, self.value = self.build_model_feedforward(4)
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            scope='network')
            self.advantages = tf.placeholder('float32', [None], name='advantages')

        with tf.variable_scope('optimizer'):
            # Compute the one hot vectors for each action given.
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

            min_policy = 1e-8
            max_policy = 1.0 - 1e-8
            self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))

            # For a given state and action, compute the log of the policy at
            # that action for that state. This also works on batches.
            self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices=1)

            # Takes in R_t - V(s_t) as in the async paper. Note that we feed in
            # the advantages so that V(s_t) is treated as a constant for the
            # gradient. This is because V(s_t) is the baseline (called 'b' in
            # the REINFORCE algorithm). As long as the baseline is constant wrt
            # the parameters we are optimising (in this case those for the
            # policy), then the expected value of grad_theta log pi * b is zero,
            # so the choice of b doesn't affect the expectation. It reduces the
            # variance though.
            # We want to do gradient ascent on the expected discounted reward.
            # The gradient of the expected discounted reward is the gradient of
            # log pi * (R - estimated V), where R is the sampled reward from the
            # given state following the policy pi. Since we want to maximise
            # this, we define the policy loss as the negative and get tensorflow
            # to do the automatic differentiation for us.
            self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)

            # The value loss is much easier to understand: we want our value
            # function to accurately estimated the sampled discounted rewards,
            # so we just impose a square error loss.
            # Note that the target value should be the discounted reward for the
            # state as just sampled.
            self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))

            # We follow Mnih's paper and introduce the entropy as another loss
            # to the policy. The entropy of a probability distribution is just
            # the expected value of - log P(X), denoted E(-log P(X)), which we
            # can compute for our policy at any given state with
            # sum(policy * -log(policy)), as below. This will be a positive
            # number, since self.policy contains numbers between 0 and 1, so the
            # log is negative. Note that entropy is smaller when the probability
            # distribution is more concentrated on one action, so a larger
            # entropy implies more exploration. Thus we penalise small entropy,
            # or equivalently, add -entropy to our loss.
            self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))

            # Try to minimise the loss. There is some rationale for choosing the
            # weighted linear combination here that I found somewhere else that
            # I can't remember, but I haven't tried to optimise it.
            # Note the negative entropy term, which encourages exploration:
            # higher entropy corresponds to less certainty.
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy\
            * 0.01

            # Compute the gradient of the loss with respect to all the weights,
            # and create a list of tuples consisting of the gradient to apply to
            # the weight.
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))

            # Create an operator to apply the gradients using the optimizer.
            # Note that apply_gradients is the second part of minimize() for the
            # optimizer, so will minimize the loss.
            self.train_op = optimizer.apply_gradients(grads_vars)

    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()

    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    def get_policy_and_value(self, state):
        policy, value = self.sess.run([self.policy, self.value], {self.state:
        state})
        return policy.flatten(), value.flatten()

    # Train the network on the given states and rewards
    def train(self, states, actions, target_values, advantages):
        # Training
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.target_value: target_values,
            self.advantages: advantages
        })

    # Builds the DQN model as in Mnih, but we get a softmax output for the
    # policy from fc1 and a linear output for the value from fc1.
    def build_model(self, h, w, channels):
        self.layers = {}
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        self.layers['state'] = state
        # First convolutional layer
        with tf.variable_scope('conv1'):
            conv1 = tf.contrib.layers.convolution2d(inputs=state,
            num_outputs=16, kernel_size=[8,8], stride=[4,4], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer())
            self.layers['conv1'] = conv1

        # Second convolutional layer
        with tf.variable_scope('conv2'):
            conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=32,
            kernel_size=[4,4], stride=[2,2], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer())
            self.layers['conv2'] = conv2

        # Flatten the network
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(inputs=conv2)
            self.layers['flatten'] = flatten

        # Fully connected layer with 256 hidden units
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc1'] = fc1

        # The policy output
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(inputs=fc1,
            num_outputs=self.action_size, activation_fn=tf.nn.softmax,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None)
            self.layers['policy'] = policy

        # The value output
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None)
            self.layers['value'] = value

        return state, policy, value

    # Builds a simple feedforward model to learn the cart and pole environment.
    def build_model_feedforward(self, input_dim, num_hidden=30):
        self.layers = {}
        state = tf.placeholder('float32', shape=(None, input_dim), name='state')

        self.layers['state'] = state
        # Fully connected layer with num_hidden hidden units
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs=state,
            num_outputs=num_hidden,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc1'] = fc1

        # Fully connected layer with num_hidden hidden units
        with tf.variable_scope('fc2'):
            fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
            num_outputs=num_hidden,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc2'] = fc2

        # The policy output to the two possible actions
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(inputs=fc2,
            num_outputs=self.action_size, activation_fn=tf.nn.softmax,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['policy'] = policy

        # The value output
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['value'] = value

        return state, policy, value
