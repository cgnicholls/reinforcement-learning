# coding: utf-8
# Implements the asynchronous advantage actor-critic algorithm.

# Pseudocode (for each actor-learner thread):
# Assume global shared theta, theta_target and counter T = 0.
# Initialize thread step counter t <- 0
# Initialize target network weights theta_target <- theta
# Initialize network gradients dtheta <- 0
# Get initial state s
# repeat
#   take action a with epsilon-greedy policy based on Q(s,a;theta)
#   receive new state s' and reward r
#   y = r (for terminal s')
#     = r + gamma * max_a' Q(s', a', theta_target) (for non-terminal s')
#   accumulate gradients wrt theta: dtheta <- dtheta + grad_theta(y-Q(s,a;theta))^2
#   s = s'
#   T <- T + 1 and t <- t + 1
#   if T mod I_target == 0:
#       update target network theta_target <- theta
#   if t mod I_asyncupdate == 0 or s is terminal:
#       perform asynchronous update of theta using dtheta
#       clear gradients dtheta <- 0
# until T > T_max

import os
import sys
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import sleep
import gym
import Queue
from custom_gym import CustomGym
import random

random.seed(100)

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

# FLAGS
ENTROPY_FACTOR = 0.01
T_MAX = 100000000
NUM_THREADS = 1
STATE_FRAMES = 4
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
VERBOSE_EVERY = 2000
EPSILON_STEPS = 1000#4000000
TESTING = False

I_ASYNC_UPDATE = 5

training_finished = False

class Agent():
    def __init__(self, session, action_size, h, w, channels,
        optimizer=tf.train.AdamOptimizer(1e-4)):

        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session
        K.set_session(self.sess)

        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            self.weights, self.state, self.policy, self.value = self.build_model(h, w, channels)
            self.advantages = tf.placeholder('float32', [None], name='advantages')

        with tf.variable_scope('optimizer'):
            # Compute the one hot vectors for each action given.
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

            # For a given state and action, compute the log of the policy at
            # that action for that state. This also works on batches.
            self.log_pi_for_action = tf.reduce_sum(tf.multiply(tf.log(self.policy), action_one_hot), reduction_indices=1)

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
            self.entropy = tf.reduce_sum(tf.multiply(self.policy, -tf.log(self.policy)))

            # Try to minimise the loss. There is some rationale for choosing the
            # weighted linear combination here that I found somewhere else that
            # I can't remember, but I haven't tried to optimise it.
            # Note the negative entropy term, which encourages exploration:
            # higher entropy corresponds to less certainty.
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * ENTROPY_FACTOR
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimizer.apply_gradients(grads_vars)

    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()
    
    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    # Train the network on the given states and rewards
    def train(self, states, actions, target_values, advantages):
        # Training
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.target_value: target_values,
            self.advantages: advantages
        })
        if TESTING:
            advantages, log_policy, policy_loss, value_loss, entropy, loss, _ =\
            self.sess.run([self.advantages, self.log_pi_for_action,
            self.policy_loss, self.value_loss, self.entropy, self.loss],
            feed_dict={
                self.state: states,
                self.action: actions,
                self.target_value: target_values,
                self.advantages: advantages
            })
            print "loss", loss
            print "policy_loss", policy_loss, "value_loss", \
            value_loss, "entropy", entropy, "loss", loss
            print "Advantages", advantages, "log policy", log_policy

    # Builds the DQN model as in Mnih, but we get a softmax output for the
    # policy from fc1 and a linear output for the value from fc1.
    def build_model(self, h, w, channels, hidden_size=256):
        model_input = Input(shape=(h,w,channels,))
        conv1 = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same', dim_ordering='tf')(model_input)
        conv2 = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same', dim_ordering='tf')(conv1)

        flatten = Flatten()(conv2)
        fc1 = Dense(output_dim=hidden_size, activation='relu')(flatten)
        policy = Dense(output_dim=self.action_size, activation='softmax')(fc1)
        value = Dense(output_dim=1, activation='linear')(fc1)

        # Store the inputs and outputs in the model
        model = Model(input=model_input, outputs=[policy, value])
        trainable_weights = model.trainable_weights

        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        policy_out, value_out = model(state)

        # Actually evaluate the inputs 
        return trainable_weights, state, policy_out, value_out

class Summary:
    def __init__(self, logdir, agent):
        with tf.variable_scope('summary'):
            summarising = ['episode_avg_reward', 'avg_value']
            self.agent = agent
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_ops = {}
            self.summary_vars = {}
            self.summary_ph = {}
            for s in summarising:
                self.summary_vars[s] = tf.Variable(0.0)
                self.summary_ph[s] = tf.placeholder('float32', name=s)
                self.summary_ops[s] = tf.summary.scalar(s, self.summary_vars[s])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.summary.merge(list(self.summary_ops.values()))
            
    def write_summary(self, summary, t):
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in summary.items()})
        summary_to_add = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in summary.items()})
        self.writer.add_summary(summary_to_add, global_step=t)
        

def get_epsilon(global_step, epsilon_steps, epsilon_min):
    epsilon = 1.0 - float(global_step) / float(epsilon_steps) * (1.0 - epsilon_min)
    return epsilon if epsilon > epsilon_min else epsilon_min

def async_trainer(agent, env, sess, thread_idx, T_queue, summary):
    print "Training thread", thread_idx
    # Choose a minimum epsilon once and for all for this agent.
    T = T_queue.get()
    T_queue.put(T+1)
    epsilon_min = random.choice(4*[0.1] + 3*[0.01] + 3*[0.5])
    epsilon = get_epsilon(T, EPSILON_STEPS, epsilon_min)
    t = 0

    last_verbose = T
    last_target_update = T

    terminal = True
    while T < T_MAX:
        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []

        if terminal:
            terminal = False
            state = env.reset()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # Save the current state
            batch_states.append(state)
            
            if random.random() < epsilon:
                action_idx = random.randrange(agent.action_size)
            else:
                # Choose an action randomly according to the policy
                # probabilities
                policy = agent.get_policy(state)
                action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal.
            state, reward, terminal, _ = env.step(action_idx)

            # Update counters
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)

            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)

        target_value = 0
        # If the last state was terminal, just put R = 0. Else we want the
        # estimated value of the last state.
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)
        # Reverse the batch target values, so they are in the correct order
        # again.
        batch_target_values.reverse()

        # Test batch targets
        if TESTING:
            temp_rewards = batch_rewards + [last_R]
            test_batch_target_values = []
            for j in range(len(batch_rewards)):
                test_batch_target_values.append(discount(temp_rewards[j:], DISCOUNT_FACTOR))
            if not test_equals(batch_target_values, test_batch_target_values,
                1e-5):
                print "Assertion failed"
                print last_R
                print batch_rewards
                print batch_target_values
                print test_batch_target_values

        # Compute the estimated value of each state
        baseline_values = sess.run(agent.value, feed_dict={
            agent.state: np.vstack(batch_states)
            })
        baseline_values = np.reshape(baseline_values, -1)
        batch_advantages = batch_target_values - baseline_values

        # Apply asynchronous gradient update
        agent.train(np.vstack(batch_states), batch_actions, batch_target_values,
        batch_advantages)

        # Anneal epsilon
        epsilon = get_epsilon(T, EPSILON_STEPS, epsilon_min)

        if thread_idx == 0:
            if T - last_verbose >= VERBOSE_EVERY and terminal:
                print "Worker", thread_idx, "T", T, "Evaluating agent"
                last_verbose = T
                episode_rewards, episode_vals = estimate_reward(agent, env, episodes=5)
                avg_ep_r = np.mean(episode_rewards)
                avg_val = np.mean(episode_vals)
                print "Avg ep reward", avg_ep_r, "epsilon", epsilon, "Average value", avg_val
                summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_value': avg_val}, T)
    global training_finished
    training_finished = True

def estimate_reward(agent, env, episodes=10):
    episode_rewards = []
    episode_vals = []
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            policy = agent.get_policy(state)
            value = agent.get_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            episode_vals.append(value)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return episode_rewards, episode_vals

def a3c(game_name, nb_threads=8):
    processes = []
    envs = []
    for _ in range(nb_threads):
        gym_env = gym.make(game_name)
        env = CustomGym(gym_env)
        envs.append(env)

    T_queue = Queue.Queue()
    T_queue.put(0)

    with tf.Session() as sess:
        agent = Agent(session=sess, action_size=envs[0].action_size,
        h=84, w=84, channels=STATE_FRAMES,
        optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))

        sess.run(tf.global_variables_initializer())

        summary = Summary('tensorboard', agent)

        for i in range(nb_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue, summary,)))
        for p in processes:
            p.daemon = True
            p.start()

        while not training_finished:
            sleep(0.01)
        for p in processes:
            p.join()

# Returns sum(rewards[i] * gamma**i)
def discount(rewards, gamma):
    return np.sum([rewards[i] * gamma**i for i in range(len(rewards))])

def test_equals(arr1, arr2, eps):
    return np.sum(np.abs(np.array(arr1)-np.array(arr2))) < eps

a3c('SpaceInvaders-v0', NUM_THREADS)
