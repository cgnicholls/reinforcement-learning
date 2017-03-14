# coding: utf-8

# We use q-learning. We approximate the q function with a neural network using
# tensorflow.

import tensorflow as tf
import random
import numpy as np
import gym
from collections import deque

import matplotlib.pyplot as plt
import time

# Set up the action space
ACTIONS = [0,2,3]
NUM_ACTIONS = len(ACTIONS)

# The number of states to compute the average q value with
BENCHMARK_STATES = 100

# The number of nonzero rewards to compute the running average with
NONZERO_REWARD_MEMORY = 500

# The initial learning rate to use
INITIAL_LEARNING_RATE = 1e-6

# The number of frames to use as our state
STATE_FRAMES = 4

# The size to resize the frame to
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = 80, 80

# Epsilon greedy
EPSILON_GREEDY_STEPS = 5000 # The total number of time steps to anneal epsilon
INITIAL_EPSILON_GREEDY = 1.0 # Initial epsilon
FINAL_EPSILON_GREEDY = 0.1 # Final epsilon

# Observation period
OBSERVATION_STEPS = 5000 # Time steps to observe before training
MEMORY_SIZE = 100000

# The minibatch size to train with
MINI_BATCH_SIZE = 100

# The discount factor to use
DISCOUNT_FACTOR = 0.99

# Output average Q value every VERBOSE_EVERY_STEPS
VERBOSE_EVERY_STEPS = 100

# Copy the network every 1000 steps
UPDATE_NETWORK_EVERY = 1000

# Train the agent
def train(sess, network, observations):
    # Sample a minibatch to train on
    mini_batch = random.sample(observations, MINI_BATCH_SIZE)

    states = [d['state'] for d in mini_batch]
    actions = [d['action'] for d in mini_batch]
    rewards = [d['reward'] for d in mini_batch]
    next_states = [d['next_state'] for d in mini_batch]

    # Compute Q(s', a'; theta_{i-1}). This is an unbiased estimator for y_i as
    # in eqn 2 in the DQN paper.
    next_q = sess.run(network.output_layer, feed_dict={network.input_layer : \
            next_states})

    target_q = []
    for i in xrange(len(mini_batch)):
        if mini_batch[i]['terminal']:
            # This was a terminal frame
            target_q.append(rewards[i])
        else:
            target_q.append(rewards[i] + DISCOUNT_FACTOR * \
                    np.max(next_q[i]))

    one_hot_actions = compute_one_hot_actions(actions)

    network.train(sess, states, one_hot_actions, target_q)

# Return a one hot vector with a 1 at the index for the action.
def compute_one_hot_actions(actions):
    one_hot_actions = []
    for i in xrange(len(actions)):
        one_hot = np.zeros([NUM_ACTIONS])
        one_hot[ACTIONS.index(actions[i])] = 1
        one_hot_actions.append(one_hot)
    return one_hot_actions

# Deep Q-learning on pong
def pong_deep_q_learn():

    # Create tensorflow session
    tf_sess = tf.Session()

    # Create tensorflow network
    q_network = NetworkDeepmind("q_network")
    target_network = NetworkDeepmind("target_network")

    tf_sess.run(tf.global_variables_initializer())

    epsilon_greedy = INITIAL_EPSILON_GREEDY

    observations = deque()
    actions = []
    avg_q_history = []
    avg_reward_history = []
    nonzero_rewards = deque()

    # Set up plotting
    plot = False
    if plot:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

    # Give this run of the program an identifier
    identifier = str(time.gmtime()[0:5])
    identifier = identifier.replace('(', '').replace(')', '')
    identifier = identifier.replace(' ', '-').replace(',','')

    env = gym.make('Pong-v0')
    obs = env.reset()

    # Compute the first state
    current_state = compute_state(None, obs)

    # Enter loop over number of time steps
    t = 0
    while True:

        # At timestep OBSERVATION_STEPS, we generate some random states to test
        # the average q value
        if t == OBSERVATION_STEPS-1:
            benchmark_observations = random.sample(observations,
                    BENCHMARK_STATES)
            benchmark_states = [d['state'] for d in benchmark_observations]

        # Copy the network parameters from the target network to the q_network
        # every UPDATE_NETWORK_EVERY timesteps.
        if (t % UPDATE_NETWORK_EVERY == 0) and t > OBSERVATION_STEPS:
            print "Updating Q network"
            copy_network_params(tf_sess, target_network, q_network)

        # Compute action
        action = compute_action(tf_sess, q_network, current_state, epsilon_greedy)

        # Take a step with action
        obs, reward, terminal, info = env.step(action)

        # Update the current and next states
        next_state = compute_state(current_state, obs)

        # Record transitions
        observations.append({'state': current_state, 'action':
            action, 'reward': reward, 'next_state': next_state,
            'terminal': terminal})

        # Compute average reward
        if reward != 0:
            nonzero_rewards.append(reward)
            if len(nonzero_rewards) > NONZERO_REWARD_MEMORY:
                nonzero_rewards.popleft()

        if t < OBSERVATION_STEPS and t % 100 == 0:
            print "Observing", t, "/", OBSERVATION_STEPS

        # Compute the average q-value, and display the average reward and
        # average q-value
        if (t >= OBSERVATION_STEPS) and (t % VERBOSE_EVERY_STEPS == 0):
            avg_q_value = compute_average_q_value(tf_sess, q_network, benchmark_states)
            print("Time: {}. Average Q-value: {}".format(t, avg_q_value))
            avg_q_history.append(avg_q_value)

            print("Average nonzero reward: {}".format(np.mean(nonzero_rewards)))
            avg_reward_history.append(np.mean(nonzero_rewards))

            print("Epsilon: {}".format(epsilon_greedy))

            # Plot the data, but don't show it
            plot_data = np.append(np.array(avg_q_history)[:,np.newaxis],
                    np.array(avg_reward_history)[:,np.newaxis], axis=1)

            # Save the plot
            plt.plot(plot_data)
            plt.savefig("rewardhistory" + identifier + ".jpg")

            if plot:
                ax1.clear()
                ax1.plot(avg_q_history)
                plt.pause(0.0001) 

        # Ensure we don't go over our memory size
        if len(observations) > MEMORY_SIZE:
            observations.popleft()

        # Train the target network if we have reached the number of 
        # observation steps
        if (t >= OBSERVATION_STEPS):
            train(tf_sess, target_network, observations)
    
        # Anneal epsilon for epsilon-greedy strategy
        if epsilon_greedy > FINAL_EPSILON_GREEDY and len(observations) > \
                OBSERVATION_STEPS:
            epsilon_greedy -= (INITIAL_EPSILON_GREEDY - FINAL_EPSILON_GREEDY) \
                    / EPSILON_GREEDY_STEPS

        # If terminal, then reset the environment
        if terminal:
            obs = env.reset()
            # Compute the first state again
            current_state = compute_state(None, obs)
        # Otherwise, update current_state
        else:
            current_state = next_state

        # Update t
        t += 1

# If current_state is None then just repeat the observation STATE_FRAMES times.
# Otherwise, remove the first frame, and append obs to get the new current
# state.
def compute_state(current_state, obs):
    # First preprocess the observation
    obs = preprocess(obs)

    if current_state is None:
        state = np.stack(tuple(obs for i in range(STATE_FRAMES)), axis=2)
    else:
        # obs is two-dimensional, so insert a dummy third dimension
        state = np.append(current_state[:,:,1:], obs[:,:,np.newaxis], axis=2)
    return state

# Preprocess the observation to remove noise. Specific to pong.
def preprocess(obs):
    # Convert to float
    obs = obs.astype('float32')

    # Crop screen and set to grayscale
    obs = obs[34:194,:,0]

    # Downsize screen
    obs = obs[::2,::2]

    # Erase background
    obs[obs == 144] = 0
    obs[obs == 109] = 0

    # Set everything else to 1
    obs[obs != 0] = 1
    return obs

# Compute the action predicted by the current parameters of the q network for
# the current state.
def compute_action(tf_sess, network, state, epsilon_greedy):
    # Choose an action randomly with probability epsilon_greedy.
    if random.random() <= epsilon_greedy:
        return random.choice(ACTIONS)
    # Otherwise, choose an action according to the Q-function
    else:
        return network.compute_action(tf_sess, state)

# Compute the average q value for a given set of states. This can be used as an
# indicator of training progress.
def compute_average_q_value(sess, network, states):
    q_vals = sess.run(network.output_layer, feed_dict={
        network.input_layer: states
    })

    # Compute the average of the maximum q-value for each state
    avg_max_q_values = np.mean(np.max(q_vals, axis=1))

    return avg_max_q_values

# Updates the network parameters -- thanks to Denny Britz's code for the idea!
def copy_network_params(sess, net1, net2):
    net1_params = [t for t in tf.trainable_variables() if t.name.startswith(net1.scope)]
    net2_params = [t for t in tf.trainable_variables() if t.name.startswith(net2.scope)]
    net1_params = sorted(net1_params, key = lambda v : v.name)
    net2_params = sorted(net2_params, key = lambda v : v.name)

    update_ops = []
    for net1_v, net2_v in zip(net1_params, net2_params):
        update_ops.append(net2_v.assign(net1_v))

    sess.run(update_ops)

class NetworkDeepmind():
    def __init__(self, scope):
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.create_network()

    def create_network(self):
        conv1_W = tf.Variable(tf.truncated_normal([8, 8, STATE_FRAMES, 32],
            stddev=0.01))
        conv1_b = tf.Variable(tf.constant(0.1, shape=[32]))

        conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
        conv2_b = tf.Variable(tf.constant(0.1, shape=[64]))

        conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
        conv3_b = tf.Variable(tf.constant(0.1, shape=[64]))

        fc1_W = tf.Variable(tf.truncated_normal([7*7*64, 512], stddev=0.1))
        fc1_b = tf.Variable(tf.constant(0.1, shape=[512]))
        
        fc2_W = tf.Variable(tf.truncated_normal([512, NUM_ACTIONS], stddev=0.1))
        fc2_b = tf.Variable(tf.constant(0.1, shape=[NUM_ACTIONS]))

        self.input_layer = tf.placeholder("float", [None, RESIZED_SCREEN_X,
            RESIZED_SCREEN_Y, STATE_FRAMES])

        conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, conv1_W,
            strides=[1,4,4,1], padding="SAME") + conv1_b) 

        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_W, strides=[1,2,2,1],
            padding="VALID") + conv2_b)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, conv3_W, strides=[1,1,1,1],
            padding="VALID") + conv3_b)

        flatten = tf.reshape(conv3, [-1, 7*7*64])

        fc1 = tf.nn.relu(tf.matmul(flatten, fc1_W) + fc1_b)

        self.output_layer = tf.matmul(fc1, fc2_W) + fc2_b

        # A one-hot vector specifying the action
        self.action = tf.placeholder("float", [None, NUM_ACTIONS])

        # The target for Q-learning. This is y_i in eqn 2 of the DQN paper.
        self.target = tf.placeholder("float", [None])

        # The q-value for the specified action, where tf_action is a one-hot vector.
        self.q_for_action = tf.reduce_sum(self.output_layer * self.action,
                reduction_indices=1)

        # The cost we try to minimise, as in eqn 2 of the DQN paper
        self.cost = tf.reduce_mean(tf.square(self.target - self.q_for_action))

        # The train operation: reduce the cost using Adam
        self.train_operation = \
                tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(self.cost)

    def compute_action(self, sess, state):
        q = sess.run(self.output_layer, feed_dict={self.input_layer: [state]})[0]
        action_index = np.argmax(q)
        return ACTIONS[action_index]

    def train(self, sess, states, actions, targets):
        sess.run(self.train_operation, feed_dict={
            self.input_layer: states,
            self.action: actions,
            self.target: targets
        })

# Run deep q learning
pong_deep_q_learn()
