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
ACTIONS = [0,1]
NUM_ACTIONS = len(ACTIONS)

# The number of states to compute the average q value with
BENCHMARK_STATES = 100

# The number of nonzero rewards to compute the running average with
NONZERO_REWARD_MEMORY = 50

# The initial learning rate to use
INITIAL_LEARNING_RATE = 1e-4

# Epsilon greedy
EPSILON_GREEDY_STEPS = 10000 # The total number of time steps to anneal epsilon
INITIAL_EPSILON_GREEDY = 1.0 # Initial epsilon
FINAL_EPSILON_GREEDY = 0.001 # Final epsilon

# Observation period
OBSERVATION_STEPS = 5000 # Time steps to observe before training
MEMORY_SIZE = 100000

# Render an episode every RENDER_EVERY epsiodes.
RENDER = False
RENDER_EVERY = 20

# The minibatch size to train with
MINI_BATCH_SIZE = 128

# The discount factor to use
DISCOUNT_FACTOR = 0.5

# Output average Q value every VERBOSE_EVERY_STEPS
VERBOSE_EVERY_STEPS = 100

# Train the agent
def train(tf_sess, observations, tf_input_layer, tf_output_layer,
        tf_train_operation, tf_action, tf_target, tf_cost):
    # Sample a minibatch to train on
    mini_batch = random.sample(observations, MINI_BATCH_SIZE)

    states = [d['state'] for d in mini_batch]
    actions = [d['action'] for d in mini_batch]
    rewards = [d['reward'] for d in mini_batch]
    next_states = [d['next_state'] for d in mini_batch]
    expected_q = []

    # Compute Q(s', a'; theta_{i-1}). This is an unbiased estimator for y_i as
    # in eqn 2 in the DQN paper.
    next_q = tf_sess.run(tf_output_layer, feed_dict={tf_input_layer : \
            next_states})

    for i in xrange(len(mini_batch)):
        if mini_batch[i]['terminal']:
            # This was a terminal frame
            expected_q.append(rewards[i])
        else:
            expected_q.append(rewards[i] + DISCOUNT_FACTOR * \
                    np.max(next_q[i]))

    one_hot_actions = compute_one_hot_actions(actions)

    # Learn that these actions in these states lead to this reward
    _, loss = tf_sess.run([tf_train_operation, tf_cost], feed_dict={
        tf_input_layer: states,
        tf_action: one_hot_actions,
        tf_target: expected_q})

    return loss

# Return a one hot vector with a 1 at the index for the action.
def compute_one_hot_actions(actions):
    one_hot_actions = []
    for i in xrange(len(actions)):
        one_hot = np.zeros([NUM_ACTIONS])
        one_hot[ACTIONS.index(actions[i])] = 1
        one_hot_actions.append(one_hot)
    return one_hot_actions

# Deep Q-learning
def deep_q_learn(restore_model="",
        checkpoint_path="tensorflow_checkpoints"):

    # Create tensorflow session
    tf_sess = tf.Session()

    # Create tensorflow network
    tf_input_layer, tf_output_layer = create_network()

    # A one-hot vector specifying the action
    tf_action = tf.placeholder("float", [None, NUM_ACTIONS])

    # The target for Q-learning. This is y_i in eqn 2 of the DQN paper.
    tf_target = tf.placeholder("float", [None])

    # The q-value for the specified action, where tf_action is a one-hot vector.
    tf_q_for_action = tf.reduce_sum(tf_output_layer * tf_action,
    reduction_indices=1)

    # The cost we try to minimise, as in eqn 2 of the DQN paper
    tf_cost = tf.reduce_mean(tf.square(tf_target - tf_q_for_action))

    # The train operation: reduce the cost using Adam
    tf_train_operation = \
            tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(tf_cost)

    tf_sess.run(tf.initialize_all_variables())

    epsilon_greedy = INITIAL_EPSILON_GREEDY

    # Give this run of the program an identifier
    identifier = str(time.gmtime()[0:5])
    identifier = identifier.replace('(', '').replace(')', '')
    identifier = identifier.replace(' ', '-').replace(',','')

    observations = deque()
    actions = []
    avg_q_history = []
    avg_reward_history = []
    episode_rewards = deque()

    # Set up plotting
    plot = False
    if plot:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

    env = gym.make('CartPole-v0')
    obs = env.reset()

    # Compute the first state
    current_state = obs

    losses = []

    # Enter loop over number of time steps
    t = 0
    episode_reward = 0
    episode_idx = 0
    while True:

        # At timestep OBSERVATION_STEPS, we generate some random states to test
        # the average q value
        if t == OBSERVATION_STEPS-1:
            benchmark_observations = random.sample(observations,
                    BENCHMARK_STATES)
            benchmark_states = [d['state'] for d in benchmark_observations]

        # Compute action
        action = compute_action(tf_sess, tf_input_layer, tf_output_layer,
                current_state, epsilon_greedy)

        # Take a step with action
        obs, reward, terminal, info = env.step(action)

        # Update the current and next states
        next_state = obs

        # Record transitions
        observations.append({'state': current_state, 'action':
            action, 'reward': reward, 'next_state': next_state,
            'terminal': terminal})

        episode_reward += 1

        if RENDER and ((episode_idx % RENDER_EVERY) == 0):
            env.render()

        # Compute average reward
        if terminal:
            episode_rewards.append(episode_reward)
            #print("reward for episode", episode_reward)
            episode_reward = 0
            episode_idx += 1
            if len(episode_rewards) > NONZERO_REWARD_MEMORY:
                episode_rewards.popleft()

        # Compute the average q-value, and display the average reward and
        # average q-value
        if (t >= OBSERVATION_STEPS) and (t % VERBOSE_EVERY_STEPS == 0):
            avg_q_value = compute_average_q_value(tf_sess, tf_input_layer,
                    tf_output_layer, benchmark_states)
            print("Time: {}. Average Q-value: {}".format(t, avg_q_value))
            avg_q_history.append(avg_q_value)

            print("Average reward: {}".format(np.mean(episode_rewards)))
            avg_reward_history.append(np.mean(episode_rewards))

            print("Epsilon: {}".format(epsilon_greedy))

            if len(losses) > 0:
                print("Loss: {}".format(losses[-1]))

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

        # Train if we have reached the number of observation steps
        if (t >= OBSERVATION_STEPS):
            loss = train(tf_sess, observations, tf_input_layer, tf_output_layer,
                    tf_train_operation, tf_action, tf_target, tf_cost)
            losses.append(loss)
    
        # Anneal epsilon for epsilon-greedy strategy
        if epsilon_greedy > FINAL_EPSILON_GREEDY and len(observations) > \
                OBSERVATION_STEPS:
            epsilon_greedy -= (INITIAL_EPSILON_GREEDY - FINAL_EPSILON_GREEDY) \
                    / EPSILON_GREEDY_STEPS

        # If terminal, then reset the environment
        if terminal:
            obs = env.reset()
            # Compute the first state again
            current_state = obs
        # Otherwise, update current_state
        else:
            current_state = next_state

        # Update t
        t += 1

# Compute the action predicted by the current parameters of the q network for
# the current state.
def compute_action(tf_sess, input_layer, output_layer, current_state,
        epsilon_greedy):
    # Choose an action randomly with probability epsilon_greedy.
    if random.random() <= epsilon_greedy:
        action_index = random.randrange(NUM_ACTIONS)
    # Otherwise, choose an action according to the Q-function
    else:
        q_function = tf_sess.run(output_layer, feed_dict={input_layer: \
            [current_state]})[0]
        action_index = np.argmax(q_function)

    # Return the action at action_index
    return ACTIONS[action_index]

# Compute the average q value for a given set of states. This can be used as an
# indicator of training progress.
def compute_average_q_value(tf_sess, tf_input_layer, tf_output_layer, states):
    # Compute the maximum q-value for each state
    tf_max_q_values = tf.reduce_max(tf_output_layer, reduction_indices=1)

    # Take the average of the maximum q-values
    tf_avg_q_values = tf.reduce_mean(tf_max_q_values)
    
    # Run the computation
    avg_q_values = tf_sess.run(tf_avg_q_values, feed_dict={tf_input_layer : \
        states})

    return avg_q_values

# Initialise the q network
def create_network():
    input_layer = tf.placeholder("float", [None, 4])

    num_hidden_1 = 5
    W1 = tf.Variable(tf.truncated_normal([4, num_hidden_1], stddev=1/np.sqrt(4)))
    b1 = tf.Variable(tf.constant(0.01, shape=[num_hidden_1]))
    hidden_1 = tf.nn.relu(tf.matmul(input_layer, W1) + b1)

    num_hidden_2 = 5
    W2 = tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2],
        stddev=1/np.sqrt(num_hidden_1)))
    b2 = tf.Variable(tf.constant(0.01, shape=[num_hidden_2]))
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([num_hidden_2, NUM_ACTIONS],
        stddev=1/np.sqrt(num_hidden_2)))
    b3 = tf.Variable(tf.constant(0.01, shape=[NUM_ACTIONS]))
    output_layer = tf.matmul(hidden_2, W3) + b3

    return input_layer, output_layer

# Run deep q learning
deep_q_learn()
