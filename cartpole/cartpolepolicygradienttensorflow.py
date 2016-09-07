# coding: utf-8

# This code works using a vanilla version of the policy-gradient method,
# implemented in tensorflow.

import numpy as np
import tensorflow as tf
import random
import gym

NUM_ACTIONS = 2
NUM_INPUT = 4

TRAIN_FREQUENCY = 200
INITIAL_LEARNING_RATE = 1e-6

DISCOUNT_FACTOR = 0.99

def create_network():
    # Weights and bias
    W = tf.Variable(tf.truncated_normal([NUM_INPUT, NUM_ACTIONS],
        stddev=1/np.sqrt(NUM_INPUT)))
    b = tf.Variable(tf.constant(0.01, shape=[NUM_ACTIONS]))

    # Take the input
    input_layer = tf.placeholder("float", [None, NUM_INPUT])

    # Compute the matrix product
    logits = tf.matmul(input_layer, W) + b

    # Output the log probability of each action
    output_layer = tf.nn.log_softmax(logits)

    return input_layer, output_layer

def policy_gradient():
    # Get the network
    tf_input_layer, tf_log_policy = create_network()

    # Want to maximise the expected reward. Policy gradient method updates the
    # parameter theta by R * grad_theta log pi(a | s; theta). So we use R * log
    # pi(a | s; theta) as our function to maximise with tensorflow.

    # This holds the actions we took
    tf_actions = tf.placeholder("float", [None, NUM_ACTIONS])
    tf_rewards = tf.placeholder("float", [None])

    # Isolate the log policy for the action we took
    tf_log_policy_for_action = tf.reduce_mean(tf.mul(tf_actions, tf_log_policy),
            reduction_indices=1)

    # Weight the log policy by the reward for taking that action
    tf_target = tf.reduce_mean(tf.mul(tf_rewards, tf_log_policy_for_action))

    # Maximise the target
    tf_train_operation = \
        tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(-tf_target)

    # Create the session and initialise all variables
    tf_sess = tf.Session()

    tf_sess.run(tf.initialize_all_variables())

    # Make the environment
    env = gym.make('CartPole-v0')

    # Record the observations, rewards and log policies
    observations = []
    rewards = []
    actions = []
    terminal = []

    nonzero_rewards = []
    running_mean = 0

    # Get initial observation
    obs = env.reset()
    observations.append(np.array(obs))

    t = 0
    i_episode = 0
    while True:
        # Compute the log policy
        log_policy = tf_sess.run(tf_log_policy, feed_dict={tf_input_layer: [obs]})

        # Compute and store the action
        action = np.argmax(log_policy)
        actions.append(action)

        # Take a step with the environment, and get the reward for the agent
        obs, reward, done, info = env.step(action)

        # Store the reward and whether the frame was terminal
        rewards.append(reward)
        terminal.append(done)

        # Update nonzero rewards to compute an average
        if reward != 0:
            nonzero_rewards.append(reward)

        # Train the agent
        if (i_episode % TRAIN_FREQUENCY == 0) and (i_episode > 0):
            # Standardise the rewards
            discounted_rewards = discount_rewards(rewards, terminal,
                    DISCOUNT_FACTOR)
            standardised_rewards = np.array(discounted_rewards)
            standardised_rewards -= np.mean(standardised_rewards)
            standardised_rewards /= np.std(standardised_rewards)

            # Convert actions to one hot actions -- a vector with a one at the
            # index
            one_hot_actions = compute_one_hot_actions(actions)

            # Run a step of training
            tf_sess.run(tf_train_operation, feed_dict={
                tf_rewards: standardised_rewards,
                tf_actions: one_hot_actions,
                tf_input_layer: np.array(observations)})

            # Reset the observations, rewards and actions
            observations = []
            rewards = []
            actions = []
            terminal = []

        # If the frame was terminal, then reset the environment
        if done:
            obs = env.reset()
            i_episode += 1
            running_mean = 0.9 * running_mean + 0.1 * t
            if i_episode % 100 == 0:
                print("Reward for episode: {}".format(running_mean))
            t = 0
            #print("Reward for episode: {}".format(np.sum(nonzero_rewards)))
            nonzero_rewards = []

        # Store the observation -- this ensures rewards and observations have
        # the same size when training
        observations.append(np.array(obs))

        # Update time
        t += 1

# Return a one hot vector with a 1 at the index for the action.
def compute_one_hot_actions(actions):
    one_hot_actions = []
    for i in xrange(len(actions)):
        one_hot = np.zeros([NUM_ACTIONS])
        one_hot[actions[i]] = 1
        one_hot_actions.append(one_hot)
    return one_hot_actions

# Compute discounted rewards. The array terminal contains True/False values, and
# indicates if the frame is terminal
def discount_rewards(rewards, terminal, discount_factor):
    discounted_rewards = np.zeros_like(rewards).astype('float32')

    running_sum = 0
    for i in reversed(range(len(rewards))):
        if terminal[i]:
            running_sum = 0
        running_sum = running_sum * discount_factor + rewards[i]
        discounted_rewards[i] = running_sum
    return discounted_rewards

def test_discount_rewards():
    actual = discount_rewards([1,-1,3,1,1,1,2,-5], [0,0,1,0,1,0,0,0], 0.5)
    expected = np.array([1-0.5+3*0.25, -1+3*0.5, 3, 1+0.5, 1, 1+0.5*2+0.25*-5,
        2+0.5*-5,-5])
    assert((actual==expected).all())

# Run tests
test_discount_rewards()

# Train the agent
policy_gradient()
