# coding: utf-8

# This code works using a vanilla version of the policy-gradient method. 

import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import pickle
env = gym.make('Pong-v0')

# Play with policy gradient agent, with given parameter vector
# - num_episodes: the number of episodes to run the agent
# - theta: the parameter to use for the policy
# - max_episode_length: the maximum length of an episode
def policy_gradient_agent(num_episodes, W1, W2, max_episode_length, render=True):
    for i_episode in range(num_episodes):
        episode_rewards, _, _ = run_episode(W1, W2, max_episode_length, render)
        print("Reward for episode:", sum(episode_rewards))

# Train an agent using policy gradients. Each episode, we sample a trajectory,
# and then estimate the gradient of the expected reward with respect to theta.
# We then update theta in the direction of the gradient.
# - num_episodes: the number of episodes to train for
# - max_episode_length: the maximum length of an episode
# - initial_step_size: the initial step size. We decrease the step size
# proportional to 1/n, where n is the episode number
def train_policy_gradient_agent(num_episodes, max_episode_length,
        initial_step_size, render=False):
    # Initialise W1, W2
    num_hidden = 10
    W1 = np.random.randn(num_hidden, 210*160) * 0.0001
    W2 = np.random.randn(1, num_hidden) * 0.0001

    win_history = []
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    # Gradient ascent with velocity
    v1 = np.zeros(np.shape(W1))
    v2 = np.zeros(np.shape(W2))

    gamma = 0.5

    for i_episode in range(num_episodes):
        # Run an episode with our current policy
        episode_rewards, episode_actions, episode_observations = \
        run_episode(W1, W2, max_episode_length, render)

        # Output episode rewards
        rewards = np.array(episode_rewards)
        won = len(rewards[rewards == 1])
        lost = len(rewards[rewards == -1])
        print("Won: {} / {}".format(won, won+lost))

        win_history.append(float(won))

        # Write W1, W2 to file every 100th episode
        if (i_episode % 100) == 0:
            name = 'W1W2-' + str(i_episode) + '.pckl'
            f = open(name, 'wb')
            pickle.dump([W1, W2], f)
            f.close()

        ax1.clear()
        ax1.plot(win_history)
        plt.pause(0.0001) 

        # Compute the policy gradient for this trajectory
        policy_gradient = compute_policy_gradient(episode_rewards,
                episode_actions, episode_observations, W1, W2)

        # Vanilla gradient ascent
        # We decrease the step size proportional to 1/i_episode
        step_size = initial_step_size / (1 + i_episode)

        v1 = gamma * v1 + step_size * policy_gradient[0]
        v2 = gamma * v2 + step_size * policy_gradient[1]
        W1 = W1 + v1
        W2 = W2 + v2

    # Return our trained theta
    return W1, W2

# observation and theta are both row vectors.
# We want to find theta such that observation . theta > 0 is a good predictor
# for the 'move right' action.
def compute_policy(state, W1, W2):
    # We first make the screen grayscale, and reshape to get a vector
    state = np.mean(state, 2)
    state = np.reshape(state, (-1, 1))

    # Compute first fully connected layer
    fc1 = np.dot(W1, state)

    # Apply relu
    relu1 = np.copy(fc1)
    relu1[relu1 < 0] = 0
    
    # Compute second fully connected layer
    fc2 = np.dot(W2, relu1)

    # Return the layer outputs
    return [state, fc1, relu1, fc2, sigmoid(fc2)]

# Samples an action from the policy
# observation: an observation from the environment
# theta: the parameter vector theta
# Returns: a sample from the policy distribution. The distribution is: move
# right with probability sigma(x dot theta), and otherwise move left.
def sample_action(observation, W1, W2):
    prob_up = compute_policy(observation, W1, W2)[-1]
    r = np.random.rand()
    if r < prob_up:
        return 2
    else:
        return 3

# Computes the sigmoid function
# u: a real number
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def compute_policy_gradient_one_step(state, W1, W2):
    layer_outputs = compute_policy(state, W1, W2)
    state = layer_outputs[0]
    fc1 = layer_outputs[1]
    relu1 = layer_outputs[2]
    fc2 = layer_outputs[3]
    softmax = layer_outputs[4]

    dpi_dfc2 = softmax * (1-softmax)

    dfc2_drelu1 = np.transpose(W2)

    dpi_drelu1 = dpi_dfc2 * dfc2_drelu1

    # We can now compute dpi_dW2
    dfc2_dW2 = np.transpose(relu1)
    dpi_dW2 = dpi_dfc2 * dfc2_dW2

    # Move on to dpi_dW1. First keep backpropagating.
    drelu1_dfc1 = np.ones(np.shape(fc1))
    drelu1_dfc1[fc1 < 0] = 0
    dpi_dfc1 = dpi_drelu1 * drelu1_dfc1

    # The (i,j,k) entry of dfc1_dW1 is d(fc1)_i / d(W1)_jk.
    dfc1_dW1 = np.zeros((np.shape(fc1)[0], np.shape(W1)[0], np.shape(W1)[1]))
    for i in xrange(np.shape(fc1)[0]):
        dfc1_dW1[i,i,:] = np.transpose(state)
    dpi_dW1 = np.tensordot(dpi_dfc1, dfc1_dW1, axes=(0,0))
    dpi_dW1 = np.reshape(dpi_dW1, np.shape(W1))

    return dpi_dW1, dpi_dW2, softmax

# This function computes the gradient of the policy with respect to theta for
# the specified trajectory.
# - episode_rewards: the rewards of the episode
# - episode_actions: the actions of the episode
# - episode_observations: the observations of the episode
# - theta: the parameter for the policy that ran the episode
def compute_policy_gradient(episode_rewards, episode_actions,
        episode_observations, W1, W2):
    # The gradient computation is explained at https://cgnicholls.github.io

    grad_W1_log_pi = np.zeros(np.shape(W1))
    grad_W2_log_pi = np.zeros(np.shape(W2))
    for t in xrange(len(episode_rewards)):
        state = episode_observations[t]
        grad_W1, grad_W2, policy = compute_policy_gradient_one_step(state, W1,
                W2)
        # Above, we've computed the gradient for going up. But if we actually
        # went down on this action, then we should compute grad log (1-pi),
        # which is (grad (1-pi)) / (1-pi) = -(grad pi) / (1-pi).
        if episode_actions[t] == 3:
            grad_W1 = -grad_W1
            grad_W2 = -grad_W2
            policy = 1-policy

        grad_W1_log_pi += grad_W1 / (1e-8 + policy)
        grad_W2_log_pi += grad_W2 / (1e-8 + policy)
    return grad_W1_log_pi * sum(episode_rewards), grad_W2_log_pi * \
sum(episode_rewards)

# Run an episode with the policy parametrised by theta.
# - theta: the parameter to use for the policy
# - max_episode_length: the maximum length of an episode
# - render: whether or not to show the episode
# Returns the episode rewards, episode actions and episode observations
def run_episode(W1, W2, max_episode_length, render=False):
    # Reset the environment
    observation = env.reset()
    episode_rewards = []
    episode_actions = []
    episode_observations = []
    episode_observations.append(observation)
    for t in xrange(max_episode_length):
        episode_observations.append(observation)
        # If rendering, draw the environment
        if render:
            env.render()

        # Set up the state
        if t > 0:
            state = episode_observations[t] - episode_observations[t-1]
        else:
            state = episode_observations[0]

        a_t = sample_action(state, W1, W2)
        episode_actions.append(a_t)
        observation, reward, done, info = env.step(a_t)
        episode_rewards.append(reward)
        if done:
            break
    return episode_rewards, episode_actions, episode_observations

def numerical_gradient(state, W1, W2, eps):
    grad_W1 = np.zeros(np.shape(W1))
    for i in xrange(np.shape(W1)[0]):
        for j in xrange(np.shape(W1)[1]):
            W1_plus = np.copy(W1)
            W1_plus[i,j] += eps
            W1_minus = np.copy(W1)
            W1_minus[i,j] -= eps
            grad_W1[i,j] = (compute_policy(state, W1_plus, W2)[-1] -
                    compute_policy(state, W1_minus, W2)[-1]) / (2*eps)
            if j % 1000 == 0:
                print("Completed {}/{}".format(j, np.shape(W1)[1]))

    grad_W2 = np.zeros(np.shape(W2))
    for i in xrange(np.shape(W2)[0]):
        for j in xrange(np.shape(W2)[1]):
            W2_plus = np.copy(W2)
            W2_plus[i,j] += eps
            W2_minus = np.copy(W2)
            W2_minus[i,j] -= eps
            grad_W2[i,j] = (compute_policy(state, W1, W2_plus)[-1] -
                    compute_policy(state, W1, W2_minus)[-1]) / (2*eps)
        print("Completed {}/{}".format(i, np.shape(W2)[0]))

    return grad_W1, grad_W2

# Test gradients
def test_gradient(eps):
    height = 10
    width = 20
    state = np.random.rand(height, width, 3)
    num_hidden = 2
    W1 = np.random.randn(num_hidden, height*width) * 0.001
    W2 = np.random.randn(1, num_hidden) * 0.001
    grad_W1_num, grad_W2_num = numerical_gradient(state, W1, W2, eps)
    grad_W1_an, grad_W2_an, _ = compute_policy_gradient_one_step(state, W1, W2)

    relative_error_W1 = relative_error(grad_W1_num, grad_W1_an, eps)
    relative_error_W2 = relative_error(grad_W2_num, grad_W2_an, eps)
    print("Relative error W1: {}".format(relative_error_W1))
    print("Relative error W2: {}".format(relative_error_W2))

def relative_error(arr1, arr2, eps):
    abs_error = np.sum(np.abs(np.reshape(arr1 - arr2, (1,-1))))
    norm1 = np.sum(np.abs(np.reshape(arr1, (1,-1))))
    norm2 = np.sum(np.abs(np.reshape(arr2, (1,-1))))
    return abs_error / min(1e-20 + norm1, 1e-20 + norm2)

# Test the gradients numerically
test_gradient(1e-6)

# Train the agent
num_episodes = 100000
max_episode_length = 2000
initial_step_size = 1e-4
W1, W2 = train_policy_gradient_agent(num_episodes, max_episode_length,
        initial_step_size, True)

# Run the agent for 10 episodes
policy_gradient_agent(10, W1, W2, max_episode_length)
