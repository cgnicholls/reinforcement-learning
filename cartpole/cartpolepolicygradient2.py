# coding: utf-8

# This code works using the policy-gradient method. It successfully gets to
# about 300 average reward.

import numpy as np
import random
import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

# Play with policy gradient agent, with given parameter vector
def policy_gradient_agent(num_episodes, max_episode_length, theta):
    rewards = []
    for i_episode in range(num_episodes):
        episode_reward = 0
        observation = env.reset()
        for t in range(max_episode_length):
            env.render()
            action = sample_policy(observation, theta)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                print("Reward for episode:", episode_reward)
                break

# We first try a vanilla policy gradient algorithm. Let tau denote a trajectory,
# i.e. a sample of actions and rewards until the terminal state, s_T. Let E_tau
# denote the expectation with respect to trajectories tau. Let grad_theta denote
# the gradient with respect to the parameter vector theta. Let R(tau) denote the
# total reward of trajectory tau. Then we estimate grad_theta(E_tau(R(tau))) by
# E_tau(sum_{t=0}^{T-1} grad_theta log pi(a_t | s_t; theta) R(tau)).
def train_policy_gradient_agent(num_episodes, max_episode_length, batch_size,
        learning_rate, gamma, regularisation, reward_to_exit=500, monitor=False,
        showPlot=True):
    # Initialise theta
    theta = np.random.randn(1,4)

    all_rewards = []
    if showPlot:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

    velocity = np.zeros((1,4), dtype='float32')

    last_avg_reward = 0

    if monitor:
        env.monitor.start('/tmp/cartpole-experiment-2')
    for i_episode in range(num_episodes):
        policy_gradients = []
        policy_gradient = 0
        batch_rewards = []
        for i_batch in range(batch_size):
            episode_rewards, episode_actions, episode_observations = run_episode(theta, max_episode_length)
            batch_rewards.append(sum(episode_rewards))

            # Now compute the policy gradient
            batch_i_gradient = compute_policy_gradient(episode_rewards,
                episode_actions, episode_observations, theta)

            policy_gradients.append(batch_i_gradient)

        mean_reward = np.mean(batch_rewards)
        policy_gradient = 0
        for i_batch in range(batch_size):
            policy_gradients[i_batch] = policy_gradients[i_batch] / float(batch_rewards[i_batch]) * (batch_rewards[i_batch] - mean_reward)
            policy_gradient = policy_gradient + 1.0 / batch_size * policy_gradients[i_batch]

        # If we are above reward_to_exit for two episodes, then stop training
        if (last_avg_reward > reward_to_exit) and (mean_reward >
                reward_to_exit):
            return theta
        last_avg_reward = mean_reward

        # Print the total reward for the episode
        if i_episode % 50 == 0:
            print("Reward", mean_reward, ", episode:", i_episode)

        if showPlot:
            all_rewards.append(mean_reward)
            ax1.clear()
            ax1.plot(all_rewards)
            plt.pause(0.0001)

        # Apply regularisation
        policy_gradient = policy_gradient - regularisation * sum(theta*theta)

        alpha = 30*learning_rate / (30 + i_episode)
        velocity = gamma * velocity + alpha * policy_gradient
        theta = theta + velocity

    if monitor:
        env.monitor.close()
    plt.show()

    # Finally, return the theta we find
    return theta

# observation and theta are both row vectors.
# We want to find theta such that observation . theta > 0 is a good predictor
# for the 'move right' action.
def policy(observation, theta):
    # We compute the dot product of our observation with theta, and then apply
    # the softmax function, to get the probability of moving right, denoted
    # probRight. The probability of moving left is then 1 - probRight.

    logit = np.dot(observation, np.transpose(theta))
    probRight = 1 / (1 + np.exp(-logit))
    return [1-probRight, probRight]

def sample_policy(observation, theta):
    probabilities = policy(observation, theta)
    rnd = np.random.rand()
    if rnd < probabilities[0]:
        return 0
    else:
        return 1

# This function computes the gradient of the policy with respect to theta.
def compute_policy_gradient(episode_rewards, episode_actions,
        episode_observations, theta):
    # We simply compute the gradient of log pi (a_t | s_t, theta) with respect
    # to theta, for each timestep t in the episode, then multiply by
    # episode_rewards, and take the sum.

    episode_length = len(episode_rewards)
    
    # By a hand calculation, the gradient of log pi(a_1 | obs, theta) is pi(a_2
    # | obs, theta), and that of log pi(a_2 | obs, theta) is -pi(a_1 | obs,
    # theta).
    gradient = np.zeros(theta.shape)
    pis = np.array([policy(obs, theta)[1] for obs in episode_observations])
    vector_grad = sum([(episode_actions[t]-pis[t]) * episode_observations[t] for
        t in xrange(episode_length)])
    return vector_grad * sum(episode_rewards)

def run_episode(theta, max_episode_length):
    observation = env.reset()
    episode_rewards = []
    episode_actions = []
    episode_observations = []
    episode_observations.append(observation)
    for t in xrange(max_episode_length):
        #env.render()
        action = sample_policy(observation, theta)
        observation, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        episode_observations.append(observation)
        episode_actions.append(action)
        if done:
            break
    return episode_rewards, episode_actions, episode_observations

def approximate_reward(theta, num_iters, max_episode_length):
    rewards = []
    for i in xrange(num_iters):
        episode_rewards = run_episode(theta, max_episode_length)[0]
        rewards.append(sum(episode_rewards))
    return float(sum(rewards)) / num_iters

theta = train_policy_gradient_agent(1000, 500, 200, 9e-3, 0.5, 1e-7, 400)

# We have now computed our parameter theta, and we run the agent to see how it
# does with a time limit of 100,000 steps.
print("Theta after training", theta)

policy_gradient_agent(10, int(1e3), theta)
