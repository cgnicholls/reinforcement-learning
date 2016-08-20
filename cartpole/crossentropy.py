# coding: utf-8

# This code works using the cross-entropy method. A write up is given on
# https://cgnicholls.github.io

import numpy as np
import random
import gym
env = gym.make('CartPole-v0')

# Play with cross entropy agent with the specified parameter vector
# num_episodes: the number of episodes to run
# max_episode_length: a cap on the maximum length of an episode
# theta: the parameter to use for the policy
# use_stochastic_policy: if false, then we move right if x dot theta > 0, and
# left otherwise; if true, then we move right with probability sigmoid(x dot
# theta), and move left otherwise. You get much better results by NOT using a
# stochastic policy
def cross_entropy_agent(num_episodes, max_episode_length, theta,
        use_stochastic_policy=False):
    rewards = []
    for i_episode in xrange(num_episodes):
        episode_reward = 0
        observation = env.reset()
        for t in xrange(max_episode_length):
            env.render()
            action = sample_action(observation, theta, use_stochastic_policy)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                print("Reward for episode: {t}".format(episode_reward))
                break

# Train the cross entropy agent.
# num_iterations: the number of iterations to use the method
# num_samples: the number of samples to take of theta
# elite_fraction: the fraction of samples to take the elite set from
# use_stochastic_policy: if false, then we move right if x dot theta > 0, and
# left otherwise; if true, then we move right with probability sigmoid(x dot
# theta), and move left otherwise. You get much better results by NOT using a
# stochastic policy
# num_iters_for_estimate: the number of iterations to run the estimate of the
# reward for each parameter vector
# max_episode_length: a cap on the maximum length of an episode
def train_cross_entropy_agent(num_iterations, num_samples, elite_fraction,
        use_stochastic_policy=False, verbose=True, num_iters_for_estimate=100,
        max_episode_length=5000):
    # Initialise mu and sigma2.
    mu = [0,0,0,0]
    sigma2 = [1,1,1,1]
    for iter in xrange(num_iterations):
        print("Iteration: {}".format(iter))
        sample_theta = []
        reward_estimates = []
        for j in xrange(num_samples):
            # Sample theta from the current distribution
            sample_theta.append(sample_from_gaussian(mu, sigma2))

            # Estimate the reward for this theta
            reward_estimates.append(estimate_reward_with_theta(sample_theta[j],
                    num_iters_for_estimate, max_episode_length,
                    use_stochastic_policy))

        # Print the average reward
        if verbose:
            print("Average reward: {}".format(np.mean(reward_estimates)))

        # Now keep the top elite_fraction fraction of parameters theta, as
        # measured by reward
        elite_set = compute_elite_set(sample_theta, reward_estimates,
                elite_fraction)
        
        # Now fit a diagonal Gaussian to this sample set
        [mu, sigma2] = fit_gaussian_to_samples(elite_set)

    # The best estimate for theta is the mean of the multivariate Gaussian
    return mu

# Return the best elite_fraction of sample_theta, when sorted by
# reward_estimates
def compute_elite_set(sample_theta, reward_estimates, elite_fraction):
    sample_theta = np.array(sample_theta)
    reward_estimates = np.array(reward_estimates)
    indices = reward_estimates.argsort()[::-1]
    sorted_sample_theta = sample_theta[indices]
    num_elite = int(elite_fraction * len(sample_theta))
    return sorted_sample_theta[1:num_elite]

# Estimate the expected reward when following the policy determined by the
# specified theta.
# theta: the parameter to use
# num_iters: the number of iterations to use to compute the average
# max_episode_length: the maximum episode length to consider
def estimate_reward_with_theta(theta, num_iters, max_episode_length,
        use_stochastic_policy):
    rewards = []
    for i in xrange(num_iters):
        episode_reward = 0
        observation = env.reset()
        for t in xrange(max_episode_length):
            action = sample_action(observation, theta, use_stochastic_policy)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        # Return the average reward
        return np.mean(rewards)

def sample_from_gaussian(mu, sigma2):
    return np.random.randn(1,4) * np.sqrt(sigma2) + mu

# Samples an action from the policy
# observation: an observation from the environment
# theta: the parameter vector theta
# Returns: a sample from the policy distribution. The distribution is: move
# right with probability sigma(x dot theta), and otherwise move left.
def sample_action(observation, theta, stochastic):
    if stochastic:
        prob_move_right = sigmoid(np.dot(observation, np.transpose(theta)))
        r = np.random.rand()
        if r < prob_move_right:
            return 1
        else:
            return 0
    else:
        if np.dot(observation, np.transpose(theta)) > 0:
            return 1
        else:
            return 0

# Computes the sigmoid function
# u: a real number
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

# Fit the maximum likelihood multivariate gaussian distribution given samples.
# samples: a 2D numpy array whose rows are samples from a multivariate gaussian
# with diagonal covariance matrix.
# Returns: the mean and variance as row vectors.
def fit_gaussian_to_samples(samples):
    samples = np.array(samples)
    mu = np.mean(samples,0)
    sigma2 = np.var(samples,0)
    return [mu, sigma2]

# Train the cross entropy agent on 5 iterations of the cross entropy method,
# using 100 samples each time, and only keeping the top 10%. We don't use a
# stochastic policy.
trained_theta = train_cross_entropy_agent(5, 100, 0.1, False)

# Run the agent with our computed theta
cross_entropy_agent(10, 10000, trained_theta)
