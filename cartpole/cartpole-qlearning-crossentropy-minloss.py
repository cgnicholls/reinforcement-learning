# coding: utf-8

# This code works using the cross-entropy method to find a Q-function satisfying
# the Bellman equation. We collect many transitions (s, a, r, s'), and store
# them in some big set Sigma. Then we use the cross-entropy method to minimise
# \sum( Q(s,a) - r - max_{a'} Q(s', a') ) / # \Sigma.

# Note that with a linear function approximator, we don't get good convergence
# with this method.

import numpy as np
import random
import gym
env = gym.make('CartPole-v0')

# Play with cross entropy agent with the specified parameter vector
# num_episodes: the number of episodes to run
# max_episode_length: a cap on the maximum length of an episode
# theta: the parameter to use for the policy
def cross_entropy_q_agent(num_episodes, max_episode_length, theta):
    rewards = []
    for i_episode in xrange(num_episodes):
        episode_reward = 0
        observation = env.reset()
        for t in xrange(max_episode_length):
            env.render()
            action = select_action(observation, theta)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                print("Reward for episode: {}".format(episode_reward))
                break

# Playing with a random policy, sample num_trajectories trajectories:
# (s,a,r,s',t), where s is a state, a is the action taken in state a, r is the
# observed reward, s' is the state the agent is transitioned to, and t is a
# boolean which is true exactly when the state was terminal.
def sample_trajectories(num_trajectories):
    trajectories = []
    obs = env.reset()
    for i_trajectory in xrange(num_trajectories):
        # Select an action randomly and step the environment
        action = np.random.randint(2)
        next_obs, reward, done, info = env.step(action)

        trajectories.append({"current_state": obs, "action": action, "reward":
            reward, "next_state": next_obs, "terminal": done})

        obs = next_obs
        if done:
            obs = env.reset()
    return trajectories

# Train the cross entropy agent.
# - max_iterations: the maximum number of iterations to use the method
# - num_samples: the number of samples to take of theta
# - elite_fraction: the fraction of samples to take the elite set from
# - reward_goal: exit the function when we reach this average reward
# - initial_variance: the initial variance for the distribution -- increase this
# to encourage the samples, theta, to be larger in magnitude, and thus be more
# deterministic.
# - num_iters_for_estimate: the number of iterations to run the estimate of the
# reward for each parameter vector
# - max_episode_length: a cap on the maximum length of an episode
def train_cross_entropy_q_agent(max_iterations, num_samples, elite_fraction,
        num_trajectories=1000, verbose=True, initial_variance=10,
        num_iters_for_estimate=100, max_episode_length=5000):

    # Initialise mu and sigma2.
    num_params = 5
    mu = np.zeros((num_params,2))
    sigma2 = np.full((num_params,2), initial_variance)
    for iter in xrange(max_iterations):
        # First collect all the trajectories
        trajectories = sample_trajectories(num_trajectories)
        print("Iteration: {}".format(iter))
        sample_theta = []
        loss_estimates = []
        reward_estimates = []
        for j in xrange(num_samples):
            # Sample theta from the current distribution
            sample_theta.append(sample_from_gaussian(mu, sigma2))

            # Estimate the reward for this theta
            loss_estimates.append(estimate_loss_with_theta(sample_theta[j],
                trajectories))
            
            # Estimate reward for this theta
            reward_estimates.append(estimate_reward_with_theta(sample_theta[j], num_iters_for_estimate, max_episode_length))

        # Print the average reward
        average_loss = np.mean(loss_estimates)
        average_reward = np.mean(reward_estimates)
        if verbose:
            print("Average loss: {}".format(average_loss))
            print("Average reward: {}".format(average_reward))

        # Now keep the top elite_fraction fraction of parameters theta, as
        # measured by reward
        elite_set = compute_elite_set(sample_theta, loss_estimates,
                elite_fraction)
        
        # Now fit a diagonal Gaussian to this sample set
        [mu, sigma2] = fit_gaussian_to_samples(elite_set)

    # The best estimate for theta is the mean of the multivariate Gaussian
    return mu

# Return the best elite_fraction of sample_theta, when sorted by loss_estimates
def compute_elite_set(sample_theta, loss_estimates, elite_fraction):
    sample_theta = np.array(sample_theta)
    loss_estimates = np.array(loss_estimates)

    # Sort the loss estimates in increasing order
    indices = loss_estimates.argsort()

    # Find the theta corresponding to these estimates
    sorted_sample_theta = sample_theta[indices]

    # Return only the elite fraction of theta
    num_elite = int(elite_fraction * len(sample_theta))
    return sorted_sample_theta[1:num_elite]

# Estimate the loss for the Q-function with parameter theta using the given
# trajectories
# theta: the parameter to use
# trajectories: a list of trajectories (s,a,r,s',t) as above
def estimate_loss_with_theta(theta, trajectories, discount=0.9):
    losses = []
    for trajectory in trajectories:
        qdash = np.max(q_function(trajectory["next_state"], theta))
        if trajectory["terminal"]:
            target = trajectory["reward"]
        else:
            target = trajectory["reward"] + discount * qdash
        q = q_function(trajectory["current_state"], theta)[trajectory["action"]]
        losses.append((q - target)**2)
    return np.mean(losses)

# Estimate the expected reward when following the policy determined by the
# specified theta.
# theta: the parameter to use
# num_iters: the number of iterations to use to compute the average
# max_episode_length: the maximum episode length to consider
def estimate_reward_with_theta(theta, num_iters, max_episode_length):
    rewards = []
    for i in xrange(num_iters):
        episode_reward = 0
        observation = env.reset()
        for t in xrange(max_episode_length):
            action = select_action(observation, theta)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        # Return the average reward
        return np.mean(rewards)

def sample_from_gaussian(mu, sigma2):
    return np.random.randn(*sigma2.shape) * np.sqrt(sigma2) + mu

# Selects the action with the largest q-value
# observation: an observation from the environment (1x4)
# theta: the parameter vector theta (4x2)
# Returns: the action with the largest q-value.
def select_action(observation, theta):
    return np.argmax(q_function(observation, theta))

def q_function(observation, theta):
    W = theta[:4,:]
    b = theta[4,:]
    return np.dot(observation, W) + b

# Fit the maximum likelihood multivariate gaussian distribution given samples.
# samples: a 2D numpy array whose rows are samples from a multivariate gaussian
# with diagonal covariance matrix.
# Returns: the mean and variance as row vectors.
def fit_gaussian_to_samples(samples):
    samples = np.array(samples)
    mu = np.mean(samples,0)
    sigma2 = np.var(samples,0)
    return [mu, sigma2]

# Train the agent using at most 10 iterations of the cross entropy method,
# using 100 samples each time, and only keeping the top 10%.
trained_theta = train_cross_entropy_q_agent(50, 200, 0.1)

print("Trained theta: {}".format(trained_theta))

# Run the agent with our computed theta
cross_entropy_q_agent(10, 10000, trained_theta)
