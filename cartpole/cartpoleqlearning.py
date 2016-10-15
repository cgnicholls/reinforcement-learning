# coding: utf-8

# This code works using Q-learning. A write up is given on
# https://cgnicholls.github.io

import numpy as np
import random
import gym
env = gym.make('CartPole-v0')

# Need a q function that takes a state and gives out q-values for each
# action. It should have parameters that we tune. The loss function is the
# squared difference between the observed q-value and the predicted q-value.

# state: a 1x4 row vector denoting the state
# theta: a 4x2 real matrix.
# The q function is np.dot(state, theta).
def q_function(state, theta):
    return np.dot(state, theta)

# Compute the loss and also dloss_dtheta with respect to a transition.
def compute_gradient(current_state, action, reward, next_state, terminal, theta,
        discount=0.9):
    # Let q = q(current_state, action)
    # Let q' = reward + discount * max( q(next_state) ), where the max is taken
    # over all actions.
    # The loss function is: L = (q'_a - q_a)^2, where a is the action in the
    # transition.
    # We minimise this loss using gradient descent. We have to compute
    # dL/dtheta. We treat q' as ground truth, so we don't differentiate with
    # respect to it. However, q is a function of theta, so we have
    # dL/dtheta = -2 (q'_a-q_a) dq_a/dtheta.
    # Now, q(s,theta) = s * theta. Also, (s*theta)_a = sum_l s_l theta_{la}.
    # Hence, grad_theta_{ij}(q_a) = s_i delta_{ja}.

    # Thus d/dtheta_{ij}(L) = -2 (q'_a - q_a) * s_i * delta_{aj}.
    # So grad_theta(L)[:,a] = -2 (q'_a - q_a) * transpose(s), and the other
    # columns are zero.

    q_dash = reward + discount * max(q_function(next_state, theta))
    
    # Think this is the right thing to do here:
    if terminal:
        q_dash = reward
    q = q_function(current_state, theta)[action]

    loss = (q_dash - q)**2

    dloss_dtheta = np.zeros(np.shape(theta))
    dloss_dtheta[:,action] = -2 * np.transpose(current_state) * (q_dash - q)

    return loss, dloss_dtheta

# We use a q-function of the form q(s,a) = s * theta. Here s is a 1x4 vector and
# theta is a 4x2 vector. Thus q(s,a) is 1x2, and the column indexes the action.
def initialise_q(mu=0.0, sigma2=1e-1):
    return np.random.randn(4,2) * np.sqrt(sigma2) + mu

def sample_environment(sample_size, theta, epsilon):
    samples = []
    current_state = np.array(env.reset())
    lengths = []
    t = 0
    for i in xrange(sample_size):
        if np.random.random() < epsilon:
            action = np.random.randint(2)
        else:
            q = q_function(current_state, theta)
            action = np.argmax(q)
        next_state, reward, terminal, info = env.step(action)
        next_state = np.array(next_state)
        if terminal:
            reward = 0.0
        samples.append({'current_state': current_state, 'action': action,
            'reward': reward, 'next_state': next_state, 'terminal': terminal})
        current_state = next_state
        t += 1

        # If the state is terminal, then reset the environment
        if terminal:
            current_state = env.reset()
            lengths.append(t)
            t = 0
    return samples, np.mean(lengths)

# Train the Q-learning agent.
def train_q_learning_agent(max_iterations, sample_size, initial_step_size=1e-2,
        epsilon=0.9):
    # Initialise the parameter for q
    theta = initialise_q()

    for t in xrange(max_iterations):
        # Sample from the environment
        experience, avg_reward = sample_environment(sample_size, theta, epsilon)

        # Update theta with this experience
        losses = []
        dloss_dthetas = []
        for sample in experience:
            loss, dloss_dtheta = compute_gradient(sample['current_state'],
                    sample['action'], sample['reward'], sample['next_state'],
                    sample['terminal'], theta)
            losses.append(loss)
            dloss_dthetas.append(dloss_dtheta)

        a = initial_step_size * 10.0
        b = 10.0
        step_size = a / (b+t)
        epsilon -= 0.001
        epsilon = max(epsilon, 0.05)

        avg_dloss_dtheta = np.mean(dloss_dthetas, axis=0)
        theta = theta - dloss_dtheta * step_size

        if t % 10 == 0:
            print("Average loss: {}".format(np.mean(losses)))
            print("Average reward: {}".format(avg_reward))
            print("Epsilon: {}".format(epsilon))
            print("Step size: {}".format(step_size))
            estimated_reward = estimate_reward_with_theta(theta, 100, 1000)
            print("Estimated reward: {}".format(estimated_reward))
            
    return theta

# Estimate the expected reward when following the policy determined by the
# specified theta.
# theta: the parameter to use
# num_iters: the number of iterations to use to compute the average
# max_episode_length: the maximum episode length to consider
def estimate_reward_with_theta(theta, num_iters, max_episode_length,
        render=False):
    rewards = []
    for i in xrange(num_iters):
        episode_reward = 0
        observation = env.reset()
        for t in xrange(max_episode_length):
            action = np.argmax(q_function(observation, theta))
            if render:
                env.render()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)

    # Return the average reward
    return np.mean(rewards)

# Train using q-learning
theta = train_q_learning_agent(10000, 5000)

# Estimate the reward
estimated_reward = estimate_reward_with_theta(theta, 100, 10000, True)
print("Estimated reward: {}".format(estimated_reward))
