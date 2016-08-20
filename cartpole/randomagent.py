import numpy as np
import random
import gym

# Make the CartPole environment
env = gym.make('CartPole-v0')

# Random agent
def random_agent(num_episodes):
    # Play num_episodes episodes
    for i_episode in range(num_episodes):
        # Initialise episode reward
        episode_reward = 0
        # Get the initial observation
        observation = env.reset()
        # Run for at most 1000 time steps
        for t in range(1000):
            # Render the environment
            env.render()
            # Randomly choose an action from the action space
            action = env.action_space.sample()
            # Take this action
            observation, reward, done, info = env.step(action)
            # Update episode reward
            episode_reward += reward
            # If the episode is over, print the episode reward
            if done:
                print("Total reward for episode: {}".format(episode_reward))
                break

# Play the random agent for 10 episodes
random_agent(10)
