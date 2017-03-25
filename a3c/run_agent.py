# coding: utf-8
# Given a save path for tensorflow to restore parameters for, run an agent on
# an OpenAI Gym environment.
from custom_gym import CustomGym
from agent import Agent
import os, getopt, sys
import gym
import tensorflow as tf
import numpy as np
from time import time, sleep

# Returns a tensorflow session with the 
def run_agent(save_path, T, game_name):
    with tf.Session() as sess:
        agent = Agent(session=sess, observation_shape=(210,160,3),
        action_size=3,
        optimizer=tf.train.AdamOptimizer(1e-4))

        # Create a saver, and only keep 2 checkpoints.
        saver = tf.train.Saver()

        saver.restore(sess, save_path + '-' + str(T))

        play(agent, game_name)

        return sess, agent

def play(agent, game_name, render=True, num_episodes=10, fps=5.0, monitor=True):
    gym_env = gym.make(game_name)
    if monitor:
        gym_env.monitor.start('videos/-v0')
    env = CustomGym(gym_env, game_name)

    desired_frame_length = 1.0 / fps

    episode_rewards = []
    episode_vals = []
    t = 0
    for ep in xrange(num_episodes):
        print "Starting episode", ep
        episode_reward = 0
        state = env.reset()
        terminal = False
        current_time = time()
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            if render:
                env.render()
            t += 1
            episode_vals.append(value)
            episode_reward += reward
            # Sleep so the frame rate is correct
            next_time = time()
            frame_length = next_time - current_time
            if frame_length < desired_frame_length:
                sleep(desired_frame_length - frame_length)
            current_time = next_time
        episode_rewards.append(episode_reward)
    if monitor:
        gym_env.monitor.close()
    return episode_rewards, episode_vals

def main(argv):
    save_path = None
    T = None
    game_name = None
    try:
        opts, args = getopt.getopt(argv, "g:s:T:")
    except getopt.GetoptError:
        print "Usage: python run_agent.py -g <game name> -s <save path> -T <T>"
    for opt, arg in opts:
        if opt == '-g':
            game_name = arg
        elif opt == '-s':
            save_path = arg
        elif opt == '-T':
            T = arg
    if game_name is None:
        print "No game name specified"
        sys.exit()
    if save_path is None:
        print "No save path specified"
        sys.exit()
    if T is None:
        print "No T specified"
        sys.exit()
    print "Reading from", save_path
    print "Running agent"
    run_agent(save_path, T, game_name)

if __name__ == "__main__":
    main(sys.argv[1:])
