# coding: utf-8
# Implements the asynchronous advantage actor-critic algorithm.

import os
import sys, getopt
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import time, sleep, gmtime, strftime
import gym
import Queue
from custom_gym import CustomGym
from custom_gym_classic_control import CustomGymClassicControl
import random
from agentlstm import Agent

random.seed(100)

# FLAGS
T_MAX = 100000000
NUM_THREADS = 8
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
VERBOSE_EVERY = 50000
TESTING = False

I_ASYNC_UPDATE = 5

FLAGS = {"T_MAX": T_MAX, "NUM_THREADS": NUM_THREADS, "INITIAL_LEARNING_RATE":
INITIAL_LEARNING_RATE, "DISCOUNT_FACTOR": DISCOUNT_FACTOR, "VERBOSE_EVERY":
VERBOSE_EVERY, "TESTING": TESTING, "I_ASYNC_UPDATE": I_ASYNC_UPDATE}

training_finished = False

class Summary:
    def __init__(self, logdir, agent):
        with tf.variable_scope('summary'):
            summarising = ['episode_avg_reward', 'avg_value']
            self.agent = agent
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_ops = {}
            self.summary_vars = {}
            self.summary_ph = {}
            for s in summarising:
                self.summary_vars[s] = tf.Variable(0.0)
                self.summary_ph[s] = tf.placeholder('float32', name=s)
                self.summary_ops[s] = tf.summary.scalar(s, self.summary_vars[s])
            self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
            self.summary_op = tf.summary.merge(list(self.summary_ops.values()))
            
    def write_summary(self, summary, t):
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in summary.items()})
        summary_to_add = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in summary.items()})
        self.writer.add_summary(summary_to_add, global_step=t)

def async_trainer(agent, env, sess, thread_idx, T_queue, summary, saver,
    save_path):
    print "Training thread", thread_idx
    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    last_verbose = T
    last_time = time()
    last_target_update = T

    terminal = True
    while T < T_MAX:
        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            terminal = False
            state = env.reset()
            # Get the initial state of the RNN (which is just zeros for both c
            # and h)
            rnn_state = agent.rnn_state_init

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # Save the current state
            batch_states.append(state)
            
            # Choose an action randomly according to the policy
            # probabilities. We do this anyway to prevent us having to compute
            # the baseline value separately.
            policy, value, rnn_state = sess.run([agent.policy, agent.value,
            agent.rnn_state_out], feed_dict={ \
                agent.state: state,
                agent.rnn_state_in[0]: rnn_state[0],
                agent.rnn_state_in[1]: rnn_state[1]})

            # policy and value are currently arrays but we want scalars.
            policy = policy.flatten()
            value = value.flatten()
            value = value[0]

            # Choose the action according to the policy.
            action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal.
            state, reward, terminal, _ = env.step(action_idx)

            # Update counters
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)

            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value)

        target_value = 0
        # If the last state was terminal, just put R = 0. Else we want the
        # estimated value of the last state.
        if not terminal:
            target_value, rnn_state = sess.run([agent.value, agent.rnn_state_out],
            feed_dict={
                agent.state: state,
                agent.rnn_state_in[0]: rnn_state[0],
                agent.rnn_state_in[1]: rnn_state[1]})
            target_value = target_value.flatten()
            target_value = target_value[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)
        # Reverse the batch target values, so they are in the correct order
        # again.
        batch_target_values.reverse()

        # Test batch targets
        if TESTING:
            temp_rewards = batch_rewards + [last_R]
            test_batch_target_values = []
            for j in range(len(batch_rewards)):
                test_batch_target_values.append(discount(temp_rewards[j:], DISCOUNT_FACTOR))
            if not test_equals(batch_target_values, test_batch_target_values,
                1e-5):
                print "Assertion failed"
                print last_R
                print batch_rewards
                print batch_target_values
                print test_batch_target_values

        # Compute the estimated value of each state
        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)
        
        # Apply asynchronous gradient update
        initial_rnn_state = agent.rnn_state_init

        # Training
        sess.run(agent.train_op, feed_dict={
            agent.state: np.vstack(batch_states),
            agent.action: batch_actions,
            agent.target_value: batch_target_values,
            agent.advantages: batch_advantages,
            agent.rnn_state_in[0]: initial_rnn_state[0],
            agent.rnn_state_in[1]: initial_rnn_state[1]})

    global training_finished
    training_finished = True

def estimate_reward(agent, env, episodes=10, max_steps=10000):
    episode_rewards = []
    episode_vals = []
    t = 0
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False

        # Initialise the rnn state
        rnn_state = agent.rnn_state_init
        while not terminal:
            # Get the policy, value and next rnn state
            policy, value, rnn_state = agent.sess.run([agent.policy, agent.value,
            agent.rnn_state_out], feed_dict={
                agent.state: state,
                agent.rnn_state_in[0]: rnn_state[0],
                agent.rnn_state_in[1]: rnn_state[1]})

            # policy and value are currently arrays but we want scalars.
            policy = policy.flatten()
            value = value.flatten()
            value = value[0]

            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            t += 1
            episode_vals.append(value)
            episode_reward += reward
            if t > max_steps:
                episode_rewards.append(episode_reward)
                return episode_rewards, episode_vals
        episode_rewards.append(episode_reward)
    return episode_rewards, episode_vals

def evaluator(agent, env, sess, T_queue, summary, saver, save_path):
    # Read T and put the same T back on.
    T = T_queue.get()
    T_queue.put(T)
    last_time = time()
    last_verbose = T
    while T < T_MAX:
        T = T_queue.get()
        T_queue.put(T)
        if T - last_verbose >= VERBOSE_EVERY:
            print "T", T
            current_time = time()
            print "Train steps per second", float(T - last_verbose) / (current_time - last_time)
            last_time = current_time
            last_verbose = T
            
            print "Evaluating agent"
            episode_rewards, episode_vals = estimate_reward(agent, env, episodes=5)
            avg_ep_r = np.mean(episode_rewards)
            avg_val = np.mean(episode_vals)
            print "Avg ep reward", avg_ep_r, "Average value", avg_val

            summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_value': avg_val}, T)
            checkpoint_file = saver.save(sess, save_path, global_step=T)
            print "Saved in", checkpoint_file
        sleep(1.0)

# If restore is True, then start the model from the most recent checkpoint.
# Else initialise as usual.
def a3c(game_name, num_threads=8, restore=None, save_path='model'):
    processes = []
    envs = []
    for _ in range(num_threads+1):
        gym_env = gym.make(game_name)
        if game_name == 'CartPole-v0':
            env = CustomGymClassicControl(gym_env)
        else:
            print "Assuming ATARI game and playing with pixels"
            env = CustomGym(gym_env, game_name, nb_frames=1)
        envs.append(env)

    # Separate out the evaluation environment
    evaluation_env = envs[0]
    envs = envs[1:]

    with tf.Session() as sess:
        agent = Agent(session=sess,
        action_size=envs[0].action_size,
        model='mnih-lstm',
        optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))

        # Create a saver, and only keep 2 checkpoints.
        saver = tf.train.Saver(max_to_keep=2)

        T_queue = Queue.Queue()

        # Either restore the parameters or don't.
        if restore is not None:
            saver.restore(sess, save_path + '-' + str(restore))
            last_T = restore
            print "T was:", last_T
            T_queue.put(last_T)
        else:
            sess.run(tf.global_variables_initializer())
            T_queue.put(0)

        summary = Summary(save_path, agent)

        # Create a process for each worker
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue, summary, saver, save_path,)))

        # Create a process to evaluate the agent
        processes.append(threading.Thread(target=evaluator, args=(agent,
        evaluation_env, sess, T_queue, summary, saver, save_path,)))

        # Start all the processes
        for p in processes:
            p.daemon = True
            p.start()

        # Until training is finished
        while not training_finished:
            sleep(0.01)

        # Join the processes, so we get this thread back.
        for p in processes:
            p.join()

# Returns sum(rewards[i] * gamma**i)
def discount(rewards, gamma):
    return np.sum([rewards[i] * gamma**i for i in range(len(rewards))])

def test_equals(arr1, arr2, eps):
    return np.sum(np.abs(np.array(arr1)-np.array(arr2))) < eps

def main(argv):
    num_threads = NUM_THREADS
    game_name = 'SpaceInvaders-v0'
    save_path = None
    restore = None
    try:
        opts, args = getopt.getopt(argv, "hg:s:r:t:")
    except getopt.GetoptError:
        print "To run the OpenAI Gym game and save to the given save path: \
        a3c.py -g <game name> -s <save path> -t <num threads>"
        print "To resume training playing on the given OpenAI Gym game from \
        the given save path: python a3c.py -g <game name> -s <save path> -t \
        <num threads> -r <the T in save_path-T>."
    for opt, arg in opts:
        if opt == '-h':
            print "Options: -g <game name>, -s <save path>, -r (restore from \
            the save path given), -t <num threads>."
            sys.exit()
        elif opt == '-g':
            game_name = arg
        elif opt == '-s':
            save_path = arg
        elif opt == '-r':
            restore = int(arg)
        elif opt == '-t':
            num_threads = int(arg)
            print "Using", num_threads, "threads."
    if game_name is None:
        print "No game name specified, so playing", game_name
    if save_path is None:
        save_path = 'experiments/' + game_name + '/' + \
        strftime("%d-%m-%Y-%H:%M:%S/model", gmtime())
        print "No save path specified, so saving to", save_path
    if not os.path.exists(save_path):
        print "Path doesn't exist, so creating"
        os.makedirs(save_path)
    print "Using save path", save_path
    print "Using flags", FLAGS
    a3c(game_name, num_threads=NUM_THREADS, restore=restore,
    save_path=save_path)

if __name__ == "__main__":
    main(sys.argv[1:])
