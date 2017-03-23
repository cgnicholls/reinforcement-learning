# coding: utf-8
# Implements the asynchronous Q-learning algorithm.

# Need shared variables between threads

# Pseudocode (for each actor-learner thread):
# Assume global shared theta, theta_target and counter T = 0.
# Initialize thread step counter t <- 0
# Initialize target network weights theta_target <- theta
# Initialize network gradients dtheta <- 0
# Get initial state s
# repeat
#   take action a with epsilon-greedy policy based on Q(s,a;theta)
#   receive new state s' and reward r
#   y = r (for terminal s')
#     = r + gamma * max_a' Q(s', a', theta_target) (for non-terminal s')
#   accumulate gradients wrt theta: dtheta <- dtheta + grad_theta(y-Q(s,a;theta))^2
#   s = s'
#   T <- T + 1 and t <- t + 1
#   if T mod I_target == 0:
#       update target network theta_target <- theta
#   if t mod I_asyncupdate == 0 or s is terminal:
#       perform asynchronous update of theta using dtheta
#       clear gradients dtheta <- 0
# until T > T_max

import os
import sys
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import sleep
import gym
import Queue
import custom_gridworld as custom
from custom_gym import CustomGym
import random

random.seed(100)

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model

T_MAX = 100000000
ACTIONS = [0,1]
NUM_ACTIONS = len(ACTIONS)
NUM_THREADS = 8
STATE_FRAMES = 4
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
SKIP_ACTIONS = 4
VERBOSE_EVERY = 40000
EPSILON_STEPS = 4000000

I_TARGET = 40000
I_ASYNC_UPDATE = 5

training_finished = False

class Agent():
    def __init__(self, session, action_size, h, w, channels, optimizer=tf.train.AdamOptimizer(1e-4)):
        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session
        K.set_session(self.sess)

        with tf.variable_scope('online_network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.reward = tf.placeholder('float32', [None], name='reward')
            model, self.state, self.q_vals = self._build_model(h, w, channels)
            self.weights = model.trainable_weights

        with tf.variable_scope('optimizer'):
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            q_val = tf.reduce_sum(tf.multiply(self.q_vals, action_one_hot), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.reward - q_val))
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimizer.apply_gradients(grads_vars)
        with tf.variable_scope('target_network'):
            target_model, self.target_state, self.target_q_vals = self._build_model(h, w, channels)
            target_weights = target_model.trainable_weights
        with tf.variable_scope('target_update'):
            self.target_update = [target_weights[i].assign(self.weights[i]) for i in range(len(target_weights))]

    def update_target(self):
        self.sess.run(self.target_update)
    
    def get_q_vals(self, state):
        return self.sess.run(self.q_vals, {self.state: state}).flatten()

    def get_target_q_vals(self, state):
        return np.max(self.sess.run(self.target_q_vals,
        feed_dict={self.target_state: state}).flatten())

    # Train the network on the given states and rewards
    def train(self, states, actions, rewards):
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.reward: rewards
        })

    # Builds the DQN model as in Mnih.
    def _build_model(self, h, w, channels, hidden_size=256):
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        inputs = Input(shape=(h,w,channels,))
        model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same', dim_ordering='tf')(inputs)
        model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same', dim_ordering='tf')(model)

        model = Flatten()(model)
        model = Dense(output_dim=hidden_size, activation='relu')(model)
        out = Dense(output_dim=self.action_size, activation='linear')(model)
        model = Model(input=inputs, output=out)
        q_vals = model(state)
        return model, state, q_vals

class Summary:
    def __init__(self, logdir, agent):
        with tf.variable_scope('summary'):
            summarising = ['episode_avg_reward', 'avg_q_value']
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
        

def get_epsilon(global_step, epsilon_steps, epsilon_min):
    epsilon = 1.0 - float(global_step) / float(epsilon_steps) * (1.0 - epsilon_min)
    return epsilon if epsilon > epsilon_min else epsilon_min

def async_trainer(agent, env, sess, thread_idx, T_queue, summary):
    print "Training thread", thread_idx
    # Choose a minimum epsilon once and for all for this agent.
    Tq = T_queue.get()
    T_queue.put(Tq+1)
    epsilon_min = random.choice(4*[0.1] + 3*[0.01] + 3*[0.5])
    epsilon = get_epsilon(Tq, EPSILON_STEPS, epsilon_min)

    last_verbose = Tq
    last_target_update = Tq

    terminal = True
    while Tq < T_MAX:
        batch_states = []
        batch_rewards = []
        batch_actions = []

        if terminal:
            terminal = False
            state = env.reset()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            Tq = T_queue.get()
            T_queue.put(Tq+1)
            batch_states.append(state)
            
            if random.random() < epsilon:
                action_idx = random.randrange(agent.action_size)
            else:
                q_vals = agent.get_q_vals(state)
                action_idx = np.argmax(q_vals)

            # Take the action
            state, reward, terminal, _ = env.step(action_idx)
            reward = np.clip(reward, -1, 1)
            
            if not terminal:
                target_q_vals = agent.get_target_q_vals(state)
                reward += DISCOUNT_FACTOR * agent.get_target_q_vals(state)

            batch_rewards.append(reward)
            batch_actions.append(action_idx)

        # Apply asynchronous gradient update
        agent.train(np.vstack(batch_states), batch_actions, batch_rewards)

        # Anneal epsilon
        epsilon = get_epsilon(Tq, EPSILON_STEPS, epsilon_min)

        if thread_idx == 0:
            if Tq - last_target_update >= I_TARGET:
                print "Worker", thread_idx, "T", Tq, "Updating target"
                last_target_update = Tq
                agent.update_target()

            if Tq - last_verbose >= VERBOSE_EVERY and terminal:
                print "Worker", thread_idx, "T", Tq, "Evaluating agent"
                last_verbose = Tq
                episode_rewards, episode_qs = estimate_reward(agent, env, episodes=5)
                avg_ep_r = np.mean(episode_rewards)
                avg_q = np.mean(episode_qs)
                print "Avg ep reward", avg_ep_r, "epsilon", epsilon, "Average q", avg_q
                summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_q_value': avg_q}, Tq)
    global training_finished
    training_finished = True

def estimate_reward(agent, env, episodes=10):
    episode_rewards = []
    episode_qs = []
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            q_vals = agent.get_q_vals(state)
            state, reward, terminal, _ = env.step(np.argmax(q_vals))
            episode_qs.append(np.max(q_vals))
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return episode_rewards, episode_qs

def qlearn(game_name, nb_threads=8):
    processes = []
    envs = []
    for _ in range(nb_threads):
        gym_env = gym.make(game_name)
        env = CustomGym(gym_env)
        envs.append(env)

    T_queue = Queue.Queue()
    T_queue.put(0)

    with tf.Session() as sess:
        agent = Agent(session=sess, action_size=envs[0].action_size,
        h=84, w=84, channels=STATE_FRAMES,
        optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))
        sess.run(tf.global_variables_initializer())
        
        summary = Summary('tensorboard', agent)

        for i in range(NUM_THREADS):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue, summary,)))
        for p in processes:
            p.daemon = True
            p.start()

        while not training_finished:
            sleep(0.01)
        for p in processes:
            p.join()

qlearn('SpaceInvaders-v0')
