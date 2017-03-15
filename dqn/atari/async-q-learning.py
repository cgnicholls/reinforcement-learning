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

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import sleep
import gym

T_MAX = 1000
NUM_ACTIONS = 4
RESIZED_SCREEN_X = 80
RESIZED_SCREEN_Y = 80
STATE_FRAMES = 4
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99

I_TARGET = 100
I_ASYNC_UPDATE = 100

class NetworkDeepmind():
    def __init__(self, scope, trainer):
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.create_network()
            
            # We need ops for training for the worker networks.
            if scope != 'global_network':
                self.action = tf.placeholder(shape=[None], dtype=tf.int32)
                self.one_hot_actions = tf.one_hot(self.action, NUM_ACTIONS,
                dtype=tf.float32)

                # Need the target network to compute this.
                self.q_for_action = tf.reduce_sum(self.output_layer * self.one_hot_actions, reduction_indices=1)
                self.target = tf.placeholder(shape=[None], dtype=tf.float32)
                self.loss = tf.reduce_mean(tf.square(self.target - self.q_for_action))
                
                # Get the gradients from the worker network using losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)

                self.trainer = trainer

                # The operator for applying worker gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_network')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))

    def create_network(self):
        conv1_W = tf.Variable(tf.truncated_normal([8, 8, STATE_FRAMES, 32],
            stddev=0.01))
        conv1_b = tf.Variable(tf.constant(0.1, shape=[32]))

        conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
        conv2_b = tf.Variable(tf.constant(0.1, shape=[64]))

        conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
        conv3_b = tf.Variable(tf.constant(0.1, shape=[64]))

        fc1_W = tf.Variable(tf.truncated_normal([7*7*64, 512], stddev=0.1))
        fc1_b = tf.Variable(tf.constant(0.1, shape=[512]))
        
        fc2_W = tf.Variable(tf.truncated_normal([512, NUM_ACTIONS], stddev=0.1))
        fc2_b = tf.Variable(tf.constant(0.1, shape=[NUM_ACTIONS]))

        self.input_layer = tf.placeholder("float", [None, RESIZED_SCREEN_X,
            RESIZED_SCREEN_Y, STATE_FRAMES])

        conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, conv1_W,
            strides=[1,4,4,1], padding="SAME") + conv1_b) 

        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_W, strides=[1,2,2,1],
            padding="VALID") + conv2_b)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, conv3_W, strides=[1,1,1,1],
            padding="VALID") + conv3_b)

        flatten = tf.reshape(conv3, [-1, 7*7*64])

        fc1 = tf.nn.relu(tf.matmul(flatten, fc1_W) + fc1_b)

        self.output_layer = tf.matmul(fc1, fc2_W) + fc2_b

class Worker():
    def __init__(self, name, game_name, T, trainer):
        self.grad_theta = None
        self.t = 0
        self.T = T
        self.name = name
        self.local_network = NetworkDeepmind(self.name, trainer)
        self.increment_T = self.T.assign_add(1)
        self.trainer = trainer
        self.update_local_network_ops = copy_network_params('global_network', self.name)
        self.env = gym.make(game_name)
        self.actions = range(NUM_ACTIONS)

    # First step is to implement one thread and get it running on its own.
    def work(self, sess, coordinator):
        print "Starting worker " + self.name
    
        with sess.as_default(), sess.graph.as_default():
            while not coordinator.should_stop():
                # First make a copy of the global parameters for our worker's
                # network.
                sess.run(self.update_local_network_ops)

                # Add one to the global number of steps
                sess.run(self.increment_T)
                T = sess.run(self.T)

                # Start a new episode
                obs = self.env.reset()
                current_state = compute_state(None, obs)
                reward_history = []

                while True:
                    a = np.random.choice(self.actions)
                    
                    obs, reward, done, info = self.env.step(a)
                    next_state = compute_state(current_state, obs)

                    reward_history.append(reward)
                    if self.t % 100 == 0:
                        print "Average reward:", np.mean(reward_history)
                    y = reward
                    if not done:
                        # Compute Q-value wrt local theta
                        next_q = sess.run(self.local_network.output_layer, 
                        feed_dict={
                            self.local_network.input_layer: [next_state]
                        })
                        y += DISCOUNT_FACTOR * np.max(next_q)
                    
                    # Compute the gradient of the loss
                    sess.run(self.local_network.apply_grads, feed_dict={
                        self.local_network.input_layer: [current_state],
                        self.local_network.action: [a],
                        self.local_network.target: [y]
                    })

                    self.t += 1
                    if T > T_MAX:
                        return

# If current_state is None then just repeat the observation STATE_FRAMES times.
# Otherwise, remove the first frame, and append obs to get the new current
# state.
def compute_state(current_state, obs):
    # First preprocess the observation
    obs = preprocess(obs)

    if current_state is None:
        state = np.stack(tuple(obs for i in range(STATE_FRAMES)), axis=2)
    else:
        # obs is two-dimensional, so insert a dummy third dimension
        state = np.append(current_state[:,:,1:], obs[:,:,np.newaxis], axis=2)
    return state

# Preprocess the observation to remove noise.
def preprocess(obs):
    # Convert to float
    obs = obs.astype('float32')

    # Crop screen and set to grayscale
    obs = obs[34:194,:,0]

    # Downsize screen
    obs = obs[::2,::2]

    # Normalise input to 0-1 values
    obs = obs / 255.0 - 0.5

    return obs
            
# Copies the parameters from the from network to the to network.
def copy_network_params(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    ops = []
    for from_var,to_var in zip(from_vars, to_vars):
        ops.append(to_var.assign(from_var))
    return ops

def async_q_learn(game_name='Breakout-v0'):
    num_workers = multiprocessing.cpu_count()
    print "Using", num_workers, "workers"
    tf.reset_default_graph()

    T = tf.Variable(0, dtype=tf.int32, name='T')
    trainer = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE)
    global_network = NetworkDeepmind('global_network', None)

    # Create num_workers workers
    workers = []
    for i in range(num_workers):
        workers.append(Worker('worker_'+str(i), game_name, T, trainer))

    with tf.Session() as sess:
        coordinator = tf.train.Coordinator()

        # Initialise all variables
        sess.run(tf.global_variables_initializer())
        
        # Start the work of each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            work = lambda: worker.work(sess, coordinator)
            t = threading.Thread(target=(work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coordinator.join(worker_threads)

async_q_learn(game_name='Breakout-v0')
