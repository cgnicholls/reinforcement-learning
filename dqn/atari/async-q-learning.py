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

import sys
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import sleep
import gym
import Queue

T_MAX = 100000000
NUM_ACTIONS = 4
RESIZED_SCREEN_X = 80
RESIZED_SCREEN_Y = 80
STATE_FRAMES = 4
INITIAL_LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
SKIP_ACTIONS = 1
VERBOSE_EVERY = 10000
EPSILON_STEPS = 1000000

I_TARGET = 40000
I_ASYNC_UPDATE = 5

class NetworkDeepmind():
    def __init__(self, scope, trainer):
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.create_network()
            
            # We need ops for training for the worker networks.
            if scope != 'global_network' and scope != 'target_network':
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

                self.grads_and_vars = trainer.compute_gradients(self.loss, local_vars)

                # The operator for applying worker gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_network')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))

    def create_network_cart_pole(self, initial_stddev=1.0, initial_bias=0.1):
        self.input_layer = tf.placeholder("float", [None, 4])
        fc1_W = tf.Variable(tf.truncated_normal([4, 32],
        stddev=initial_stddev/np.sqrt(4)))
        fc1_b = tf.Variable(tf.constant(initial_bias, shape=[32]))
        fc1 = tf.nn.tanh(tf.matmul(self.input_layer, fc1_W) + fc1_b)

        fc2_W = tf.Variable(tf.truncated_normal([32, 2],
        stddev=initial_stddev/np.sqrt(32)))
        fc2_b = tf.Variable(tf.constant(initial_bias, shape=[2]))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        self.output_layer = fc2

    def create_network(self, initial_stddev=1.0, initial_bias=0.1):
        self.input_layer = tf.placeholder("float", [None, RESIZED_SCREEN_X,
            RESIZED_SCREEN_Y, STATE_FRAMES])

        fan_in1 = 8 * 8 * STATE_FRAMES
        conv1_W = tf.Variable(tf.truncated_normal([8, 8, STATE_FRAMES, 16],
            stddev=initial_stddev/np.sqrt(fan_in1)))
        conv1_b = tf.Variable(tf.constant(initial_bias, shape=[16]))
        conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, conv1_W,
            strides=[1,4,4,1], padding="SAME") + conv1_b) 

        fan_in2 = 4 * 4 * 16
        conv2_W = tf.Variable(tf.truncated_normal([4, 4, 16, 32],
        stddev=initial_stddev/np.sqrt(fan_in2)))
        conv2_b = tf.Variable(tf.constant(initial_bias, shape=[32]))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_W, strides=[1,2,2,1],
            padding="VALID") + conv2_b)

        flatten = tf.reshape(conv2, [-1, 2592])
        
        fan_in3 = 2592
        fc1_W = tf.Variable(tf.truncated_normal([2592, 256],
        stddev=initial_stddev/np.sqrt(fan_in3)))
        fc1_b = tf.Variable(tf.constant(initial_bias, shape=[256]))
        fc1 = tf.nn.relu(tf.matmul(flatten, fc1_W) + fc1_b)
        
        fan_in4 = 256
        fc2_W = tf.Variable(tf.truncated_normal([256, NUM_ACTIONS],
        stddev=initial_stddev/np.sqrt(fan_in4)))
        fc2_b = tf.Variable(tf.constant(initial_bias, shape=[NUM_ACTIONS]))

        self.output_layer = tf.matmul(fc1, fc2_W) + fc2_b

class Worker():
    def __init__(self, name, game_name, T_queue, ep_r_queue, trainer,
        target_network, global_network):
        self.grad_theta = None
        self.t = 0
        self.T_queue = T_queue
        self.ep_r_queue = ep_r_queue
        self.name = name
        self.local_network = NetworkDeepmind(self.name, trainer)
        self.target_network = target_network
        self.trainer = trainer
        self.update_local_network_ops = copy_network_params('global_network', self.name)
        self.update_target_op = copy_network_params('global_network', 'target_network')
        self.game_name = game_name
        self.env = gym.make(game_name)
        self.actions = range(NUM_ACTIONS)
        self.global_network = global_network

    # First step is to implement one thread and get it running on its own.
    def work(self, sess, coordinator):
        print "Starting worker " + self.name

        with sess.as_default(), sess.graph.as_default():
            # First make a copy of the global parameters for our worker's
            # network.
            sess.run(self.update_local_network_ops)
            # Also make sure that the target network is initialised the same
            # as the global network
            sess.run(self.update_target_op)

            # Start a new episode
            obs = self.env.reset()
            current_state = compute_state(None, obs)
            episode_rewards = []
            episode_reward = 0

            initial_epsilons = np.array([1.0, 1.0, 1.0])
            final_epsilons = np.array([0.1, 0.01, 0.5])
            epsilons = initial_epsilons

            batch_states = []
            batch_as = []
            batch_ys = []

            last_action = None
            while not coordinator.should_stop():

                epsilon = np.random.choice(epsilons, p=[0.4,0.3,0.3])

                while True:
                    is_training_step = self.t % SKIP_ACTIONS == 0 or last_action == None
                    Tq = self.T_queue.get()
                    if is_training_step:
                        # Add one to the global number of steps
                        self.T_queue.put(Tq+1)

                        # Update epsilons
                        epsilons = np.max([initial_epsilons - float(Tq)*(initial_epsilons-final_epsilons)/float(EPSILON_STEPS), final_epsilons], axis=0)
                        if np.random.rand() < epsilon:
                            a = np.random.choice(self.actions)
                        else:
                            qs = sess.run(self.local_network.output_layer, feed_dict={
                                self.local_network.input_layer: [current_state]
                            })
                            a = np.argmax(qs)
                        # Set last action
                        last_action = a
                    else:
                        a = last_action
                    
                    obs, reward, done, info = self.env.step(a)

                    next_state = compute_state(current_state, obs)

                    episode_reward += reward

                    if is_training_step:
                        y = reward
                        if not done:
                            # Compute Q-value wrt theta_target
                            next_q_target = sess.run(self.target_network.output_layer, feed_dict={
                                self.target_network.input_layer: [next_state]
                            })
                            next_q = sess.run(self.local_network.output_layer, 
                            feed_dict={
                                self.local_network.input_layer: [next_state]
                            })
                            y += DISCOUNT_FACTOR * np.max(next_q_target)
                        
                        # Add state, action and y to the batch
                        batch_states.append(current_state)
                        batch_as.append(a)
                        batch_ys.append(y)
                    
                    if len(batch_states) == I_ASYNC_UPDATE:
                        # Compute the gradient of the loss and update the global
                        # theta
                        sess.run(self.local_network.apply_grads, feed_dict={
                            self.local_network.input_layer: batch_states, 
                            self.local_network.action: batch_as,
                            self.local_network.target: batch_ys 
                        })

                        # Reset our batch
                        batch_states = []
                        batch_as = []
                        batch_ys = []

                        # Get a new copy of the global theta
                        sess.run(self.update_local_network_ops)

                    self.t += 1

                    if Tq % VERBOSE_EVERY == 0:
                        recent_ep_rewards = []
                        while not self.ep_r_queue.empty():
                            recent_ep_rewards.append(self.ep_r_queue.get())
                        if len(recent_ep_rewards) > 0:
                            print "T", Tq, self.name, "Episode rewards", np.mean(recent_ep_rewards)
                        else:
                            print "T", Tq, self.name, "No episode rewards yet"

                    #if self.t % VERBOSE_EVERY == 0 and len(episode_rewards) > 0:
                    #    print "T", Tq, self.name, "Average episode reward", np.mean(episode_rewards), "average last 5:", np.mean(episode_rewards[-5:])

                    if done:
                        obs = self.env.reset()
                        current_state = compute_state(None, obs)
                        episode_rewards.append(episode_reward)
                        self.ep_r_queue.put(episode_reward)
                        episode_reward = 0

                        # Choose new epsilon
                        epsilon = np.random.choice(epsilons, p=[0.4,0.3,0.3])
                    else:
                        current_state = next_state

                    if is_training_step and Tq % I_TARGET == 0 and Tq > 0:
                        print "Estimating value, T = ", Tq
                        #p = multiprocessing.Process(target=estimate_value, args=(sess, self.game_name, self.global_network))
                        #p.start()
                        #p.join()
                        estimated_v = estimate_value(sess, self.game_name, self.global_network)

                    if is_training_step and Tq % I_TARGET == 0:
                        print "T", Tq, "Updating target network"
                        sess.run(self.update_target_op)

                    if Tq > T_MAX:
                        return

                    if done:
                        break

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
    #num_workers = multiprocessing.cpu_count()
    num_workers = 16
    print "Using", num_workers, "workers"
    tf.reset_default_graph()

    trainer = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE)
    global_network = NetworkDeepmind('global_network', None)
    target_network = NetworkDeepmind('target_network', None)

    T_queue = Queue.Queue()
    T_queue.put(0)

    ep_r_queue = Queue.Queue()

    # Create num_workers workers
    workers = []
    for i in range(num_workers):
        workers.append(Worker('worker_'+str(i), game_name, T_queue, ep_r_queue, trainer,
        target_network, global_network))

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

# Estimate the value of the global parameters
def estimate_value(sess, game_name, global_network, max_episodes=5,
    max_steps=1000):
    env = gym.make(game_name)
    obs = env.reset()
    current_state = compute_state(None, obs)
    episode_rewards = []
    episode_reward = 0
    ep = 0
    t = 0
    while ep < max_episodes:
        qs = sess.run(global_network.output_layer, feed_dict={
            global_network.input_layer: [current_state]
        })
        a = np.argmax(qs)
        obs, reward, done, info = env.step(a)
        next_state = compute_state(current_state, obs)

        episode_reward += reward
        t += 1
        if t >= max_steps:
            break

        if done:
            obs = env.reset()
            current_state = compute_state(None, obs)
            episode_rewards.append(episode_reward)
            episode_reward = 0
            ep += 1
        else:
            current_state = next_state

    if len(episode_rewards) > 0:
        estimated_v = np.mean(episode_rewards)
    else:
        estimated_v = episode_reward
    print "Estimated value", estimated_v
    return estimated_v

if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_name = sys.argv[1]
    else:
        game_name = 'Pong-v0'
    print "Trying to play", game_name
    async_q_learn(game_name=game_name)
