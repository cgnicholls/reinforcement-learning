import numpy as np
import gym
import tensorflow as tf
from random import sample, random
from collections import deque

def fully_connected(inputs, num_outputs, activation_fn, l2_reg=0.1):
    return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=num_outputs, activation_fn=activation_fn, weights_initializer=tf.contrib.layers.xavier_initializer(),
    biases_initializer=tf.zeros_initializer(),
    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

def create_network(input_dim, hidden_dims, output_dim, scope):
    with tf.variable_scope(scope):
        input_layer = tf.placeholder('float32', [None, input_dim])

        hidden_layer = input_layer
        for dim in hidden_dims:
            hidden_layer = fully_connected(inputs=hidden_layer, num_outputs=dim, activation_fn=tf.nn.relu)
            #hidden_layer = tf.nn.dropout(hidden_layer, 1.0)

        output_layer = fully_connected(inputs=hidden_layer, num_outputs=output_dim, activation_fn=None)
        return input_layer, output_layer

# Update the weights in to_scope to the ones from from_scope.
def update_ops_from_to(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

    ops = []
    for from_var, to_var in zip(from_vars, to_vars):
        ops.append(tf.assign(to_var, from_var))
    return ops

def qlearning(env, input_dim, num_actions, max_episodes=100000, update_target_every=2000, min_transitions=10000, batch_size=128, discount=0.9):
    transitions = deque()

    # Create the current network as well as the target network.
    hidden_dims = []
    input_layer, output_layer = create_network(input_dim, hidden_dims, num_actions, 'current')
    target_input_layer, target_output_layer = create_network(input_dim, hidden_dims, num_actions, 'target')
    update_ops = update_ops_from_to('current', 'target')

    tf_q_values = output_layer
    tf_action = tf.placeholder('int32', [None])
    tf_one_hot_action = tf.one_hot(tf_action, num_actions)
    tf_q_for_action = tf.reduce_sum(tf.multiply(tf_one_hot_action, tf_q_values), reduction_indices=1)
    tf_next_q = tf.placeholder('float32', [None])
    tf_reward = tf.placeholder('float32', [None])
    tf_loss = tf.reduce_mean(tf.square(tf_q_for_action - tf_reward - tf_next_q))

    tf_train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(tf_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sess.run(update_ops)

    all_rewards = []
    epsilon = 1.0

    for ep in xrange(max_episodes):
        state = env.reset()
        terminal = False
        epsilon -= 0.0001
        if epsilon <= 0.01:
            epsilon = 0.01

        t = 0
        while not terminal:
            # Given the state, we predict the action to be the one with largest q-value
            q_values = sess.run(output_layer, feed_dict={
                input_layer: [state]
            })
            action = np.argmax(q_values)
            if random() < epsilon:
                action = np.random.choice(num_actions)

            next_state, reward, terminal, _ = env.step(action)
            transition = {'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'terminal': terminal}

            transitions.append(transition)
            state = next_state
            t += 1
            if t > 1000:
                print "Maximum episode length"
                break
        all_rewards.append(t)

        # Only train if we have enough transitions
        if len(transitions) >= min_transitions:
            samples = sample(transitions, batch_size)
            states = [d['state'] for d in samples]
            next_states = [d['next_state'] for d in samples]
            rewards = [d['reward'] for d in samples]
            actions = [d['action'] for d in samples]
            not_terminals = np.array([not d['terminal'] for d in samples], 'float32')

            next_qs = sess.run(target_output_layer, feed_dict={
                target_input_layer: next_states
            })
            max_next_qs = np.amax(next_qs, axis=1)

            target_qs = discount * max_next_qs * not_terminals

            _, loss = sess.run([tf_train_op, tf_loss], feed_dict={
                input_layer: states,
                tf_reward: rewards,
                tf_next_q: target_qs,
                tf_action: actions
            })
            if ep % 100 == 0:
                print "Loss:", loss
                print "Average ep length", np.mean(all_rewards)
                print "Epsilon", epsilon
                all_rewards = []

            if ep % update_target_every == 0:
                print "Updating target"
                sess.run(update_ops)


qlearning(gym.make('CartPole-v0'), 4, 2)
