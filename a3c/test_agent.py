import tensorflow as tf
import numpy as np
from agent import Agent

def test_network():
    with tf.Session() as sess:
        # Create an agent
        agent = Agent(session=sess, action_size=3, channels=4,
        optimizer=tf.train.AdamOptimizer(1e-4))

        # Initialise all variables and then check the output of the network
        sess.run(tf.global_variables_initializer())

        print agent.layers
        trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope='network')
        print trainable_variables
    
test_network()
