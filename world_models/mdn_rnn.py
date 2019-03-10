import tensorflow as tf


class MDNRNN:

    def __init__(self, input_dim, action_dim, hidden_dim, sequence_length,
                 batch_size):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        with tf.variable_scope('mdn'):
            self.build_graph()

    def build_graph(self):
        self.input_state = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.sequence_length, self.input_dim],
            name='input_state')
        self.input_action = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.sequence_length],
            name='input_action')

        print("TEST", self.input_state.get_shape().as_list())

        one_hot_action = tf.one_hot(self.input_action, self.action_dim)

        input_rnn = tf.concat([self.input_state, one_hot_action], axis=2)

        print(input_rnn.get_shape().as_list())

        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_dim)
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        cell_output, cell_last_state = tf.nn.dynamic_rnn(
            cell, input_rnn, time_major=False, initial_state=initial_state,
            dtype=tf.float32)

        # For now ignore the MDN.
        self.output_layer = cell_output

        print(cell_output.get_shape().as_list())
        #
        # self.output_layer = tf.layers.dense(cell_output, self.input_dim,
        #                                     name='mixture_output')

    def predict(self, sess, z, a):
        return sess.run(self.output_layer, feed_dict={
            self.input_state: z,
            self.input_action: a
        })

