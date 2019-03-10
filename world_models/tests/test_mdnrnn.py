import numpy as np
import tensorflow as tf

from world_models.mdn_rnn import MDNRNN


class TestMDNRNN(tf.test.TestCase):

    def test_can_predict_mixture_density(self):
        input_dim = 10
        action_dim = 2
        hidden_dim = 12
        sequence_length = 20
        batch_size = 5
        net = MDNRNN(input_dim, action_dim, hidden_dim, sequence_length,
                     batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            batch_size = 5
            x = np.random.randn(batch_size, sequence_length, input_dim)
            a = np.random.randint(low=0, high=5, size=(batch_size, sequence_length))

            computed = net.predict(sess, x, a)

            assert computed.shape == (batch_size, sequence_length,
                                      input_dim + action_dim)