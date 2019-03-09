import tensorflow as tf


class VAE:

    def __init__(self, stride=2, latent_dim=32, kl_tolerance=0.5):
        self.stride = stride
        self.latent_dim = latent_dim
        self.kl_tolerance = kl_tolerance # From Ha's implementation.

        self.build_graph()

    def build_graph(self):
        with tf.variable_scope('vae'):
            self.input_layer = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
            batch_size = tf.shape(self.input_layer)[0]
            next = self.input_layer

            # Encoder
            next = tf.layers.conv2d(next, filters=32, kernel_size=4, strides=self.stride, activation=tf.nn.relu,
                                    name='encoder_conv1')
            next = tf.layers.conv2d(next, filters=64, kernel_size=4, strides=self.stride, activation=tf.nn.relu,
                                    name='encoder_conv2')
            next = tf.layers.conv2d(next, filters=128, kernel_size=4, strides=self.stride, activation=tf.nn.relu,
                                    name='encoder_conv3')
            next = tf.layers.conv2d(next, filters=256, kernel_size=4, strides=self.stride, activation=tf.nn.relu,
                                    name='encoder_conv4')
            next = tf.layers.flatten(next)

            # VAE part
            self.mu = tf.layers.dense(next, self.latent_dim, name='mu')
            self.log_var = tf.layers.dense(next, self.latent_dim, name='log_var')
            self.sigma = tf.exp(self.log_var / 2.0)
            self.white_noise = tf.random_normal([batch_size, self.latent_dim])

            self.z = self.mu + self.sigma * self.white_noise

            # Decoder
            next = tf.layers.dense(self.z, 1024, name='decoder_fc')
            next = tf.reshape(next, [-1, 1, 1, 1024])
            next = tf.layers.conv2d_transpose(next, 128, 5, strides=2, activation=tf.nn.relu, name='decoder_deconv1')
            next = tf.layers.conv2d_transpose(next, 64, 5, strides=2, activation=tf.nn.relu, name='decoder_deconv2')
            next = tf.layers.conv2d_transpose(next, 32, 6, strides=2, activation=tf.nn.relu, name='decoder_deconv3')
            self.output_layer = tf.layers.conv2d_transpose(next, 3, 6, strides=2, activation=tf.nn.sigmoid,
                                                           name='decoder_deconv4')

            # Reconstruction loss
            self.reconstruction_loss = tf.reduce_sum(tf.square(self.input_layer - self.output_layer),
                                                     reduction_indices=[1, 2, 3])
            self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss)

            # KL loss
            self.kl_loss = -0.5 * tf.reduce_sum(
                1 + self.log_var - tf.square(self.mu) - tf.exp(self.log_var),
                reduction_indices=1
            )
            self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.latent_dim)
            self.kl_loss = tf.reduce_mean(self.kl_loss)

            self.loss = self.reconstruction_loss + self.kl_loss

            self.learning_rate = tf.placeholder(dtype=tf.float32)
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(
                self.grads, global_step=self.global_step, name='train_op'
            )

            self.init_op = tf.global_variables_initializer()

            tf.summary.scalar('loss', self.loss, collections=['vae_summary'])
            tf.summary.scalar('reconstruction_loss', self.reconstruction_loss, collections=['vae_summary'])
            tf.summary.scalar('kl_loss', self.kl_loss, collections=['vae_summary'])
            tf.summary.scalar('learning_rate', self.learning_rate, collections=['vae_summary'])
            self.merged_summary = tf.summary.merge_all(key='vae_summary')

    def initialise(self, sess):
        sess.run(self.init_op)

    def train(self, sess, xs, learning_rate=1e-4):
        """Trains on the given xs. The xs should be a list of xs of the correct input shape.
        """
        assert xs.shape == (xs.shape[0], 64, 64, 3)

        loss, _, summary = sess.run([self.loss, self.train_op, self.merged_summary], feed_dict={
            self.input_layer: xs,
            self.learning_rate: learning_rate
        })
        return loss, summary

    def encode(self, sess, xs):
        assert xs.shape == (xs.shape[0], 64, 64, 3)
        return sess.run(self.z, feed_dict={self.input_layer: xs})

    def encode_to_mu_sigma(self, sess, xs):
        """Returns mu, sigma of the encoding.
        """
        assert xs.shape == (xs.shape[0], 64, 64, 3)
        return sess.run([self.mu, self.sigma], feed_dict={self.input_layer: xs})

    def get_var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae')

    def save(self, sess, save_file, global_step=None):
        var_list = self.get_var_list()
        print("CHRIS VAR LIST: {}".format(var_list))
        saver = tf.train.Saver(var_list)
        saver.save(sess, save_file, global_step=global_step)

    def restore(self, sess, save_file):
        var_list = self.get_var_list()
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_file)
