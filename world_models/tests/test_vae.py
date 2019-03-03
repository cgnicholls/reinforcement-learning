import numpy as np
import tensorflow as tf
import tempfile
import pytest

from world_models.vae import VAE


def setup_function(function):
    np.random.seed(0)


def teardown_function(function):
    tf.reset_default_graph()


def test_can_train_vae():

    vae = VAE()

    batch_size = 100
    training_data = np.random.randn(batch_size, 64, 64, 3)

    x = np.random.randn(1, 64, 64, 3)

    with tf.Session(graph=vae.graph) as sess:
        sess.run(tf.global_variables_initializer())

        encoding_before = vae.encode(sess, x)

        _ = vae.train(sess, training_data)

        encoding_after = vae.encode(sess, x)

        # Check that the encoding of x has changed after training.
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(encoding_before, encoding_after)


def test_can_save_and_restore_vae():

    vae = VAE()

    batch_size = 100
    training_data = np.random.randn(batch_size, 64, 64, 3)

    save_file = tempfile.NamedTemporaryFile()

    with tf.Session(graph=vae.graph) as sess:
        vae.initialise(sess)

        _ = vae.train(sess, training_data)

        vae.save(sess, save_file.name)

        x = np.random.randn(1, 64, 64, 3)
        encoding_before = vae.encode_to_mu_sigma(sess, x)
        encoding_before2 = vae.encode_to_mu_sigma(sess, x)

        # Check the encoding to mu, sigma is deterministic.
        np.testing.assert_allclose(encoding_before, encoding_before2)

    vae2 = VAE()

    with tf.Session(graph=vae2.graph) as sess:
        # When we restore to vae2, we want the same weights as when we saved vae.
        vae2.restore(sess, save_file.name)
        encoding_after = vae2.encode_to_mu_sigma(sess, x)

        # Check the encoding to mu and sigma we saved is the same as for what we restored.
        np.testing.assert_allclose(encoding_before, encoding_after)

    save_file.close()


def test_generate_white_noise():

    latent_dim = 30
    vae = VAE(latent_dim=latent_dim)

    batch_size = 100
    training_data = np.random.randn(batch_size, 64, 64, 3)

    with tf.Session(graph=vae.graph) as sess:
        vae.initialise(sess)

        white_noise = sess.run(vae.white_noise, feed_dict={
            vae.input_layer: training_data
        })

        assert white_noise.shape == (batch_size, latent_dim)

        _ = vae.train(sess, training_data)
