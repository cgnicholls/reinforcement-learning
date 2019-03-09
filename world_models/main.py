import argparse
import os
import random
import subprocess

import numpy as np
import tensorflow as tf

from world_models.experience_collector import RolloutCollector, get_rollout_states, StatesServer
from world_models.environment import Pong
from world_models.actor import RandomActor
from world_models.vae import VAE


def get_directory_size_bytes(path):
    return int(subprocess.check_output(['du', '-k', path]).split()[0].decode(
        'utf-8')) * 1024


def collect_rollouts(environment, save_dir, num_rollouts, max_dir_size_gb=10):
    """Collects rollouts for the given environment with a random actor.
    Saves them in the form rollouts_i.h5 in the given save directory.
    """
    actor = RandomActor(environment.action_space)
    experience_collector = RolloutCollector()

    total_transitions = 0
    rollout_file_names = get_rollout_file_names(save_dir)

    if len(rollout_file_names) > 0:
        i = get_last_rollout_index(rollout_file_names) + 1
    else:
        i = 0

    while i < num_rollouts:
        print("Progress: {}/{}".format(i, num_rollouts))
        print("Collecting {} rollouts".format(1))
        experience_collector.collect_experience(actor, environment, 1)

        save_file = os.path.join(save_dir, 'rollouts_{}.h5'.format(i))

        print("Writing experience to {}".format(save_file))
        experience_collector.save_experience(save_file)
        total_transitions += len(experience_collector.rollouts[0].states)
        experience_collector.reset_experience()

        print("Transitions collected: {}".format(total_transitions))

        i += 1

        sz_gb = get_directory_size_bytes(save_dir) / 1024.0**3
        print("Size of directory: {:.2f} GB".format(sz_gb))
        if sz_gb > max_dir_size_gb:
            print("Size of directory bigger than limit: {}. Breaking.".format(max_dir_size_gb))
            break


def get_rollout_file_names(dataset_dir):
    rollout_files = []
    for dir_name, _, file_names in os.walk(dataset_dir):
        rollout_files.extend([os.path.join(dir_name, file_name) for file_name in file_names])
        break

    return [file_name for file_name in rollout_files if file_name.endswith('.h5')]


def get_last_rollout_index(rollout_file_names):
    """Assumes all file names are of the form rollouts_{idx}.h5, and we want to return the largest idx."""
    indices = []
    for name in rollout_file_names:
        s = name.split('/')[-1]
        s = s.split('_')[1]
        idx = s.split('.')[0]
        indices.append(int(idx))

    return max(indices)


def run_vae(dataset_dir, checkpoint_dir, log_dir, batch_size=32, num_epochs=50, write_summary_every=100):

    rollout_file_names = get_rollout_file_names(dataset_dir)

    vae = VAE()

    writer = tf.summary.FileWriter(log_dir)

    with tf.Session() as sess:
        vae.initialise(sess)
        global_step = 0
        for i_epoch in range(num_epochs):
            states_server = StatesServer(rollout_file_names)

            for x_minibatch in states_server.serve(batch_size):
                xs = np.stack(x_minibatch, axis=0)

                loss, summary = vae.train(sess, xs)

                if global_step % write_summary_every == 0:
                    writer.add_summary(summary, global_step=global_step)
                    print("Global step: {}".format(global_step))
                    print("Minibatch size: {}".format(xs.shape[0]))
                    print("Loss: {}".format(loss))

                global_step += 1

            # Save checkpoint at the end of each epoch.
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint')
            vae.save(sess, checkpoint_file, global_step=global_step)


def validate_experiment_id(experiment_id):
    return experiment_id.isalnum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='This python program has four stages: collect, train_v, '
              'train_m, train_c, which should be run in that order.')
    parser.add_argument(
        'experiment_id',
        help='The experiment ID to use for naming the save files.')
    parser.add_argument(
        '--env_name', default='Pong-v0',
        help='The environment name to use. Default value is Pong-v0.')
    parser.add_argument('--experiment_dir', default='experiments/world_models')

    subparsers = parser.add_subparsers(dest='command')

    parser_collect = subparsers.add_parser(
        'collect',
        help='Collect rollouts for training the V stage.')
    parser_collect.add_argument(
        '--num_rollouts', default=10000, type=int,
        help='The number of rollouts to collect to train the V model.')
    parser_collect.add_argument(
        '--max_dir_size_gb', default=10, type=int,
        help='The maximum size of the rollouts directory in GB.')

    parser_v = subparsers.add_parser(
        'v', help='Train the VAE on the collected rollouts.')
    parser_v.add_argument('--num_train', default=10000, type=int,
                          help='The number of training steps.')

    args = parser.parse_args()

    environment = Pong()

    assert validate_experiment_id(args.experiment_id)

    assert os.path.exists(args.experiment_dir), "{} does not exist".format(args.experiment_dir)
    experiment_dir = os.path.join(args.experiment_dir, args.experiment_id)

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    data_dir = os.path.join(experiment_dir, 'data')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if args.command == 'collect':
        # The VAE stage:
        collect_rollouts(environment, experiment_dir, args.num_rollouts, args.max_dir_size_gb)
    elif args.command == 'v':
        run_vae(data_dir, checkpoint_dir, log_dir)
