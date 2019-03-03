import argparse
import os
import subprocess

import gym

from world_models.experience_collector import RolloutCollector
from world_models.actor import RandomActor
from world_models.vae import VAE


def get_directory_size_bytes(path):
    return int(subprocess.check_output(['du', path]).split()[0].decode('utf-8'))


def collect_rollouts(environment, save_dir, num_rollouts):
    """Runs the VAE stage of the experiment.
    """
    # TODO: Make action space general.
    action_space = [2, 3]
    # action_space = list(range(environment.action_space.n))
    actor = RandomActor(action_space)
    experience_collector = RolloutCollector()

    i = 0
    while i < num_rollouts:
        print("Collecting {} rollouts".format(1))
        experience_collector.collect_experience(actor, environment, 1)

        save_file_i = os.path.join(save_dir, 'rollouts_{}.pickle'.format(i))

        print("Writing experience to {}".format(tmp_save_file))
        experience_collector.save_experience()
        print("Renaming to {}".format(save_file_i))
        os.rename(tmp_save_file, save_file_i)

        print("Progress: {}/{}".format(i, num_rollouts))
        print("Total transitions: {}".format(len(experience_collector.experience)))

        experience_collector.reset_experience()

        i += rollouts_per_save

        sz = get_directory_size_bytes(save_dir)
        print("Size of directory: {} G".format(sz / (1024 * 1024 * 1024)))
        if sz > 100 * 1024 * 1024 * 1024:
            print("Size too big, breaking")
            break


def run_vae(dataset_dir, batch_size=100):
    vae = VAE(batch_size=batch_size)

    experience_collector = RolloutCollector()



def validate_experiment_id(experiment_id):
    return experiment_id.isalnum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='This python program has four stages: collect, train_v, train_m, train_c, '
                                           'which should be run in that order.')
    parser.add_argument('experiment_id', help='The experiment ID to use for naming the save files.')
    parser.add_argument('--env_name', default='Pong-v0', help='The environment name to use. Default value is Pong-v0.')
    parser.add_argument('--experiment_dir', default='/home/chris/repos/reinforcement-learning/experiments/world_models')

    subparsers = parser.add_subparsers(dest='command')

    parser_collect = subparsers.add_parser('collect', help='Collect rollouts for training the V stage.')
    parser_collect.add_argument('--num_rollouts', default=10000, type=int,
                                help='The number of rollouts to collect to train the V model.')

    parser_v = subparsers.add_parser('v', help='Train the VAE on the collected rollouts.')
    parser_v.add_argument('--num_train', default=10000, type=int, help='The number of training steps.')

    args = parser.parse_args()

    environment = gym.make('Pong-v0')

    assert validate_experiment_id(args.experiment_id)

    assert os.path.exists(args.experiment_dir)
    experiment_dir = os.path.join(args.experiment_dir, args.experiment_id)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    if args.command == 'collect':
        # The VAE stage:
        collect_rollouts(environment, experiment_dir, args.num_rollouts)
    elif args.command == 'v':
        run_vae(environment)
