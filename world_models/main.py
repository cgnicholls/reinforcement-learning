import argparse
import os
import subprocess

from world_models.experience_collector import RolloutCollector
from world_models.environment import Pong
from world_models.actor import RandomActor
from world_models.vae import VAE


def get_directory_size_bytes(path):
    return int(subprocess.check_output(['du', '-k', path]).split()[0].decode(
        'utf-8')) * 1024


def collect_rollouts(environment, save_dir, num_rollouts, max_dir_size_gb=1):
    """Collects rollouts for the given environment with a random actor.
    Saves them in the form rollouts_i.h5 in the given save directory.
    """
    actor = RandomActor(environment.action_space)
    experience_collector = RolloutCollector()

    total_transitions = 0
    for i in range(num_rollouts):
        print("Collecting {} rollouts".format(1))
        experience_collector.collect_experience(actor, environment, 1)

        save_file = os.path.join(save_dir, 'rollouts_{}.h5'.format(i))

        print("Writing experience to {}".format(save_file))
        experience_collector.save_experience(save_file)
        total_transitions += len(experience_collector.rollouts[0].states)
        experience_collector.reset_experience()

        print("Progress: {}/{}".format(i, num_rollouts))
        print("Total transitions: {}".format(total_transitions))

        sz = get_directory_size_bytes(save_dir)
        print("Size of directory: {:.2f} MB".format(sz / 1024**2))
        if sz / 1024**3 > max_dir_size_gb:
            print("Size too big, breaking")
            break


def run_vae(dataset_dir, batch_size=100):
    vae = VAE(batch_size=batch_size)

    experience_collector = RolloutCollector()


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

    parser_v = subparsers.add_parser(
        'v', help='Train the VAE on the collected rollouts.')
    parser_v.add_argument('--num_train', default=10000, type=int,
                          help='The number of training steps.')

    args = parser.parse_args()

    environment = Pong()

    assert validate_experiment_id(args.experiment_id)

    assert os.path.exists(args.experiment_dir), "{} does not exist".format(
        args.experiment_dir)
    experiment_dir = os.path.join(args.experiment_dir, args.experiment_id)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    if args.command == 'collect':
        # The VAE stage:
        collect_rollouts(environment, experiment_dir, args.num_rollouts)
    elif args.command == 'v':
        run_vae(environment)
