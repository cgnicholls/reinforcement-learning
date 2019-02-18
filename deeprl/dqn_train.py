
import tensorflow as tf

from deeprl.agent import DQNAgent
from deeprl.env import GymEnv
from deeprl.experience_collector import ExperienceCollectorKSteps


class DQNetwork:

    def __init__(self, state_shape, action_dim):

        with tf.variable_scope('current'):
            self.current_q = self.build_network(state_shape, action_dim)
        with tf.variable_scope('target'):
            self.target_q = self.build_network(state_shape, action_dim)

    def build_network(self, state_shape, action_dim):
        self.x = tf.placeholder(tf.float32, shape=[None, *state_shape], name='x')

        self.

    def __call__(self, x):



def train_dqn(env_name, num_train=1000, k=3, gamma=0.9, maxlen=20):

    env = GymEnv(env_name)
    agent = DQNAgent(net)
    experience_collector = ExperienceCollectorKSteps(env, agent, k, gamma, maxlen)

    for i_train in range(num_train):
        experience_collector.collect(5)


if __name__ == "__main__":

    train_dqn("Pong-v0")
