from collections import deque, namedtuple

import random

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class ReplayBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=self.maxlen)

    def append(self, x):
        self.buffer.append(x)

    def sample(self, n):
        return random.sample(self.buffer)

    def __len__(self):
        return len(self.buffer)


class ExperienceCollector:
    """An experience collector bridges an environment and an agent.

    The ExperienceCollector collects and stores experience, and generates samples of experience for training.
    """
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def collect(self, num_exp):
        """This function gets num_exp tuples of experience from the agent and environment, automatically handling end of
        episodes.
        """
        pass

    def sample(self, num_sample):
        """This function samples num_sample items of experience from the experience store.
        """
        pass


class ExperienceCollectorKSteps(ExperienceCollector):
    """Collects and stores transitions of the form (s_t, a_t, R_t, s_{t+k}), where
    R_t = r_t + gamma * r_{t+1} + ... + gamma^{k-1} * r_{t+k-1}.

    These are n-step transitions with discount factor gamma.
    """
    def __init__(self, env, agent, k, gamma, maxlen, reward_length=50):
        super().__init__(env, agent)

        self.gamma = gamma
        self.k = k
        self.buffer = ReplayBuffer(maxlen)

        self.states = deque(maxlen=self.k)
        self.actions = deque(maxlen=self.k)
        self.rewards = deque(maxlen=self.k)

        self.episode_rewards = deque(maxlen=reward_length)

        self.current_state = None
        self.episode_reward = 0

    def collect(self, num_exp):
        """Collect num_exp steps of experience."""
        for i in range(num_exp):
            # Take a step in the environment.
            if self.current_state is None:
                self.current_state = self.env.reset()
                self.episode_reward = 0

            action = self.agent.act(self.current_state)

            next_state, reward, done = self.env.step(action)
            self.episode_reward += reward

            # The one step transition is (self.current_state, action, reward, next_state).

            # We can store the state from k steps ago.
            # self.states[0] is s_t, and next_state is s_{t+k}.
            if len(self.states) == self.k:
                total_reward = sum([r * self.gamma ** j for j, r in enumerate(self.rewards)])
                self.rewards.popleft()
                transition = Transition(self.states.popleft(), self.actions.popleft(), total_reward, next_state)
                self.buffer.append(transition)

            # Append state, action and reward
            self.states.append(self.current_state)
            self.actions.append(action)
            self.rewards.append(reward)

            # If this is the end of the episode, then we need to add all transitions to the buffer.
            if done:
                next_state = None
                while len(self.states) > 0:
                    total_reward = sum([r * self.gamma ** l for l, r in enumerate(self.rewards)])
                    self.rewards.popleft()
                    self.buffer.append(
                        Transition(self.states.popleft(), self.actions.popleft(), total_reward, next_state))

                # Add the episode reward to episode rewards
                self.episode_rewards.append(self.episode_reward)

            self.current_state = next_state

    def sample(self, num_sample):
        return self.buffer.sample(num_sample)
