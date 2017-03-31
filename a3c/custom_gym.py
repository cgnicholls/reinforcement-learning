from scipy.misc import imresize
import gym
import numpy as np
import random



class CustomGym:
    def __init__(self, game_name, skip_actions=4, num_frames=4, w=84, h=84):
        self.env = gym.make(game_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h
        if game_name == 'SpaceInvaders-v0':
            self.action_space = [1,2,3] # For space invaders
        elif game_name == 'Pong-v0':
            self.action_space = [1,2,3]
        elif game_name == 'Breakout-v0':
            self.action_space = [1,4,5]
        else:
            # Use the actions specified by Open AI. Sometimes this has more
            # actions than we want, and some actions do the same thing.
            self.action_space = range(env.action_space.n)

        self.action_size = len(self.action_space)
        self.observation_shape = self.env.observation_space.shape

        self.state = None
        self.game_name = game_name

    def preprocess(self, obs, is_start=False):
        grayscale = obs.astype('float32').mean(2)
        s = imresize(grayscale, (self.w, self.h)).astype('float32') * (1.0/255.0)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)
        return self.state

    def render(self):
        self.env.render()

    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        accum_reward = 0
        prev_s = None
        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break
            prev_s = s
        # Takes maximum value for each pixel value over the current and previous
        # frame. Used to get round Atari sprites flickering (Mnih et al. (2015))
        if self.game_name == 'SpaceInvaders-v0' and prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info
