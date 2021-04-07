import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import vizdoomgym
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

env = gym.make('VizdoomCorridor-v0')


class GrayScaleObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


def visualize(state, i):
    state = state.__array__()
    # print(state.shape)
    # print('state before', state)
    state = state[0]
    # print(state.shape)
    # print('\n')
    # print('state after', state)

    # state = np.squeeze(state, axis=0)
    # state = state.squeeze(0, axis= 3)
    # state = np.transpose(state, (1, 2, 0))              # We then need to undo our format change
    # data = np.squeeze(state,
    #                   axis=2)                           # This is done as a workaround against a pillow bug when dealing with images with one channel
    image = Image.fromarray(state.astype(np.uint8))
    # print(state)
    image.save(f'test_images/frame{i}.png')


class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape


        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        # Uncomment this is for visualization
        # transforms = T.Resize(self.shape)
        observation = transforms(observation).squeeze(0)
        return observation


env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=(120, 160))
env = FrameStack(env, num_stack=4)
env.reset()

for i in range(100):
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # print(state.shape)
    visualize(state, i)

    if done:
        env.reset()

env.close()
