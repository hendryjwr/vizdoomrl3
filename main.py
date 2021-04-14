import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import vizdoomgym
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from torchsummary import summary

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

env = gym.make('VizdoomCorridor-v0')


class ImagePreProcessing(gym.ObservationWrapper):

    def __init__(self, env, shape):
        super().__init__(env)
        self.resize_shape = shape
        height, width, _ = env.observation_space.shape
        self.observation_space = Box(0, 255, shape=self.resize_shape, dtype=np.uint8)

    def observation(self, state):
        # Applies Grayscale
        state = np.transpose(state, (2, 0, 1))
        to_convert = state.copy()
        converted_state = torch.tensor(to_convert, dtype=torch.float32)
        transform = transforms.Grayscale()
        converted_state = transform(converted_state)

        # Applies resizing

        transformation = transforms.Compose([transforms.Resize(self.resize_shape), transforms.Normalize(0, 255)])
        # Uncomment this is for visualization
        # transformation = transforms.Resize(self.resize_shape)
        observation = transformation(converted_state).squeeze(0)
        return observation


def visualise(state, i):
    state = state.__array__()
    state = state[0]
    image = Image.fromarray(state.astype(np.uint8))
    image.save(f'test_images/frame{i}.png')


class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        super().__init__(env)
        self.frames_to_skip = skip

    def step(self, action):
        reward_sum = 0.0
        for _ in range(self.frames_to_skip):
            new_state, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done:
                break
        return new_state, reward_sum, done, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        # print('before', self.observation_space.shape, self.shape)
        # obs_shape = self.shape + self.observation_space.shape[2:]
        # print('after', obs_shape)
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        transformation = transforms.Compose(
            [transforms.Resize(self.shape), transforms.Normalize(0, 255)]
        )
        # Uncomment this is for visualization
        # transformation = transforms.Resize(self.shape)
        observation = transformation(observation).squeeze(0)
        return observation


env = SkipFrame(env, skip=4)
env = ImagePreProcessing(env, shape=(120, 160))
env = FrameStack(env, num_stack=4)
env.reset()


#

# 1. Constructing the Neural network

class DoomNN(nn.Module):
    def __init__(self, input_image, output_image):
        super().__init__()

        c, h, w = input_image

        self.online = nn.Sequential(
            nn.Conv2d(c, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(3136, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_image),
        )

    def forward(self, x):
        print(x.vgg16())
        return x


state, reward, done, info = env.step(env.action_space.sample())

model = DoomNN(state.shape, 9999)
model(state)

# 2. Making the epsilon greedy policy
# 3. Implementing the Q learning pseudocode

# for i in range(4):
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
#     # print(state.shape)
#     visualise(state, i)
#
#     if done:
#         env.reset()
#
# env.close()
