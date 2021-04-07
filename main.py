import time

import torch
import gym
from PIL import Image
import matplotlib.pyplot as plt
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as T
import numpy as np
import vizdoomgym

env = gym.make('VizdoomBasic-v0')


class Grayscale(gym.Wrapper):
    """
    Takes the environment as an input and returns an environment where the states are transformed to
    grayscale equivalent
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # We reimplement the step function to convert all states into grayscale equivalents
        next_state, reward, done, info = self.env.step(action)
        # The state has the format (height, width, channel) while pytorch takes as input (channel, height, width)
        gray_sc_img = np.transpose(next_state, (2, 0, 1))
        gray_sc_img = torch.tensor(gray_sc_img.copy(), dtype=torch.float)
        transform = T.Grayscale()
        gray_sc_img = transform(gray_sc_img)
        return gray_sc_img, reward, done, info

    def visualize(self, state):
        state = state.numpy()                   # We need to first convert the tensor back to a numpy vector
        state = np.transpose(state, (1, 2, 0))  # We then need to undo our format change
        data = np.squeeze(state,axis=2)         # This is done as a workaround against a pillow bug when dealing with images with one channel
        image = Image.fromarray(data.astype(np.uint8))
        image.show()
        image.close()
        time.sleep(4)




env = Grayscale(env)

while True:
    next_state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # To visualize the grayscale equivalent.
    Grayscale(env).visualize(next_state)

    if done:
        env.reset()

