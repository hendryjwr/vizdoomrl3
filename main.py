import torch
import gym
import matplotlib.pyplot as plt
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
        gray_sc_img = np.transpose(next_state, (2, 0, 1))
        gray_sc_img = torch.tensor(gray_sc_img.copy(), dtype=torch.float)
        transform = T.Grayscale()
        gray_sc_img = transform(gray_sc_img)
        return gray_sc_img, reward, done, info


env = Grayscale(env)

while True:
    next_state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # To visualize the grayscale equivalent, the blue hue is caused by the way matplotlib lib renders the images
    plt.imshow(next_state.squeeze(0))
    plt.show()
    if done:
        env.reset()
