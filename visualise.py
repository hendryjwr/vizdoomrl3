import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import vizdoomgym
from pathlib import Path
from collections import deque
import random, datetime, os, copy, time
from torchsummary import summary
from torchvision import models
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


env = gym.make('VizdoomHealthGathering-v0')


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
env = ImagePreProcessing(env, shape=(60, 80))
env = FrameStack(env, num_stack=4)

env.reset()

def construct_tensor(value):
    if torch.cuda.is_available():
        value = torch.tensor(value, device=torch.device('cuda:0'))
        return value
    else:
        return torch.tensor(value)

class DoomNN(nn.Module):
    def __init__(self, input_image, output_dim):
        super().__init__()
        c, h, w = input_image
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class DoomAgent:

    def __init__(self, state_dim, action_dim, checkpoint=None):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Learning Parameters
        self.alpha = 0.00025
        self.current_epsilon = 1
        self.epsilon_rate_decay = 0.99999975
        self.epsilon_rate_min = 0.1
        self.checkpoint = checkpoint

        # Tracking the current step
        self.curr_step = 0

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.neural_net = DoomNN(self.state_dim, self.action_dim).float()

        if checkpoint:
            self.load(checkpoint)

        # We
        self.optimizer = torch.optim.Adam(self.neural_net.parameters(), self.alpha)
        self.loss_func = torch.nn.SmoothL1Loss()

        if torch.cuda.is_available():
            self.neural_net = self.neural_net.to(device="cuda")

    def act(self, state):

        """
        The state is a lazy frame
        We convert it to an array then to a tensor
        """

        # EXPLORE
        if np.random.random() < self.current_epsilon:
            action_idx = np.random.randint(0, self.action_dim)


        # EXPLOIT
        else:
            state = state.__array__()
            state = construct_tensor(state)
            state = state.unsqueeze(0)
            action_values = self.neural_net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.current_epsilon *= self.epsilon_rate_decay
        self.current_epsilon = self.current_epsilon if self.current_epsilon >= self.epsilon_rate_min else self.epsilon_rate_min

        # increment step
        self.curr_step += 1

        return action_idx

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=device)
        # current_epsilon = ckp.get('exploration_rate')
        current_epsilon = 0.01
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {current_epsilon}")
        self.neural_net.load_state_dict(state_dict)
        self.current_epsilon = current_epsilon

checkpoint = Path('checkpoints/medic/2021-04-20T15-36-59/doom_net_6.pt')
ddqn_agent = DoomAgent(env.observation_space.shape, env.action_space.n, checkpoint=checkpoint)

def vis():

    episodes = 10

    for i in range(episodes):

        state = env.reset()
        done = False

        while not done:
            time.sleep(0.03)
            env.render()
            action = ddqn_agent.act(state)
            new_state, reward, done, info = env.step(action)
            state = new_state

            if done:
                print('done')
                break

vis()