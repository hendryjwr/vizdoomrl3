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
from torchvision import models
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

mini_batch_size = 32

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
env = ImagePreProcessing(env, shape=(90, 120))
env = FrameStack(env, num_stack=4)

env.reset()


#
def construct_tensor(value):
    if torch.cuda.is_available():
        value = torch.tensor(value).cuda()
        return value
    else:
        return torch.tensor(value)


# 1. Constructing the Neural network

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
            nn.Linear(4928, 512),
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

    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.save_dir = save_dir

        # Learning Parameters
        self.gamma = 0.99
        self.alpha = 0.00025
        self.current_epsilon = 1
        self.epsilon_rate_decay = 0.99999975
        self.epsilon_rate_min = 0.1

        self.learn_every = 3

        # Syncing parameters
        self.syncing_frequency = 10000

        # Tracking the current step
        self.curr_step = 0

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.neural_net = DoomNN(self.state_dim, self.action_dim).float()

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

    def learn(self):
        if self.curr_step % self.learn_every != 0:
            return
        if self.curr_step < 10000:
            return

        # Step 1: Recall from memory
        current_state_array, new_state_array, action_array, reward_array, done_array = experience.recall()

        # Step 2: Calculate TD estimate based on the online network
        current_q_values = self.td_estimate(current_state_array, action_array)

        # Step 3: Calculate TD target
        target_q_values = self.td_target(new_state_array, reward_array, done_array)        

        # Step 4: Perform gradient descent
        self.update(current_q_values, target_q_values)

        # Step 5: Sync online network with target network

        if self.curr_step % self.syncing_frequency == 0:
            self.neural_net.target.load_state_dict(self.neural_net.online.state_dict())

    def td_estimate(self, current_state_array, action_array):
        indexing_array = np.arange(0, mini_batch_size)
        current_q_values = self.neural_net(current_state_array, "online")[indexing_array, action_array]
        return current_q_values

    def td_target(self, next_state, reward, done):

        future_q_values = self.neural_net(next_state, "online")
        best_action_array = torch.argmax(future_q_values, axis=1)

        indexing_array = np.arange(0, mini_batch_size)
        target_q_values = self.neural_net(next_state, "target")[indexing_array, best_action_array]

        return (reward + (1 - done.float()) * self.gamma * target_q_values).to(torch.float32)

    def update(self, current_q_values, target_q_values):

        # Here parameter = neural network weights
        loss = self.loss_func(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.curr_step % 50 == 0:
            print(loss.item())


class ExperienceReplay:
    def __init__(self):
        self.memory = deque(maxlen=100000)  # We leave this value at 100k for now

    def construct_tensor(self, value):

        if self.cuda:
            return torch.tensor(value).cuda()
        else:
            return torch.tensor(value)

    def cache(self, current_state, new_state, action, reward, done):

        current_state = construct_tensor(current_state.__array__())
        new_state = construct_tensor(new_state.__array__())
        action = construct_tensor([action])
        reward = construct_tensor([reward])
        done = construct_tensor([done])

        self.memory.append((current_state, new_state, action, reward, done))

    def recall(self):

        # 100% to change later
        batch = random.sample(self.memory, mini_batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


# summary(DoomNN((4, 120, 160), env.action_space.n), (4, 120, 160))

# 2. Making the epsilon greedy policy
# 3. Implementing the Q learning pseudocode

ddqn_agent = DoomAgent(env.observation_space.shape, env.action_space.n)
experience = ExperienceReplay()


def play():
    episodes = 1000000000000000000000000000000000
    for i in range(episodes):

        state = env.reset()
        done = False

        while not done:
            if ddqn_agent.curr_step % 1000 == 0:
                print(ddqn_agent.curr_step)

            action = ddqn_agent.act(state)
            new_state, reward, done, info = env.step(action)
            experience.cache(state, new_state, action, reward, done)
            ddqn_agent.learn()
            state = new_state

            # Step 1 Calculate td_estimate
            # Step 2 Calculate td_target
            # Step 3 Calculate loss
            # Step 4 Backward propagation
            # Step 5 Sync online and target networks
            # Step 6 make new state, old state

            # visualise(state, i)

            if done:
                break


play()

# TD estimate as was done in Mario
# current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
# TD estimate rewritten
# current_Q = [0] * 32
# for index, element in enumerate(states):
# current_Q[index] = self.net(element, model="online")[action[index]]
