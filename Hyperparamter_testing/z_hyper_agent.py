import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
import numpy as np
from collections import deque
import random, datetime, os, copy, time
import gym
from gym.spaces import Box
import vizdoomgym
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Max Allocated:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    print('Max Cached:   ', round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 1), 'GB')

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
        # transformation = transforms.Compose(
        #     [transforms.Resize(self.shape), transforms.Normalize(0, 255)]
        # )
        # Uncomment this is for visualization
        transformation = transforms.Resize(self.shape)
        observation = transformation(observation).squeeze(0)
        return observation


def construct_tensor(value):
    if torch.cuda.is_available():
        value = torch.tensor(value, device=torch.device('cuda:0'))
        return value
    else:
        return torch.tensor(value)


class DoomNN(nn.Module):
    def __init__(self, input_image, output_dim, flatten):
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
            nn.Linear(flatten, 512),
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

    def __init__(self, state_dim, action_dim, save_dir, memory, gamma, alpha, num_of_steps, syncingfrequency,
                 minibatch, flatten, save_every):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_every = save_every
        self.save_dir = save_dir
        self.mini_batch = minibatch

        self.experience = memory
        # Learning Parameters
        self.gamma = gamma
        self.alpha = alpha  # 0.00025
        self.current_epsilon = 1

        self.epsilon_rate_min = 0.1
        self.num_of_steps = num_of_steps
        self.epsilon_rate_decay = pow(self.epsilon_rate_min,
                                      1 / self.num_of_steps)  # Number of steps for epsilon to decay to 0.1

        self.burnin = 10000
        self.learn_every = 3

        # Syncing parameters
        self.syncing_frequency = syncingfrequency

        # Tracking the current step
        self.curr_step = 0

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        print(self.state_dim, self.action_dim)
        self.neural_net = DoomNN(self.state_dim, self.action_dim, flatten).float()

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

        if self.curr_step % self.syncing_frequency == 0:
            self.neural_net.target.load_state_dict(self.neural_net.online.state_dict())
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step % self.learn_every != 0:
            return None, None
        if self.curr_step < self.burnin:
            return None, None

        # Step 1: Recall from memory
        current_state_array, new_state_array, action_array, reward_array, done_array = self.experience.recall()

        # Step 2: Calculate TD estimate based on the online network
        current_q_values = self.td_estimate(current_state_array, action_array)

        # Step 3: Calculate TD target
        target_q_values = self.td_target(new_state_array, reward_array, done_array)

        # Step 4: Perform gradient descent
        loss_value = self.update(current_q_values, target_q_values)

        # Step 5: Sync online network with target network

        return (current_q_values.mean().item(), loss_value)

    def td_estimate(self, current_state_array, action_array):
        indexing_array = np.arange(0, self.mini_batch)
        current_q_values = self.neural_net(current_state_array, "online")[indexing_array, action_array]
        return current_q_values

    def td_target(self, next_state, reward, done):

        future_q_values = self.neural_net(next_state, "online")
        best_action_array = torch.argmax(future_q_values, axis=1)

        indexing_array = np.arange(0, self.mini_batch)
        target_q_values = self.neural_net(next_state, "target")[indexing_array, best_action_array]

        return (reward + (1 - done.float()) * self.gamma * target_q_values).to(torch.float32)

    def update(self, current_q_values, target_q_values):

        # Here parameter = neural network weights
        loss = self.loss_func(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    # Save function from https://github.com/yuansongFeng/MadMario/ to work alongsite logging features.
    def save(self):
        save_path = (
                self.save_dir / f"doom_net_{int(self.curr_step // self.save_every)}.pt"
        )
        torch.save(
            dict(model=self.neural_net.state_dict(), exploration_rate=self.current_epsilon),
            save_path,
        )


class ExperienceReplay:
    def __init__(self, size, minibach):
        self.memory = deque(maxlen=size)  # We leave this value at 100k for now
        self.mini_batch = minibach

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
        batch = random.sample(self.memory, self.mini_batch)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
