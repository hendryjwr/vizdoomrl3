import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy, time
from torchsummary import summary
from torchvision import models
import torch.nn.functional as func
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import matplotlib.pyplot as plt
import vizdoomgym

mini_batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# For debugging purposes
if device.type == 'cuda':
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Max Allocated:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    print('Max Cached:   ', round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 1), 'GB')

env = gym.make('VizdoomPredictPosition-v0')


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
        rgb_to_grayscale = transforms.Grayscale()
        converted_state = rgb_to_grayscale(converted_state)

        # Applies resizing
        transformation = transforms.Compose([transforms.Resize(self.resize_shape), transforms.Normalize(0, 255)])
        # Uncomment this is for visualization
        # transformation = transforms.Resize(self.resize_shape)
        processed_state = transformation(converted_state).squeeze(0)
        return processed_state


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
        reward_sum = 0
        for _ in range(self.frames_to_skip):
            new_state, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done:
                break
        return new_state, reward_sum, done, info


env = SkipFrame(env, skip=4)
env = ImagePreProcessing(env, shape=(90, 120))
env = FrameStack(env, num_stack=4)

env.reset()


#
def construct_tensor(value):
    if torch.cuda.is_available():
        value = torch.tensor(value, device=torch.device('cuda:0'))
        return value
    else:
        return torch.tensor(value)


# 1. Constructing the Neural network

class DoomNN(nn.Module):
    def __init__(self, input_image, output_dim):
        super().__init__()
        c, h, w = input_image
        # The CNN hyper-parameters were selected based  on: https://arxiv.org/abs/1509.06461
        self.conv_layer_1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear_1 = nn.Linear(4928, 512)
        self.linear_output = nn.Linear(512, output_dim)

    def forward(self, input):
        # If it doesn't work come back here
        input = copy.deepcopy(input)
        fw_net = func.relu(self.conv_layer_1(input))
        fw_net = func.relu(self.conv_layer_2(fw_net))
        fw_net = func.relu(self.conv_layer_3(fw_net))
        fw_net = torch.flatten(fw_net, 1)
        fw_net = func.relu(self.linear_1(fw_net))
        fw_net = self.linear_output(fw_net)
        return fw_net


class DoomAgent:

    def __init__(self, state_dim, action_dim, save_dir):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.save_every = 100000
        self.num_of_steps = 2000000

        # Learning Parameters
        self.gamma = 0.99
        self.alpha = 0.00025  # 0.00025
        self.current_epsilon = 1
        self.epsilon_rate_min = 0.1
        self.epsilon_rate_decay = pow(self.epsilon_rate_min,
                                      1 / self.num_of_steps)  # Number of steps for epsilon to decay to 0.1

        self.burnin = 32
        self.learn_every = 3

        # Syncing parameters
        self.syncing_frequency = 10000

        # Tracking the current step
        self.curr_step = 0

        self.nn_online = DoomNN(self.state_dim, self.action_dim).float()
        self.nn_target = DoomNN(self.state_dim, self.action_dim).float()

        self.optimizer = torch.optim.Adam(self.nn_online.parameters(), self.alpha)
        self.loss_func = torch.nn.SmoothL1Loss()

        if torch.cuda.is_available():
            self.nn_online = self.nn_online.to(device="cuda")

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
            action_values = self.nn_online(state)
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.current_epsilon *= self.epsilon_rate_decay
        self.current_epsilon = self.current_epsilon if self.current_epsilon >= self.epsilon_rate_min else self.epsilon_rate_min

        # increment step
        self.curr_step += 1

        return action_idx

    def learn(self):

        if self.curr_step % self.syncing_frequency == 0:
            self.nn_target.load_state_dict(self.nn_online.state_dict())
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step % self.learn_every != 0:
            return None, None
        if self.curr_step < self.burnin:
            return None, None

        current_q_values, loss = self.update(experience.recall())

        return (current_q_values.mean().item(), loss)

    def update(self, memory):

        current_state_array, new_state_array, action_array, reward_array, done_array = memory

        # TD_estimate
        indexing_array = range(mini_batch_size)
        current_q_values = self.nn_online(current_state_array)[indexing_array, action_array]

        # TD_target
        future_q_values = self.nn_online(new_state_array)
        best_action_array = torch.argmax(future_q_values, axis=1)
        target_q_values = self.nn_target(new_state_array)[indexing_array, best_action_array]
        q_target = (reward_array + (1 - done_array.float()) * self.gamma * target_q_values).to(torch.float32)

        # Apply gradient descent
        loss = self.loss_func(current_q_values, q_target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return current_q_values, loss.item()

    def save(self):
        save_path = (
                self.save_dir / f"doom_net_{int(self.curr_step // self.save_every)}.pt"
        )
        torch.save(
            dict(model=self.nn_online.state_dict(), exploration_rate=self.current_epsilon),
            save_path,
        )


class ExperienceReplay:
    def __init__(self):
        self.memory = deque(maxlen=50000)  # We leave this value at 100k for now

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

        states, next_states, actions, rewards, dones = zip(*random.sample(self.memory, mini_batch_size))

        states = torch.reshape(torch.cat(states), (mini_batch_size, 4, 90, 120))
        next_states = torch.reshape(torch.cat(next_states), (mini_batch_size, 4, 90, 120))
        # actions = torch.stack(actions, dim=1).squeeze()
        actions = torch.cat(actions).squeeze()
        rewards = torch.cat(rewards).squeeze()
        dones = torch.cat(dones).squeeze()
        return states, next_states, actions, rewards, dones


# The following class is taken from https://github.com/yuansongFeng/MadMario/ for metric logging of data. 
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


# summary(DoomNN((4, 120, 160), env.action_space.n), (4, 120, 160))

# 2. Making the epsilon greedy policy
# 3. Implementing the Q learning pseudocode


save_dir = Path("checkpoints") / "corridor_exp_3" / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

print(env.action_space.n)
ddqn_agent = DoomAgent(env.observation_space.shape, env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)
experience = ExperienceReplay()


def log_hyper_parameters():
    f = open(str(save_dir) + "/Parameter_values.txt", "w")
    f.write("Env is : " + str(env.__str__()) + '\n')
    f.write("Image Size is: " + str(env.resize_shape) + '\n')
    f.write("Memory size is: " + str(experience.memory.maxlen) + '\n')
    f.write("GAMMA is: " + str(ddqn_agent.gamma) + '\n')
    f.write("Batch size  is: " + str(mini_batch_size) + '\n')
    f.write("Epsilon decay  is: " + str(ddqn_agent.epsilon_rate_decay) + '\n')
    # f.write("Burn in  is: " + str(ddqn_agent.burn) + '\n') # To add
    f.write("ALPHA is: " + str(ddqn_agent.alpha) + '\n')
    f.write("Learn every is: " + str(ddqn_agent.learn_every) + '\n')
    f.write("Syncing Frequency is: " + str(ddqn_agent.syncing_frequency) + '\n')
    f.close()


log_hyper_parameters()


def play():
    episodes = 300000

    for i in range(episodes):

        state = env.reset()
        done = False

        while not done:

            # env.render()
            action = ddqn_agent.act(state)
            new_state, reward, done, info = env.step(action)
            experience.cache(state, new_state, action, reward, done)
            q, loss = ddqn_agent.learn()
            logger.log_step(reward, loss, q)

            # Step 1 Calculate td_estimate
            # Step 2 Calculate td_target
            # Step 3 Calculate loss
            # Step 4 Backward propagation
            # Step 5 Sync online and target networks
            # Step 6 make new state, old state

            state = new_state

            if done:
                break

        logger.log_episode()

        if i % 20 == 0:
            logger.record(episode=i, epsilon=ddqn_agent.current_epsilon, step=ddqn_agent.curr_step)


play()

# TD estimate as was done in Mario
# current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
# TD estimate rewritten
# current_Q = [0] * 32
# for index, element in enumerate(states):
# current_Q[index] = self.net(element, model="online")[action[index]]
