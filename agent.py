# https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
import torch
from torch import nn
import numpy as np
from collections import deque
import random, copy, time, datetime, os
import matplotlib.pyplot as plt
from pathlib import Path

import gym 

env = gym.make('Assault-ram-v0')
env.reset()
next_state, reward, done, info = env.step(0)

# get_action_meanings() = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']  (7 possible actions for this game)
# step() = (array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  96, 254,
    #      0,   0,   0,   6, 100, 100, 100,   0,  54,   0,   0,   0, 253,
    #      0,   0, 192,   0, 136, 252,   2,  34, 226,  34, 162,  83, 162,
    #      6, 188, 255,   0,  25,   0, 253,   0, 253, 128,  64, 128, 128,
    #     64, 128,   0,   0,   0,   0,   0,   0,  30,  30,  17,  17,   0,
    #    253,   0, 127,  83,  68,  64,  19,  24,   0, 253,   0,   0,   0,
    #      0,   0,  34, 162,   0, 254,   0, 254,   0, 254,   0, 254,   0,
    #    254, 144,  60,   0,   0,   0,   0,   0,  80, 254,   4, 218,  69,
    #      0,  10,   0,   5,   0,   0, 255, 248,   0,   0,  64,   0, 172,
    #      0,   0,   0, 248, 251, 189, 251,  64, 251,   0, 245], dtype=uint8), 0.0, False, {'ale.lives': 4})
# step returns ob (ram in this case), reward, if game is over (Bool) and a dict of lives remaining


class Agent():

    def __init__(self, state_dim, action_dim, save_dir): 
        # super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  
        self.learn_every = 3  
        self.sync_every = 1e4 

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = AssaultNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_rate_decay = 0.99999975
        self.epsilon_min_value = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """Given a state, what action should be taken. Epsilon-greedy. Outputs an action to be performed, int. """
        
        # Explore
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)

        # Exploit
        else:
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0) # makes our state [1, 128]. Easier to explain in person. Not sure if this is essential for us.
            action_values = self.net(state, model="online")
            action = torch.argmax(action_values, axis=1).item()

        self.epsilon *= self.epsilon_rate_decay
        self.epsilon = max(self.epsilon_min_value, self.epsilon)

        self.curr_step += 1
        return action

    def memory_storage(self, state, next_state, action, reward, done):
        """Adds recent experience to memory (S, a) and what S' and r was observed. Keeping something semi tabular for an accurate lookup. Replay Buffer"""
        
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done))

    def memory_recall(self):
        """Picks a batch of experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch)) # Not 100% sure what exactly how this map function works
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze() # Look into why we need to squeeze this. Presumably to do with above map function

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ] # Q_online(s, a) [[0...31], [32 actions]]
        return current_Q

    # @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"assault_ram_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.epsilon),
            save_path,
        )
        print(f"AssaultNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        """Update Q values with mini-batch"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.memory_recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

class AssaultNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output.

    We don't need the 3 conv2d layers. Just dense + relu x 2 -> output.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, input, model):
        if model == "online":
            return self.online(input.float())
        elif model == "target":
            return self.target(input.float())

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

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = Agent(state_dim=128, action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 10000
for e in range(episodes):
    if e % 10 == 0:
        print('entering state', e)
    
    state = env.reset()

    while True:
        action = agent.act(state)
        # if e % 5 == 0:
        #     env.render()
        # time.sleep(0.01) Watch in real time
        next_state, reward, done, info = env.step(action)
        agent.memory_storage(state, next_state, action, reward, done)
        q, loss = agent.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        if done:
            break
    # if e % 5 == 0:
    #     env.close() # Not working for some reason
    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent.epsilon, step=agent.curr_step)