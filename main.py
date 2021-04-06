import gym
import vizdoomgym
import torch
env = gym.make('VizdoomCorridor-v0')

# state.shape = (240, 320, 3)

# use like a normal Gym environment
state = env.reset()
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    # print(state, reward, done, info)
    # print(state.shape)
    env.render()
    if done:
        env.reset()


env.close()