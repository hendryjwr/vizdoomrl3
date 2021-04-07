import gym
import vizdoomgym
import torch
from PIL import Image

def save_images(state, i):
    img = Image.fromarray(state, 'RGB')
    img.save(f'test_images/frame{i}.png')

env = gym.make('VizdoomCorridor-v0')

# state.shape = (240, 320, 3)

# use like a normal Gym environment
state = env.reset()
for i in range(5):
    state, reward, done, info = env.step(env.action_space.sample())
    save_images(state, i)
    # print(state, reward, done, info)
    # print(state.shape)
    env.render()
    if done:
        env.reset()


env.close()