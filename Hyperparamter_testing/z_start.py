# py -3.7 z_start.py -alpha 0.0025 -gamma 0.99 -epsilon_rate_decay 0.9999995 -syncingfrequency 10000 -batchsize 32 -memorysize 50000
import argparse
import datetime
from pathlib import Path

import numpy as np
from gym.wrappers import FrameStack

import z_hyper_agent
import matplotlib.pyplot as plt

import gym


def start():
    parser = argparse.ArgumentParser(description="trains model based on different paramters")
    parser.add_argument('-env', type=str)
    parser.add_argument('-width', type=int)
    parser.add_argument('-height', type=int)
    parser.add_argument('-flatten', type=int)
    parser.add_argument('-episode_num', type=int)
    parser.add_argument('-save_every', type=int)
    parser.add_argument('-alpha', type=float)
    parser.add_argument('-gamma', type=float)
    parser.add_argument('-num_of_steps_to_decay', type=int)
    parser.add_argument('-syncingfrequency', type=int)
    parser.add_argument('-batchsize', type=int)
    parser.add_argument('-memorysize', type=int)

    args = parser.parse_args()
    image_size = (args.width, args.height)

    env = gym.make(args.env)
    memory = z_hyper_agent.ExperienceReplay(args.memorysize, args.batchsize)

    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = Path("Neural saves") / current_time
    save_dir.mkdir(parents=True)

    env = z_hyper_agent.SkipFrame(env, skip=4)
    env = z_hyper_agent.ImagePreProcessing(env, shape=image_size)
    env = FrameStack(env, num_stack=4)
    logger = z_hyper_agent.MetricLogger(save_dir)

    env.reset()

    ddqn_agent = z_hyper_agent.DoomAgent(env.observation_space.shape, env.action_space.n, save_dir=save_dir,
                                         memory=memory,
                                         gamma=args.gamma, alpha=args.alpha,
                                         num_of_steps= args.num_of_steps_to_decay ,
                                         syncingfrequency=args.syncingfrequency, minibatch=args.batchsize,
                                         flatten=args.flatten, save_every=args.save_every)

    eps_history, scores = [], []

    episodes = args.episode_num

    def log_hyper_parameters():
        f = open(str(save_dir) + "/Parameter_values.txt", "w")
        f.write("Env is : " + str(env.__str__()) + '\n')
        f.write("Image Size is: " + str(env.resize_shape) + '\n')
        f.write("Memory size is: " + str(memory.memory.maxlen) + '\n')
        f.write("GAMMA is: " + str(ddqn_agent.gamma) + '\n')
        f.write("Batch size  is: " + str(args.batchsize) + '\n')
        f.write("Epsilon decay  is: " + str(ddqn_agent.epsilon_rate_decay) + '\n')

        f.write("ALPHA is: " + str(ddqn_agent.alpha) + '\n')
        f.write("Learn every is: " + str(ddqn_agent.learn_every) + '\n')
        f.write("Syncing Frequency is: " + str(ddqn_agent.syncing_frequency) + '\n')
        f.close()

    log_hyper_parameters()

    for i in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:

            action = ddqn_agent.act(state)
            new_state, reward, done, info = env.step(action)
            ddqn_agent.experience.cache(state, new_state, action, reward, done)
            score += reward
            q, loss = ddqn_agent.learn()
            state = new_state
            if done:
                break

        logger.log_episode()

        if i % 20 == 0:
            logger.record(episode=i, epsilon=ddqn_agent.current_epsilon, step=ddqn_agent.curr_step)

        eps_history.append(ddqn_agent.current_epsilon)
        scores.append(score)
        if i % 100 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10):(i + 1)])
            print('episode', i, 'score: ', score, 'avg score %.3f' % avg_score,
                  'epsilon %.4f ' % ddqn_agent.current_epsilon)

    x = [i + 1 for i in range(episodes)]

    filename = '_alpha_' + str(args.alpha) + '_gamma_' + str(
        args.gamma) + '_epsilon_rate_decay_ ' + str(args.epsilon_rate_decay) + '_syncing_frequency_' + str(
        args.syncingfrequency) + "_syncing_batch_size_" + str(args.batchsize) + "_memeory_size_" + str(
        args.memorysize) + str(current_time) + '.png'

    def plotLearning(x, scores, epsilons, filename, lines=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)

        ax.plot(x, epsilons, color="C0")
        ax.set_xlabel("Game", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

        ax2.scatter(x, running_avg, color="C1")
        # ax2.xaxis.tick_top()
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        # ax2.set_xlabel('x label 2', color="C1")
        ax2.set_ylabel('Score', color="C1")
        # ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        # ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors="C1")

        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        plt.savefig(filename)

    plotLearning(x, scores, eps_history, filename)


start()
