import matplotlib.pyplot as plt
import csv

number_of_episodes = []
rewards = []

with open('checkpoints/defend/2021-04-25T10-34-13/log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
    for index, row in enumerate(plots):
        if index == 0:
            pass
        else:
            number_of_episodes.append(int(row[0]))
            rewards.append(float(row[3]))


plt.plot(number_of_episodes,rewards)
plt.xlabel('Number of episodes')
plt.ylabel('Mean reward per episode')
plt.title('Defend the centre')
plt.grid()
plt.legend()
plt.show()