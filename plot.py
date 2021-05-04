import matplotlib.pyplot as plt
import csv

number_of_episodes_dtl = []
rewards_dtl = []

number_of_episodes_dtc = []
rewards_dtc = []

number_of_episodes_dc = []
rewards_dc = []




# Defend the line

with open('Hyperparamter_testing/Neural saves/2021-04-25T10-09-53/log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
    for index, row in enumerate(plots):
        if index == 0:
            pass
        else:
            number_of_episodes_dtl.append(int(row[0]))
            rewards_dtl.append(float(row[3]))



# Defend the centre

with open('checkpoints/defend/2021-04-25T10-34-13/log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
    for index, row in enumerate(plots):
        if index == 0:
            pass
        else:
            number_of_episodes_dtc.append(int(row[0]))
            rewards_dtc.append(float(row[3]))


# Deadly corridor

with open('checkpoints/corridor_exp_3/2021-04-24T13-15-55/log','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
    for index, row in enumerate(plots):
        if index == 0:
            pass
        else:
            number_of_episodes_dc.append(int(row[0]))
            rewards_dc.append(float(row[3]))





fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(number_of_episodes_dtl, rewards_dtl)
ax1.set_title("Defend the line")
ax1.set_ylabel("Mean reward per episode")
ax1.set_xlabel("Number of episodes")
ax1.grid()

ax2.plot(number_of_episodes_dtc, rewards_dtc)
ax2.set_title("Defend the centre")
ax2.set_ylabel("Mean reward per episode")
ax2.set_xlabel("Number of episodes")
ax2.grid()


ax3.plot(number_of_episodes_dc, rewards_dc)
ax3.set_title("Deadly corridor")
ax3.set_ylabel("Mean reward per episode")
ax3.set_xlabel("Number of episodes")
ax3.grid()


plt.show()

