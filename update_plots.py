import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# training lineplot
output_data=pd.read_csv('training_reward.csv')
del output_data["run_count"]
plt.figure(figsize=(10, 5))
sns.lineplot(data=output_data)
plt.xlim(0,1000)
plt.savefig("training.png")

# test lineplot
test_data = pd.read_csv("test_reward.csv")
del test_data["run_count"]
plt.figure(figsize=(10,5))
sns.lineplot(data=test_data)
plt.xlim(0,9)
plt.savefig("test.png")

# test barplot
barWidth = 0.20
 
bars1 = test_data["Retailer - Average of 50 runs"]
bars2 = test_data["SARSA-500-300 - Reward"]
bars3 = test_data["Genetic Optimizer - Average of 7 runs"]
bars4 = test_data[test_data.columns[-1]]


r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.style.use("bmh")

fig = plt.figure(figsize =(8, 4))
  
ax = plt.subplot(111)


ax.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Retailer')
ax.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='SARSA')
ax.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='GO')
ax.bar(r4, bars4, color='#638c80', width=barWidth, edgecolor='white', label='Latest')


ax.set_xlabel('Dia')
ax.set_xticks([r + barWidth for r in range(len(bars1))], [50+i for i in range(len(bars1))])
ax.set_ylabel("Reward")
ax.hlines(0,-10,10, color="k")
ax.set_xlim(-0.5,10)
ax.grid(None)
 

ax.legend()
fig.show()
plt.savefig("barplot.png")