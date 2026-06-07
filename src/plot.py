import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [8, 10, 12, 15, 14, 17, 19, 18, 21, 22])
plt.title("Training Reward over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig("../assets/reward_plot.png")
print("Plot saved!")