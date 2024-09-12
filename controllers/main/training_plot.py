import pandas as pd
import matplotlib.pyplot as plt

test = 3
method = "YOLO"
dir = f"./controllers/main/test_results_test_{test}/{method}_20240913/"
filename = f"{method}_pengujian{test}_"

# Load the CSV files
reward_data = pd.read_csv(dir + filename + "rew.csv")
duration_data = pd.read_csv(dir + filename + "dur.csv")
episode_length_data = pd.read_csv(dir + filename + "eps_len.csv")

# Extract data for plotting
steps_reward = reward_data["Step"]
reward_values = reward_data["Value"]

steps_episode_length = episode_length_data["Step"]
episode_length_values = episode_length_data["Value"]

# Plot Reward and Episode Length
plt.figure(figsize=(10, 6))

# Plot Reward
plt.subplot(2, 1, 1)
plt.plot(steps_reward, reward_values, label="Reward", color="blue")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Reward Mean/Eps")
plt.grid(True)

# Plot Episode Length
plt.subplot(2, 1, 2)
plt.plot(
    steps_episode_length, episode_length_values, label="Episode Length", color="green"
)
plt.xlabel("Timesteps")
plt.ylabel("Episode Length")
plt.title("Episode Length Mean")
plt.grid(True)

plt.tight_layout()
plt.savefig(dir + filename + "plot.png")
plt.show()

# Calculate total training duration
start_time = duration_data["Wall time"].iloc[0]
end_time = duration_data["Wall time"].iloc[-1]
total_duration_seconds = end_time - start_time
total_duration_hours = total_duration_seconds / 3600

print(f"Total Training Duration: {total_duration_hours:.2f} hours")
