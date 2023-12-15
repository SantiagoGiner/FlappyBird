import matplotlib.pyplot as plt
import csv

LOGGING_PATH = "logging/"

def plot_ppo_results():
    elapsed_timesteps = []
    episode_rewards = []
    with open(LOGGING_PATH + 'progress.csv') as logfile:
        reader = csv.reader(logfile)
        parsed_header = False
        reward_index = None
        timestep_index = None
        for row in reader:
            if not parsed_header:
                # The first row in the log file contains column headings
                parsed_header = True
                reward_index = row.index('rollout/ep_rew_mean')
                timestep_index = row.index('time/total_timesteps')
                continue
            episode_rewards.append(float(row[reward_index]))
            elapsed_timesteps.append(float(row[timestep_index]))

    plt.plot(elapsed_timesteps, episode_rewards)
    plt.title("PPO Episode Reward")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.show()
