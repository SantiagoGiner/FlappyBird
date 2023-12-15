import csv
from typing import Callable

import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from ppo_plot import plot_ppo_results, LOGGING_PATH


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


vec_env = make_vec_env("FlappyBird-v0", n_envs=1)

logger = configure(LOGGING_PATH, ["stdout", "csv"])
model = ppo("mlppolicy", vec_env, learning_rate=linear_schedule(0.001), verbose=1)
model.set_logger(logger)
model.learn(total_timesteps=250000)
model.save("models/ppo_flappybird")

plot_ppo_results()
