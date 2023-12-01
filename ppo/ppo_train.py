import gymnasium as gym
import flappy_bird_gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("FlappyBird-v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=2500000)
model.save("ppo_flappybird")
