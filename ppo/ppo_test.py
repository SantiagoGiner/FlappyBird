import gymnasium as gym
import flappy_bird_gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("FlappyBird-v0", n_envs=1)
model = PPO.load("ppo_flappybird")

obs = vec_env.reset()
cumulative_reward = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    cumulative_reward += rewards
    print(cumulative_reward)