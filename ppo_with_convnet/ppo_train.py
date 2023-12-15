import gymnasium as gym
import flappy_bird_gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from convnet_wrapper import ResNetWrapper, ConvNetWrapper

def make_wrapped_env():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    wrapped_env = ConvNetWrapper(env)
    return wrapped_env

# Parallel environments
vec_env = make_vec_env(make_wrapped_env, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100)
model.save("ppo_flappybird")
