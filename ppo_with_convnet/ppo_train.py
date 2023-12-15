import cv2
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.logger import configure


class ImageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,128,72), dtype=np.float32)

    def get_observation(self):
        image = self.env.render()
        image = cv2.resize(image, (72, 128))
        image = np.moveaxis(image, -1, 0)

        return image/255
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.get_observation()

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info


def make_wrapped_env():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    wrapped_env = ImageWrapper(env)
    return wrapped_env

# Parallel environments
vec_env = make_vec_env(make_wrapped_env, n_envs=1)
vec_env = VecFrameStack(vec_env, n_stack=4)

new_logger = configure("./", ["stdout", "csv"])
model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=dict(normalize_images=False))
model.set_logger(new_logger)
model.learn(total_timesteps=250000)
model.save("ppo_flappybird")
