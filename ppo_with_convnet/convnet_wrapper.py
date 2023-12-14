import cv2
import gymnasium as gym
import numpy as np
import torch
from torchvision import models, transforms

class ConvNetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2048,), dtype=np.float32)
        self.env = env

    def pass_through_resnet(self):
        image = self.env.render()
        image = cv2.resize(image, (224, 224))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(transforms.ToTensor()(image))

        resnet50 = models.resnet50(pretrained=True)
        # removing the last layer for inference
        resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

        with torch.no_grad():
            observation = resnet50(image_tensor.unsqueeze(0)) # unsqueeze makes the single image a batch of 1
        observation = observation.flatten()

        return observation
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.pass_through_resnet()

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation = self.pass_through_resnet()

        return observation, reward, terminated, truncated, info