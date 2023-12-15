import cv2
import gymnasium as gym
import numpy as np
import torch
from torchvision import models, transforms

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def pass_through_model(self):
        pass
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.pass_through_model()

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation = self.pass_through_model()

        return observation, reward, terminated, truncated, info
    
class ResNetWrapper(BaseWrapper):
    def pass_through_model(self):
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
    

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fully_connected = torch.nn.Linear(32 * 112 * 112, 200)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                # Weights are initialized to a constant value of 0.01
                torch.nn.init.constant_(module.weight, 0.01) 
                if module.bias is not None:
                    # Biases are initialized to a constant value of 0.01 as well
                    torch.nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
       x = self.pool(torch.nn.functional.relu(self.conv(x)))
       x = x.view(x.size(0), -1)
       x = self.fully_connected(x)
       return x

    
class ConvNetWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(200,), dtype=np.float32)


    def pass_through_model(self):
        image = self.env.render()
        image = cv2.resize(image, (224, 224))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = normalize(transforms.ToTensor()(image))

        with torch.no_grad():
            observation = ConvNet().forward(image_tensor.unsqueeze(0))
        observation.flatten()

        return observation

