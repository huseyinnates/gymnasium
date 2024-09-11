import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
import os
import cv2

current_directory = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_directory, "logs/ppo_car_racing")
os.makedirs(log_dir, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

env = gym.make("CarRacing-v2")

model = PPO(
    "CnnPolicy",
    env, 
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log=log_dir,
    verbose=1,
    device='cuda'
)

def resize_observation(observation, width=84, height=84):
    return cv2.resize(observation, (width, height), interpolation=cv2.INTER_AREA)

class ResizeEnvWrapper(gym.Wrapper):
    def __init__(self, env, width=84, height=84):
        super(ResizeEnvWrapper, self).__init__(env)
        self.width = width
        self.height = height

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs_resized = resize_observation(obs, width=self.width, height=self.height)
        return obs_resized, reward, done, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        obs_resized = resize_observation(obs, width=self.width, height=self.height)
        return obs_resized, info

env = ResizeEnvWrapper(env, width=84, height=84)

timesteps = 10000

model.learn(total_timesteps=timesteps, tb_log_name="PPO_CarRacing")

model_save_path = os.path.join(current_directory, "ppo_car_racing_model.zip")
model.save(model_save_path)

env.close()
