import gymnasium as gym
from stable_baselines3 import PPO
import os

# Initialize the LunarLander environment
env = gym.make("LunarLander-v2", render_mode="human")

# Load the trained model
current_directory = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_directory, "ppo_lunarlander_model.zip")
model = PPO.load(model_save_path)

# Reset the environment and test the agent
observation, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(observation)
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()
