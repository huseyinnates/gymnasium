import gymnasium as gym
from stable_baselines3 import PPO
import os

# Humanoid ortamını başlat
env = gym.make("Humanoid-v4", render_mode="human")

# Eğitilmiş modeli yükle
current_directory = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_directory, "ppo_humanoid_model.zip")
model = PPO.load(model_save_path)

# Ortamı sıfırla ve ajanı test et
observation, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(observation)
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()
