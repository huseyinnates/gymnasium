import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os
 
video_save_path = "videos/"
os.makedirs(video_save_path, exist_ok=True)
 
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_save_path, episode_trigger=lambda x: True)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_directory, "best_model.zip")   
print(f"Loading the best model from {model_save_path}...")
best_model = DQN.load(model_save_path)

num_episodes = 5
episode_rewards = []
 
for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _states = best_model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        if done or truncated:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1} finished with reward: {episode_reward}")
 
plt.figure()
plt.bar(range(1, num_episodes + 1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Performance of the Best Model on Test Episodes')
plt.show()
 
env.close()
 
print(f"Videos are saved in {video_save_path}")
