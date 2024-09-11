import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
import os

#CUDA'nın etkin olup olmadığını kontrol edin
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Customize the policy network architecture
policy_kwargs = dict(
    net_arch=[256, 256]  # Daha büyük sinir ağı
)

# Create the DQN model
model = DQN(
    'MlpPolicy',  # DQN uses a Multi-Layer Perceptron (MLP) policy
    env, 
    learning_rate=0.00005,  # Daha düşük öğrenme oranı
    batch_size=128,         # Daha büyük batch size
    buffer_size=50000,      # Daha büyük replay buffer
    target_update_interval=2000,  # Hedef ağın daha seyrek güncellenmesi
    exploration_fraction=0.2,  # Keşif oranını artırıyoruz
    exploration_final_eps=0.01,  # Daha düşük final epsilon değeri
    policy_kwargs=policy_kwargs, 
    verbose=1,
    device='cuda'  # GPU kullanıyoruz
)

# Training timesteps
timesteps = 100000
eval_interval = 10000
reward_history = []  # Store the mean reward over time

# Variable to store the best reward and model
best_mean_reward = -float('inf')  # Initially set to negative infinity
model_save_path = "best_model.zip"

# Train the model and evaluate it periodically
for i in range(0, timesteps, eval_interval):
    model.learn(total_timesteps=eval_interval)
    
    # Evaluate the policy for 10 episodes and record the mean reward
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    reward_history.append(mean_reward)
    print(f"Mean reward after {i + eval_interval} timesteps: {mean_reward}")

    # Save the model if the mean reward is the best so far
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        print(f"New best mean reward: {best_mean_reward}, saving model...")
        model.save(model_save_path)  # Save the best model
        # Optionally, save the mean reward value as well (in a text file)
        with open("best_reward.txt", "w") as f:
            f.write(f"Best mean reward: {best_mean_reward}")

# Plotting the reward history to visualize performance over time
plt.plot(range(0, timesteps, eval_interval), reward_history)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('DQN Performance on CartPole-v1 with Best Model Saving')
plt.show()

# -------------------------------------------
# Loading the best model and visualizing it
# -------------------------------------------

# Load the saved best model
print("Loading the best model...")
best_model = DQN.load(model_save_path)

# Test the best model and visualize its performance
num_episodes = 5
episode_rewards = []

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Use the best model to predict the action
        action, _states = best_model.predict(observation)
        
        # Apply the action to the environment
        observation, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        
        if done or truncated:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1} finished with reward: {episode_reward}")

# Plot the rewards obtained in the test episodes
plt.figure()
plt.bar(range(1, num_episodes + 1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Performance of the Best Model on Test Episodes')
plt.show()

# Close the environment
env.close()