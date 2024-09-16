import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch
 
current_directory = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_directory, "logs/ppo_lunarlander")
os.makedirs(log_dir, exist_ok=True)
 
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
 
env = gym.make("LunarLander-v2")
 
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    tensorboard_log=log_dir,
    verbose=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
 
timesteps = 1000000   
model.learn(total_timesteps=timesteps)

# Save the trained model
model_save_path = os.path.join(current_directory, "ppo_lunarlander_model.zip")
model.save(model_save_path)

# Close the environment after training
env.close()
