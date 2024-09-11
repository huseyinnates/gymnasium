import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
import os 

current_directory = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_directory, "logs/dqn_cartpole")  
os.makedirs(log_dir, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

env = gym.make("CartPole-v1")

# Hidden layers 4x512x512x2
policy_kwargs = dict(
    net_arch=[512, 512, 512]  
)

model = DQN(
    'MlpPolicy', 
    env, 
    learning_rate=0.00001,  
    batch_size=256,         
    buffer_size=50000,      
    target_update_interval=2000, 
    exploration_fraction=0.45,  
    exploration_final_eps=0.005,  
    policy_kwargs=policy_kwargs, 
    verbose=0,
    tensorboard_log=log_dir,   
    device='cuda'  
)

timesteps = 300000  
eval_interval = 10000
reward_history = [] 

best_mean_reward = -float('inf') 
model_save_path = os.path.join(current_directory, "best_model.zip")   

# Training loop
for i in range(0, timesteps, eval_interval):
    model.learn(total_timesteps=eval_interval)

    percent_complete = (i + eval_interval) / timesteps * 100
    print(f"Training Progress: {percent_complete:.2f}% complete.")
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    reward_history.append(mean_reward)
    print(f"Mean reward after {i + eval_interval} timesteps: {mean_reward}")
    
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        print(f"New best mean reward: {best_mean_reward}, saving model...")
        model.save(model_save_path) 
        
        with open(os.path.join(current_directory, "best_reward.txt"), "w") as f:
            f.write(f"Best mean reward: {best_mean_reward}")

plt.plot(range(0, timesteps, eval_interval), reward_history)
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('DQN Performance on CartPole-v1 with Best Model Saving')
plt.show()

env.close()
