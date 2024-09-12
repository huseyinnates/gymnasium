import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import torch

# Eğitim logları için klasör oluştur
current_directory = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_directory, "logs/ppo_humanoid")
os.makedirs(log_dir, exist_ok=True)

# CUDA'nın mevcut olup olmadığını kontrol et
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Humanoid ortamını başlat
env = gym.make("Humanoid-v4")

# PPO modelini başlat
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    tensorboard_log=log_dir,
    verbose=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Modeli eğit
timesteps = 100000  # Eğitmek istediğiniz adım sayısı
model.learn(total_timesteps=timesteps)

# Eğitilen modeli kaydet
model_save_path = os.path.join(current_directory, "ppo_humanoid_model.zip")
model.save(model_save_path)

# Ortamı kapat
env.close()
