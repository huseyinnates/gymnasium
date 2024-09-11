import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Ortamı oluştur
env = make_vec_env('CartPole-v1', n_envs=1)

# Modeli tanımla ve eğit
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Modeli kaydet
model.save("best_model")
