import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 5

for episode in range(num_episodes):
    observation, info = env.reset() 
    done = False
    episode_reward = 0
    
    while not done:
        action = env.action_space.sample() 
        
        observation, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        
        if done or truncated:
            break

    print(f"Episode {episode + 1} finished with reward: {episode_reward}")

env.close()
