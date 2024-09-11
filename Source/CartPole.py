import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Number of episodes to run
num_episodes = 5

# Run through episodes
for episode in range(num_episodes):
    observation, info = env.reset()  # Reset the environment at the start of each episode
    done = False
    episode_reward = 0
    
    while not done:
        # Random action: 0 (push cart to the left) or 1 (push cart to the right)
        action = env.action_space.sample() 
        
        # Apply the action to the environment and get feedback
        observation, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        
        # If the episode ends (either done or truncated), break the loop
        if done or truncated:
            break

    print(f"Episode {episode + 1} finished with reward: {episode_reward}")

# Close the environment
env.close()
