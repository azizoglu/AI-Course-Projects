# Import necessary libraries
import gymnasium as gym  # Provides simulation environments like LunarLander
from stable_baselines3 import DQN  # Deep Q-Network algorithm

# Create the LunarLander environment
# "render_mode='human'" opens a simulation window to watch the landing in real time
env = gym.make("LunarLander-v3", render_mode="human")

# Create a model using the DQN algorithm
# "MlpPolicy": uses a multilayer perceptron (neural network)
# "verbose=1": prints step-by-step training info to the console
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent — it learns by trial and error over 100,000 timesteps
model.learn(total_timesteps=100_000)

# After training is complete, start testing the agent
obs, _ = env.reset()  # Reset the environment and get the initial observation

# Test the agent for 1000 steps
for _ in range(1000):
    # Predict the action to take based on the current observation
    action, _states = model.predict(obs)
    
    # Apply the action in the environment, receive new state and reward
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # Render the simulation (display it on screen)
    env.render()
    
    # If the episode is done or max steps are exceeded, reset the environment
    if terminated or truncated:
        obs, _ = env.reset()

# Save the trained model to a file (for later use)
model.save("dqn_lunarlander")

# Close the environment, release the window and resources
env.close()
