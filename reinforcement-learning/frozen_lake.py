# Import necessary libraries
import gymnasium as gym  # To simulate the environment
import numpy as np       # For numerical operations and the Q-table

# Create the FrozenLake environment (non-slippery surface, graphical interface)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Get the total number of states and actions in the environment
state_size = env.observation_space.n    # 4x4 grid → 16 states
action_size = env.action_space.n        # 4 directions: left, down, right, up

# Initialize the Q-table with zeros (size: state x action)
q_table = np.zeros((state_size, action_size))

# Hyperparameters for the Q-learning algorithm
learning_rate = 0.8         # How quickly to learn
discount_rate = 0.95        # Influence of future rewards
episodes = 10               # Number of episodes (trials) to run
max_steps = 100             # Max steps per episode
epsilon = 1.0               # Initial exploration rate (100% random actions)
epsilon_decay = 0.995       # Reduce epsilon after each episode
epsilon_min = 0.01          # Minimum value for epsilon

# Q-learning loop (runs for the number of episodes)
for episode in range(episodes):
    state, _ = env.reset()  # Reset the environment and get the initial state
    done = False            # Game is not over yet
    for step in range(max_steps):
        # Choose action using ε-greedy strategy (explore or exploit)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()        # Take a random action (exploration)
        else:
            action = np.argmax(q_table[state])        # Take the best known action (exploitation)

        # Apply the selected action and get the new state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated                # Check if the episode is over

        # Update the Q-table using the Q-learning formula
        old_value = q_table[state, action]            # Old Q-value
        next_max = np.max(q_table[next_state])        # Max Q-value for the next state
        new_value = old_value + learning_rate * (reward + discount_rate * next_max - old_value)
        q_table[state, action] = new_value            # Write the updated value

        state = next_state                            # Update the current state

        if done:
            break                                     # Exit loop if the game is over

    # Decay the exploration rate (epsilon) over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Print the learned Q-table at the end of training
print("🎯 Learned Q-Table:")
print(np.round(q_table, 2))  # Print rounded Q-values for readability

# After training, run one episode using the learned policy
state, _ = env.reset()       # Reset the environment
env.render()                 # Render the initial state

for step in range(max_steps):
    action = np.argmax(q_table[state])                # Select the best known action
    next_state, reward, terminated, truncated, _ = env.step(action)  # Apply action and get result
    env.render()                                      # Render the environment after each step
    state = next_state

    if terminated or truncated:
        print("🏁 Game Over. Reward:", reward)         # Print result when the game ends
        break

env.close()  # Close the environment
