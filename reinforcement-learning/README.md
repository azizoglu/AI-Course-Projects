🧠 Reinforcement Learning

Welcome! This repository provides a foundational and practical introduction to **Reinforcement Learning (RL)** — a type of machine learning where an agent learns to make decisions by interacting with its environment.

---

## 📂 Project Structure

```text
.
├── frozen_lake.py           # Q-Learning algorithm implementation on the FrozenLake environment
├── lunar_lander.py          # Deep Q-Network (DQN) implementation using the LunarLander-v3 environment
├── requirements.txt         # List of Python dependencies required to run the project
└── README.md                # Project documentation (you're reading it!)
```
---

## 📚 Contents

### 🧩 Core Concepts Covered

Throughout this module, you will explore key RL principles, including:

- **Agent** and **Environment**  
- **Action** and **Reward**  
- **Markov Property**  
- **Discount Factor**  
- **Exploration vs. Exploitation Trade-Off**

These elements form the basis of how agents learn and adapt over time.

---

### 🛠️ RL Approaches

#### 🔹 Value-Based Methods
- Focus on estimating the value of actions or states.
- Learn about **Q-Learning**, a tabular approach to finding optimal policies.
- Understand how **Deep Q-Networks (DQN)** extend Q-Learning using neural networks.

#### 🔸 Policy-Based Methods
- Learn how agents can directly learn policies without estimating value functions.
- Understand their usefulness in high-dimensional or continuous action spaces.

---

#### 🧊 FrozenLake with Q-Learning  
Apply the **Q-Learning** algorithm to solve the classic **FrozenLake** environment using a discrete grid world setup. This environment is ideal for introducing tabular reinforcement learning in a simple and visual context.

You will:
- Understand the core loop of Q-Learning  
- Build and update a Q-table from agent interactions  
- Balance exploration and exploitation  
- Observe the agent navigating to the goal through learned behavior

---

#### 🚀 LunarLander with Deep Q-Network (DQN)  
Implement a **Deep Q-Network (DQN)** to solve the more complex and physics-based **LunarLander-v3** environment. This task demonstrates how neural networks can be used to approximate Q-values in environments with continuous observations.

You will:
- Use `stable-baselines3` to set up and train a DQN agent  
- Visualize the lander’s progress during training and inference  
- Understand how deep RL can handle more sophisticated state spaces  
- Save and reuse the trained model for evaluation

---

## 🔗 Recommended Resources  
Key references to support your learning on **reinforcement learning**, including environments, foundational theory, and visual explanations:

### 📘 Core References:

- [Deep Reinforcement Learning Course by Hugging Face](https://huggingface.co/learn/deep-rl-course/unit0/introduction)  
  A beginner-friendly, interactive course covering key RL concepts and implementations.
  
- [Gymnasium Documentation](https://gymnasium.farama.org/)  
  Official docs for the Gymnasium library — a suite of environments used to develop and benchmark RL algorithms.
  
- [Reinforcement Learning: An Introduction (Sutton & Barto, 2nd Edition)](http://incompleteideas.net/book/RLbook2020.pdf)  
  The foundational textbook in RL — a must-read for theory and algorithms.

### 🎥 Video Channels:

- [AI Warehouse (YouTube)](https://www.youtube.com/@aiwarehouse)  
  Features agent training simulations in various environments — a great way to **visualize how agents learn** through trial and error in reinforcement learning.

- [CatNavi Desk (YouTube)](https://www.youtube.com/@CatNaviDesk)  
  Demonstrates a cat learning to press a bell to receive food — a simple and intuitive analogy to understand the **reward system in reinforcement learning**.



## 🚀 Getting Started

### ✅ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/azizoglu/AI-Course-Projects.git
cd reinforcement-learning
pip install -r requirements.txt
