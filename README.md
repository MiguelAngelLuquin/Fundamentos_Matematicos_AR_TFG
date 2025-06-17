# Mathematical Foundations of Reinforcement Learning – Final Degree Project

This repository contains the code developed for the final undergraduate thesis in Mathematics at the University of Zaragoza. The project focuses on the mathematical foundations of **Reinforcement Learning (RL)**, with a special emphasis on the **Q-learning** algorithm and its deep learning extensions.

The implementations demonstrate how RL can be applied to different environments of increasing complexity.

## 📁 Project Structure

The repository is organized into three main directories, each corresponding to a practical case study analyzed in the thesis:

- [`FrozenLake/`](./FrozenLake):  
  Classical Q-learning applied to the Frozen Lake environment. Includes a basic implementation and experimentation with different hyperparameters.

- [`tictactoe_qlearning/`](./tictactoe_qlearning):  
  Custom Q-learning implementation for the Tic-Tac-Toe game. The agent trains through self-play to learn an optimal strategy.

- [`CartPole-DQN/`](./CartPole-DQN):  
  Deep Q-learning using a neural network to balance a pole on a moving cart (CartPole environment). Implements experience replay, target networks, and training routines using PyTorch.

## 🧠 Topics Covered

- Markov Decision Processes (MDPs)
- Bellman equations
- Tabular Q-learning
- Exploration vs. Exploitation (ε-greedy strategies)
- Deep Q-Networks (DQN) with function approximation
- Empirical analysis of convergence and hyperparameter tuning

## 🚀 Requirements

Install dependencies (example for Python projects):

```bash
pip install gymnasium torch numpy
