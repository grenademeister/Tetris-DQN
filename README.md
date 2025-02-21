# Tetris-DQN
rainbow dqn implementation for tetris

This project implements a Rainbow Deep Q-Network (Rainbow DQN) to solve the classic game of Tetris. Rainbow DQN integrates several advanced techniques in Deep Reinforcement Learning to achieve superior performance compared to standard DQN algorithms.

## Project Overview

The primary goal is to train an AI agent to play Tetris efficiently, maximizing the score by clearing lines and surviving for as long as possible. The agent leverages the following Rainbow DQN enhancements:

- **Double DQN**: Reduces overestimation bias in Q-value predictions.
- **Dueling DQN**: Separates state-value estimation from action advantages.
- **Prioritized Experience Replay (PER)**: Focuses learning on important experiences by prioritizing them based on their significance.
- **Multi-Step Learning**: Considers multiple future steps for more stable and efficient learning.
- **Noisy Networks**: Adds noise to the network parameters for better and more adaptive exploration.

## Environment

The project uses a custom Tetris environment compatible with OpenAI Gym, featuring:
- **State Representation**: Binary encoding of the Tetris board and active tetromino.
- **Reward Structure**:
  - Line clear: +40
  - Step reward: +1
  - Game over penalty: -10

## Requirements

- Python 3.7+
- PyTorch 2.5.1+
- OpenAI Gym 0.26.2+
- NumPy 1.26.4+
- Matplotlib 3.10.0+
- PyYAML 6.0.2+

Install dependencies:
```bash
pip install torch gym numpy matplotlib pyyaml
```

## How to Run

### Training

To train the agent from scratch:
```bash
python agent.py <hyperparameter_set_name> --train
```

To resume training from an existing model:
```bash
python agent.py <hyperparameter_set_name> --train --resume
```

### Evaluation

To evaluate a trained model:
```bash
python agent.py <hyperparameter_set_name>
```

## Project Structure
```
.
├── agent.py                # Main script for training and evaluating the agent
├── dqn.py                  # Implementation of DQN architectures
├── experience_replay.py    # Replay memory implementations
├── tetris_env.py           # Custom Tetris environment
├── hyperparameters.yml     # Configuration file for hyperparameters
└── runs/                   # Directory containing logs, saved models, and graphs
```

## Results

Training progress and results are logged in the `runs/` directory, including saved models (`.pt` files) and performance graphs (`.png` files).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

