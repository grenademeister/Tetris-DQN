import gym
from gym.envs.registration import register
import torch
from torch import nn

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import itertools
import yaml
import random
import os
from datetime import datetime, timedelta
import argparse
import cv2

# import intel_npu_acceleration_library
# from intel_npu_acceleration_library.nn.module import NPUContextManager
# from intel_npu_acceleration_library.compiler import CompilerConfig

from tetris_env import TetrisEnv
from dqn import DuelingDQN as DQN
from experience_replay import PERReplayMemory as ReplayMemory

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use("Agg")

register(
    id="Tetris-v0",
    entry_point="tetris_env:TetrisEnv",
    kwargs={
        "state_encoding": "binary",
        "reward_params": {"line_clear": 40, "step": 1, "game_over": -10},
        "render_mode": "human",
    },
)


class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters["env_id"]
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.env_make_params = hyperparameters.get("env_make_params", {})
        self.n_step = hyperparameters["n_step"]
        self.reward_multiplier = hyperparameters["reward_multiplier"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")

    def run(self, is_training=True, render=False, resume=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            last_model_save_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")

        # initialize environment
        print(f"Creating environment with render_mode={'human' if render else 'None'}")
        print(self.env_id)
        env = gym.make(
            self.env_id, render_mode="human" if render else None, **self.env_make_params
        )

        # environmental variables
        observation_space_shapes = [
            values.shape for values in env.observation_space.values()
        ]
        num_states = observation_space_shapes[0]
        num_actions = env.action_space.nvec
        channel_size = len(observation_space_shapes)

        rewards_per_episode = []

        # compiler_conf = CompilerConfig(
        #     dtype=torch.float32,
        #     training=True,
        # )

        # initialize DQN
        policy_dqn = DQN(num_states, num_actions, channel_size)
        # policy_dqn = intel_npu_acceleration_library.compile(policy_dqn, compiler_conf)

        if is_training:
            memory = ReplayMemory(
                self.replay_memory_size, self.discount_factor_g, self.n_step
            )
            # load existing model to resume training if resume = True
            if resume:
                policy_dqn = torch.load(self.MODEL_FILE)

            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, channel_size)
            # target_dqn = intel_npu_acceleration_library.compile(
            #     policy_dqn, compiler_conf
            # )
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0
            epsilon_history = []
            best_reward = -9999999

            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate_a
            )
        else:
            # load and evaluate trained model if inference mode
            policy_dqn = torch.load(self.MODEL_FILE)
            policy_dqn.eval()
            print("loaded saved model...")

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(self.process_state(state), dtype=torch.float)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = torch.tensor(
                    self.process_state(new_state), dtype=torch.float
                )

                # tried reward shaping but failed
                # reward = shape_reward(new_state, reward, self.reward_multiplier)
                reward = torch.tensor(reward, dtype=torch.float)

                # Accumulate reward
                episode_reward += reward

                # Calculate TD error for PER
                with torch.no_grad():
                    best_q = policy_dqn(new_state.unsqueeze(dim=0)).squeeze().max()
                    td_error = (
                        reward
                        + self.discount_factor_g * best_q
                        - policy_dqn(state.unsqueeze(dim=0)).squeeze()[action]
                    )
                if is_training:
                    memory.append(
                        (state, action, new_state, reward, terminated),
                        td_error=td_error,
                    )
                    step_count += 1

                # move to new state
                state = new_state

                # render and wait 100ms for every step if inference mode
                if not is_training:
                    env.render()
                    key = cv2.waitKey(100)

            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained
            # or when 30 minutes have passed since last save(only for occasional cases)
            if is_training:
                current_time = datetime.now()
                if (
                    episode_reward > best_reward
                    or current_time - last_model_save_time > timedelta(minutes=30)
                ):
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message + "\n")

                    torch.save(policy_dqn, self.MODEL_FILE)
                    best_reward = episode_reward
                    last_model_save_time = current_time

                # Update graph every 10 seconds
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(resume, rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    # sample batch and optimize the dqn
                    mini_batch_data = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch_data, policy_dqn, target_dqn, memory)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, resume, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)
        # only plot moving average of reward if training is resumed(no need to plot constant epsilon)
        if resume:
            # Plot average rewards (Y-axis) vs episodes (X-axis)
            mean_rewards = np.zeros(len(rewards_per_episode))
            for x in range(len(mean_rewards)):
                mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])
            plt.xlabel("Episodes")
            plt.ylabel("Mean Rewards")
            plt.plot(mean_rewards)
        else:
            # Plot average rewards (Y-axis) vs episodes (X-axis)
            mean_rewards = np.zeros(len(rewards_per_episode))
            for x in range(len(mean_rewards)):
                mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])
            plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
            plt.xlabel("Episodes")
            plt.ylabel("Mean Rewards")
            plt.plot(mean_rewards)
            # Plot epsilon decay (Y-axis) vs episodes (X-axis)
            plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
            plt.xlabel("Time Steps")
            plt.ylabel("Epsilon Decay")
            plt.plot(epsilon_history)
            plt.subplots_adjust(wspace=1.0, hspace=1.0)
        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch_data, policy_dqn, target_dqn, memory):
        # Transpose the list of experiences and separate each element
        mini_batch, weights, indices = mini_batch_data
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float()
        weights = torch.tensor(weights, dtype=torch.float)

        with torch.no_grad():
            # enable double dqn(important)
            best_action_from_policy = policy_dqn(new_states).argmax(dim=1)
            # Calculate target Q values (expected returns)
            target_q = (
                rewards
                + (1 - terminations)
                * self.discount_factor_g
                * target_dqn(new_states)
                .gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1))
                .squeeze()
            )
            """
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
            """

        # Calcuate Q values from current policy
        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )
        """
                policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                    actions.unsqueeze(dim=1)
                    .gather(1, actions.unsqueeze(dim=1))  ==>
                        .squeeze()                    ==>
            """

        # compute TD error
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        # update priority of PER RM
        memory.update_priorities(indices, td_errors)

        # Compute loss
        loss = (weights * (current_q - target_q) ** 2).mean()

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network parameters i.e. weights and biases

    def process_state(self, state):
        # handle state(dictionary) and return combined ndarray to the agent
        board = state["board"]
        tetromino = state["active_tetromino_mask"]
        combined = np.stack([board, tetromino], axis=0)

        return combined


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")
    parser.add_argument("--resume", help="Resume training", action="store_true")
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        if args.resume:
            dql.run(is_training=True, resume=True)
        else:
            dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
