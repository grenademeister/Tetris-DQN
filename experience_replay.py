from collections import deque
import random
import numpy as np


class ReplayMemory:
    def __init__(self, maxlen, gamma, n_step=1, seed=None):
        self.memory = deque([], maxlen=maxlen)
        self.gamma = gamma
        if seed is not None:
            random.seed(seed)

    def append(self, transition, td_error):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class PERReplayMemory:
    def __init__(self, maxlen, gamma, alpha=0.6, beta=0.4, n_step=1, seed=None):
        self.memory = deque([], maxlen=maxlen)
        self.priorities = deque([], maxlen=maxlen)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if seed is not None:
            random.seed(seed)

    def append(self, transition, td_error):
        self.memory.append(transition)
        priority = PERReplayMemory.get_priority(td_error)
        self.priorities.append(priority)

    def sample(self, sample_size):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), sample_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Importance Sampling Weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = PERReplayMemory.get_priority(td_error)

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def get_priority(td_error, epsilon=1e-6):
        return abs(td_error) + epsilon


class NStepReplayMemory:
    def __init__(self, maxlen, gamma, n_step=1, seed=None):
        self.memory = deque([], maxlen=maxlen)
        self.n_step_buffer = deque([], maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, new_state, reward, termination = self.n_step_buffer[0]
        total_reward = sum(
            self.n_step_buffer[i][3] * (self.gamma**i) for i in range(self.n_step)
        )
        next_state, done = self.n_step_buffer[-1][2], self.n_step_buffer[-1][4]
        self.memory.append((state, action, next_state, total_reward, done))

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    memory = PERReplayMemory(maxlen=10, gamma=1)
    memory.append((1, 2, 3), 1)
    memory.append((1, 2, 4), 2)
    memory.append((1, 2, 5), 3)
    print(memory.sample(2))
