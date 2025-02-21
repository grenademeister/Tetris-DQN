import torch
from torch import nn
import torch.nn.functional as F
import time

import intel_npu_acceleration_library
from intel_npu_acceleration_library.nn.module import NPUContextManager
from intel_npu_acceleration_library.compiler import CompilerConfig

device = torch.device("npu")


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, channel_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_output_size = (state_dim[1] // 2) * (state_dim[0] // 2) * 32
        hidden_size = 64

        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, channel_size):
        super(DuelingDQN, self).__init__()

        # Convolutional Layers (Feature Extractor)
        self.conv1 = nn.Conv2d(
            in_channels=channel_size,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # stride=2로 다운샘플링
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully Connected Layers
        conv_output_size = (
            (state_dim[1] // 2) * (state_dim[0] // 2) * 64
        )  # 마지막 Conv output 크기
        hidden_size = 256

        # Value Stream
        self.fc_value = nn.Linear(conv_output_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        # Advantage Stream
        self.fc_advantage = nn.Linear(conv_output_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        # Value Stream
        value = F.relu(self.fc_value(x))
        value = self.value(value)

        # Advantage Stream
        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # Dueling DQN 공식 적용
        q_val = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_val


class SimpleDuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, channel_size):
        super(SimpleDuelingDQN, self).__init__()
        out_channels = 64
        # Convolutional Layers (Feature Extractor)
        self.conv1 = nn.Conv2d(
            in_channels=channel_size,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # Fully Connected Layers

        conv_output_size = (
            (state_dim[1]) * (state_dim[0]) * out_channels
        )  # 마지막 Conv output 크기
        hidden_size = 64

        # Value Stream
        self.fc_value = nn.Linear(conv_output_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        # Advantage Stream
        self.fc_advantage = nn.Linear(conv_output_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)

        # Value Stream
        value = F.relu(self.fc_value(x))
        value = self.value(value)

        # Advantage Stream
        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # Dueling DQN 공식 적용
        q_val = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_val


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for weight and bias
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise parameters
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        std = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.sigma_init * std)
        self.bias_mu.data.uniform_(-std, std)
        self.bias_sigma.data.fill_(self.sigma_init * std)

    def reset_noise(self):
        eps_in = self.f_epsilon(self.in_features)
        eps_out = self.f_epsilon(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def f_epsilon(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyDuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, channel_size):
        super(NoisyDuelingDQN, self).__init__()

        # Convolutional Layers (Feature Extractor)
        self.conv1 = nn.Conv2d(
            in_channels=channel_size,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        # Fully Connected Layers
        conv_output_size = (
            (state_dim[1]) * (state_dim[0]) * 64
        )  # 마지막 Conv output 크기
        hidden_size = 512

        # Value Stream
        self.fc_value = NoisyLinear(conv_output_size, hidden_size)
        self.value = NoisyLinear(hidden_size, 1)

        # Advantage Stream
        self.fc_advantage = NoisyLinear(conv_output_size, hidden_size)
        self.advantage = NoisyLinear(hidden_size, action_dim)

    def forward(self, x):
        # CNN Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)

        # Value Stream
        value = F.relu(self.fc_value(x))
        value = self.value(value)

        # Advantage Stream
        advantage = F.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)

        # Dueling DQN 공식 적용
        q_val = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_val

    def reset_noise(self):
        self.value.reset_noise()
        self.advantage.reset_noise()


if __name__ == "__main__":
    state_dim = (20, 10)
    action_dim = 40

    net_1 = DuelingDQN(state_dim, action_dim, 2)
    net_2 = NoisyDuelingDQN(state_dim, action_dim, 2)

    start_time = time.time()
    for _ in range(1000):
        state = torch.randn((10, 2, 20, 10))
        output = net_1(state)

    print(f"time of net 1: {time.time()-start_time:.2f}")
    start_time = time.time()

    for _ in range(1000):
        state = torch.randn((10, 2, 20, 10))
        output = net_2(state)
        net_2.reset_noise()
    print(f"time of net 2: {time.time()-start_time:.2f}")
    start_time = time.time()

    net_2 = net_2.to("npu")
    for _ in range(1000):
        state = torch.randn((10, 2, 20, 10))
        output = net_2(state)
        net_2.reset_noise()
    print(f"time of net 2 npu: {time.time()-start_time:.2f}")

    output = net_1(state)
    output = net_2(state)
