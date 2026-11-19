import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 16), 
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)

        return logits


class MultilayerPerceptronRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )

    def forward(self, x):
        return self.network(x).squeeze(-1)  # [B, 1] -> [B], o mesmo formato do alvo


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        out_channels = channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)  # stride no 3x3, como no torchvision (ResNet v1.5)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return F.relu(x + shortcut)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 64, n_blocks=3, stride=1)
        self.layer2 = self.make_layer(256, 128, n_blocks=4, stride=2)
        self.layer3 = self.make_layer(512, 256, n_blocks=6, stride=2)
        self.layer4 = self.make_layer(1024, 512, n_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    @staticmethod
    def make_layer(in_channels: int, channels: int, n_blocks: int, stride: int):
        blocks = [Bottleneck(in_channels, channels, stride)] + [Bottleneck(channels * Bottleneck.expansion, channels) for _ in range(n_blocks - 1)]

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)

        return self.fc(x)


class Autoencoder(nn.Module):
    """Autoencoder para dados tabulares (ex.: Iris): n_features -> hidden -> latent -> hidden -> n_features. Sem Sigmoid: as features são padronizadas (podem ser negativas)."""

    def __init__(self, n_features: int, hidden_size: int = 16, latent_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class MultiDiscreteActorCritic(nn.Module):
    """Política (actor) de um PPO treinado no fleet-zero, só a parte obs -> logits: a normalização das observações vem embutida (buffers), então a entrada é a observação crua. O critic não é usado na explicabilidade."""

    def __init__(self, n_observations, n_actions, hidden_size: int = 128, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("obs_mean", torch.zeros(n_observations))
        self.register_buffer("obs_var", torch.ones(n_observations))
        self.actor = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.actor((x - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon))
