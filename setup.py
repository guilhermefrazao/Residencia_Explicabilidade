import torchvision
import torch
import torch.optim as optim
import wandb

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from networks import ConvolutionalNeuralNetwork, MultilayerPerceptron



def setup_dataset(dataset_type: str="Mnist"):
    if dataset_type == "Mnist":
        transform = transforms.ToTensor()

        train_data, test_data = torchvision.datasets.MNIST("datasets", train=True, download=True, transform=transform), torchvision.datasets.MNIST("datasets", train=False, download=False, transform=transform)

        return DataLoader(train_data, batch_size=64, num_workers=4, pin_memory=True, shuffle=True), DataLoader(test_data, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    else:
        pass


def setup_model(model_type: str="MLP", device: str="cuda"):
    if model_type == "Convolutional":
        return ConvolutionalNeuralNetwork().to(device)

    elif model_type == "MLP":
        return MultilayerPerceptron().to(device)
    else:
        pass


def setup_loss_type(loss_type="CrossEntropy"):
    if loss_type == "CrossEntropy":
        return nn.CrossEntropyLoss()
    else:
        pass


def setup_optimizer(model: nn.Module, optimizer_type: str= "Adam", learning_rate: float= 0.001):
    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    else:
        pass


def setup_wandb(run_name: str, config=None):
    return wandb.init(entity="ResidenciaIA", project="Explicabilidade_Repo", name=run_name, dir="runs/", config=config)
