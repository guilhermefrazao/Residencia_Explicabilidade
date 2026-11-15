import gymnasium as gym
import torchvision
import torch
import torch.optim as optim
import wandb
import glob, io
import numpy as np
import pyarrow.parquet as pq


from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from networks import ConvolutionalNeuralNetwork, MultilayerPerceptron, ResNet50, DQN


class ParquetImageDataset(Dataset):
    def __init__(self, pattern: str, transform):
        self.tables = [pq.read_table(file) for file in sorted(glob.glob(pattern))]
        self.offsets = np.cumsum([0] + [len(table) for table in self.tables])
        self.transform = transform

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        shard = np.searchsorted(self.offsets, index, side="right") - 1
        row = self.tables[shard].slice(index - self.offsets[shard], 1).to_pylist()[0]

        return self.transform(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")), row["label"]



LAYER_CONFIG = {
    "MLP": {"layers": ["network.0", "network.2"], "reduction": ["ei_split"]},
    "Convolutional": {"layers": ["conv1", "conv2", "fc1", "fc2"], "reduction": ["ei_split"]},
    "ResNet50": {"layers": ["layer1", "layer2", "layer3", "layer4"], "reduction": ["ei_split"]},
}


def setup_dataset(dataset_type: str="Mnist"):
    if dataset_type == "Mnist":
        transform = transforms.ToTensor()

        train_data, test_data = torchvision.datasets.MNIST("datasets", train=True, download=True, transform=transform), torchvision.datasets.MNIST("datasets", train=False, download=False, transform=transform)

        return DataLoader(train_data, batch_size=64, num_workers=4, pin_memory=True, shuffle=True), DataLoader(test_data, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    elif dataset_type == "Imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

        train_data, test_data = ParquetImageDataset("datasets/Imagenet/data/train-*.parquet", transform=train_transform), ParquetImageDataset("datasets/Imagenet/data/validation-*.parquet", transform=test_transform)

        return DataLoader(train_data, batch_size=64, num_workers=8, pin_memory=True, shuffle=True), DataLoader(test_data, batch_size=64, num_workers=8, pin_memory=True, shuffle=False)
    else:
        pass


def setup_test_data(test_dataloader: DataLoader):
    dataset = test_dataloader.dataset

    return dataset.data.float().unsqueeze(1) / 255, dataset.targets.numpy()



def setup_model(model_type: str="MLP", device: str="cuda", **model_kwargs):
    if model_type == "Convolutional":
        return ConvolutionalNeuralNetwork().to(device)

    elif model_type == "MLP":
        return MultilayerPerceptron().to(device)

    elif model_type == "ResNet50":
        return ResNet50().to(device)

    else:
        pass


def setup_layers(model: nn.Module, model_type: str, layer_names: list=None, reduction: list=None):
    config = LAYER_CONFIG[model_type]

    layers = [model.get_submodule(name) for name in layer_names or config["layers"]]

    return layers, reduction or config["reduction"]


def setup_loss_type(loss_type="CrossEntropy"):
    if loss_type == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "SmoothL1":
        return nn.SmoothL1Loss()
    else:
        pass


def setup_optimizer(model: nn.Module, optimizer_type: str= "Adam", learning_rate: float= 0.001):
    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    else:
        pass


def setup_wandb(run_name: str, config=None):
    return wandb.init(entity="ResidenciaIA", project="Explicabilidade_Repo", name=run_name, dir="runs/", config=config)
