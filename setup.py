import gymnasium as gym
import torchvision
import torch
import torch.optim as optim
import wandb
import glob, io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


from PIL import Image
from torch import nn
from functools import cached_property, lru_cache
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from networks import ConvolutionalNeuralNetwork, MultilayerPerceptron, MultilayerPerceptronRegression, ResNet50, Autoencoder, DQN


class ParquetImageDataset(Dataset):
    def __init__(self, pattern: str, transform):
        self.tables = [pq.read_table(file) for file in sorted(glob.glob(pattern))]
        self.offsets = np.cumsum([0] + [len(table) for table in self.tables])
        self.transform = transform

    def __len__(self):
        return int(self.offsets[-1])

    @property
    def targets(self):
        return np.concatenate([table["label"].to_numpy() for table in self.tables])

    def __getitem__(self, index: int):
        shard = np.searchsorted(self.offsets, index, side="right") - 1
        row = self.tables[shard].slice(index - self.offsets[shard], 1).to_pylist()[0]

        return self.transform(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")), row["label"]



IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_LABELS = 3  # quantas labels são sorteadas quando nenhuma é pedida
MAX_TEST_DATA = 1000

LAYER_CONFIG = {
    "MLP": {"layers": ["network.0", "network.2"], "reduction": ["ei_split"]},
    "Convolutional": {"layers": ["conv1", "conv2", "fc1", "fc2"], "reduction": ["ei_split", "spatial_sum"]},
    "ResNet50": {"layers": ["layer1", "layer2", "layer3", "layer4"], "reduction": ["ei_split", "spatial_sum"]},
    "DQN": {"layers": ["layer1", "layer2"], "reduction": ["ei_split"]},
    "MLPRegression": {"layers": ["network.0", "network.2"], "reduction": ["ei_split"]},
}


# O Gymnasium não expõe o que cada índice da observação mede: cadastre aqui os nomes de cada ambiente
OBSERVATION_NAMES = {
    "CartPole-v1": ["posição", "vel. carrinho", "ângulo", "vel. angular"],
    "California_Housing": ["renda mediana", "idade da casa", "média de cômodos", "média de quartos", "população", "média de ocupantes", "latitude", "longitude"],
}

IMAGE_DATASETS = ("Mnist", "Imagenet")
TABULAR_DATASETS = ("California_Housing",)
PRICE_BANDS = ["baixo", "médio", "alto"]  # California Housing: as "classes" são faixas do preço (tercis do treino)


CALIFORNIA_HOUSING_CSV, CALIFORNIA_HOUSING_TARGET = "datasets/California_Housing/california_housing.csv", "MedHouseVal"


@lru_cache(maxsize=1)
def california_housing_splits(train_fraction: float = 0.8, seed: int = 0):
    """{True: treino, False: teste} como TensorDataset(features, alvo); as features são padronizadas com a média e o desvio do treino."""
    data = pd.read_csv(CALIFORNIA_HOUSING_CSV)
    features = data.drop(columns=CALIFORNIA_HOUSING_TARGET).to_numpy(np.float32)
    target = data[CALIFORNIA_HOUSING_TARGET].to_numpy(np.float32)

    order = np.random.default_rng(seed).permutation(len(data))
    train, test = order[:int(train_fraction * len(data))], order[int(train_fraction * len(data)):]
    mean, std = features[train].mean(0), features[train].std(0)

    return {train_split: TensorDataset(torch.from_numpy((features[idx] - mean) / std), torch.from_numpy(target[idx])) for train_split, idx in ((True, train), (False, test))}


@lru_cache(maxsize=1)
def price_band_edges():
    """Cortes (tercis do preço no treino) que separam as faixas PRICE_BANDS: len(PRICE_BANDS) - 1 valores."""
    prices = california_housing_splits()[True].tensors[1].numpy()

    return np.quantile(prices, np.linspace(0, 1, len(PRICE_BANDS) + 1)[1:-1])


def price_band(prices: np.ndarray):
    """Preço (contínuo) -> id da faixa (0 = baixo, 1 = médio, 2 = alto)."""
    return np.digitize(prices, price_band_edges())


def dataset_observation_names(dataset_type: str):
    return OBSERVATION_NAMES.get(dataset_type)  # None: ambiente sem nomes cadastrados (o plot usa o índice)


def dataset_class_names(dataset_type: str):
    if dataset_type == "Imagenet":
        return torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    if dataset_type in TABULAR_DATASETS:
        return PRICE_BANDS

    return [str(digit) for digit in range(10)]


def random_labels(n_classes: int, n_labels: int = 3):
    """Sorteia até `n_labels` ids distintos entre 0 e n_classes - 1, em ordem crescente."""
    return np.sort(np.random.choice(n_classes, size=min(n_labels, n_classes), replace=False))


def class_ids(dataset_type: str, classes):
    names = dataset_class_names(dataset_type)

    return [int(c) if str(c).isdigit() else names.index(c) for c in ([classes] if isinstance(classes, (str, int)) else classes)]


class Setup:
    def __init__(self, dataset_type: str="Mnist", model_type: str="MLP", loss_type: str="CrossEntropy", optimizer_type: str="Adam", learning_rate: float=0.001, layer_names: list=None, reduction: list=None, classes: list=None, render: bool=False, experiment_name: str=""):
        self.experiment_name = experiment_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type, self.model_type, self.optimizer_type, self.learning_rate = dataset_type, model_type, optimizer_type, learning_rate
        self.model_kwargs = {}
        self.environment, self.train_dataloader, self.test_dataloader, self.class_names = None, None, None, None

        if dataset_type in ("Mnist", "Imagenet"):
            self.train_dataloader, self.test_dataloader = self.create_dataloaders()
        elif dataset_type == "California_Housing":
            self.train_dataloader, self.test_dataloader = self.create_dataloaders()
            self.model_kwargs = {"n_features": california_housing_splits()[True].tensors[0].shape[1]}
        else:
            self.environment = gym.make(dataset_type, render_mode="human" if render else None)
            self.model_kwargs = {"n_observations": len(self.environment.reset()[0]), "n_actions": self.environment.action_space.n}

        n_classes = self.model_kwargs["n_actions"] if self.environment is not None else len(dataset_class_names(dataset_type))
        self.labels = random_labels(n_classes, N_LABELS) if classes is None or len(np.atleast_1d(classes)) == 0 else np.sort(class_ids(dataset_type, classes))

        self.model = self.create_model()
        self.criterion = self.create_loss(loss_type)
        self.optimizer = self.create_optimizer(self.model, optimizer_type, learning_rate)

        if model_type in LAYER_CONFIG:
            self.layers = [self.model.get_submodule(name) for name in layer_names or LAYER_CONFIG[model_type]["layers"]]
            self.reduction = reduction or LAYER_CONFIG[model_type]["reduction"]

        self._test_data = {}


    def create_dataloaders(self):
        return self.create_dataloader(train=True), self.create_dataloader(train=False)


    def create_dataloader(self, train: bool):
        if self.dataset_type == "Mnist":
            data = torchvision.datasets.MNIST("datasets", train=train, download=train, transform=transforms.ToTensor())

            return DataLoader(data, batch_size=64, num_workers=2, pin_memory=True, shuffle=train, persistent_workers=True)

        elif self.dataset_type == "California_Housing":
            return DataLoader(california_housing_splits()[train], batch_size=64, shuffle=train)  # tudo em memória: sem workers

        elif self.dataset_type == "Imagenet":
            normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

            if train:
                transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            else:
                transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

            data = ParquetImageDataset(f"datasets/Imagenet/data/{'train' if train else 'validation'}-*.parquet", transform=transform)

            return DataLoader(data, batch_size=64, num_workers=2, pin_memory=True, shuffle=train, persistent_workers=True, prefetch_factor=4)


    def create_class_names(self):
        names = dataset_class_names(self.dataset_type)

        return [names[c] for c in self.labels]


    @staticmethod
    def balanced_indices(targets: np.ndarray, per_class: int):
        order = np.argsort(targets, kind="stable")
        rank = np.arange(len(order)) - np.searchsorted(targets[order], targets[order])

        return np.sort(order[rank < per_class])


    def extract_test_data(self, only_labels: bool):
        loader = self.test_dataloader if self.test_dataloader is not None else self.create_dataloader(train=False)
        dataset = loader.dataset

        if self.is_tabular:
            features, prices = dataset.tensors
            targets = price_band(prices.numpy())  # regressão: a "classe" de cada amostra é a faixa do preço
        else:
            targets = np.asarray(dataset.targets)

        if only_labels:
            classes = self.labels
            indices = np.flatnonzero(np.isin(targets, classes))[:MAX_TEST_DATA]
        else:
            classes = np.unique(targets)
            indices = self.balanced_indices(targets, max(1, MAX_TEST_DATA // len(classes)))

        if self.is_tabular:
            return features[indices], np.searchsorted(classes, targets[indices])

        xs, ys = zip(*DataLoader(Subset(dataset, indices), batch_size=64, num_workers=4, shuffle=False))

        return torch.cat(xs), np.searchsorted(classes, torch.cat(ys).numpy())


    def test_data(self, only_labels: bool = False):
        if only_labels not in self._test_data:
            self._test_data[only_labels] = self.extract_test_data(only_labels)

            self.train_dataloader = self.test_dataloader = None
            pa.default_memory_pool().release_unused()

        return self._test_data[only_labels]


    def test_observations(self, path: str):
        """Observações (e ações escolhidas) do agente treinado, salvas por reinforcement_learning.main."""
        data = torch.load(f"{path}/observations.pt", map_location="cpu")

        return data["observations"], data["actions"].numpy()


    def create_model(self):
        if self.model_type == "Convolutional":
            return ConvolutionalNeuralNetwork().to(self.device)

        elif self.model_type == "MLP":
            return MultilayerPerceptron().to(self.device)

        elif self.model_type == "MLPRegression":
            return MultilayerPerceptronRegression(**self.model_kwargs).to(self.device)

        elif self.model_type == "ResNet50":
            return ResNet50().to(self.device)

        elif self.model_type == "Autoencoder":
            return Autoencoder().to(self.device)

        elif self.model_type == "DQN":
            return DQN(**self.model_kwargs).to(self.device)


    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))

        return self.model.eval()


    def create_loss(self, loss_type: str="CrossEntropy"):
        if loss_type == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif loss_type == "SmoothL1":
            return nn.SmoothL1Loss()
        elif loss_type == "MSE":
            return nn.MSELoss()


    def create_optimizer(self, model: nn.Module, optimizer_type: str="Adam", learning_rate: float=0.001):
        if optimizer_type == "Adam":
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            return optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == "AdamW":
            return optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)

    def init_wandb(self, run_name: str, config=None):
        return wandb.init(entity="ResidenciaIA", project="Explicabilidade_Repo", name=f"{self.experiment_name}_{run_name}" if self.experiment_name else run_name, dir="runs/", config=config)

    @property
    def is_image(self):
        return self.dataset_type in IMAGE_DATASETS

    @property
    def is_tabular(self):
        return self.dataset_type in TABULAR_DATASETS

    @property
    def is_reinforcement(self):
        return self.environment is not None

    @cached_property
    def visualizer(self):
        from visualization.visualization import Visualization

        return Visualization(dataset=self.dataset_type, network=self.model_type, class_names=self.class_names, experiment_name=self.experiment_name)

