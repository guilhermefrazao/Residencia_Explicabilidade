import yaml

from dataclasses import dataclass, fields
from typing import Optional

LEARNING_TYPES = ("supervized", "unsupervized", "reinforcement")


@dataclass
class Configs:
    """Todos os parâmetros que podem ser passados para treinar/avaliar; cada experimento é um .yml com um subconjunto deles."""

    # identificação
    experiment_name: str = ""  # prefixo da pasta de figuras (images/) e dos runs do wandb; o checkpoint não leva prefixo
    learning_type: str = "supervized"  # supervized | unsupervized | reinforcement
    checkpoints: str = "test"  # nome base da pasta, dentro de checkpoint/

    # dados e modelo
    dataset: str = "Mnist"
    model: str = "MLP"
    loss_type: str = "CrossEntropy"
    optimizer_type: str = "Adam"
    learning_rate: float = 0.001
    n_epochs: int = 20

    # interpretação (contribuições, SAE, mapas, ablação, preservação)
    layer_names: Optional[list] = None
    reduction: Optional[list] = None
    classes: Optional[list] = None  # nomes ou ids; None = sorteia
    ig_batch_size: int = 16

    # aprendizado por reforço (só com learning_type: reinforcement)
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 2500
    memory_capacity: int = 10000
    num_episodes: int = 600
    n_test_observations: int = 1000
    render: bool = False

    def __post_init__(self):
        if self.learning_type not in LEARNING_TYPES:
            raise ValueError(f"learning_type deve ser um de {LEARNING_TYPES}, recebido: {self.learning_type}")

        if self.classes is not None and not isinstance(self.classes, list):
            self.classes = [self.classes]

    @property
    def checkpoint_dir(self):
        return f"checkpoint/{self.checkpoints}"  

    @classmethod
    def load(cls, path: str):
        with open(path, encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}

        unknown = set(values) - {field.name for field in fields(cls)}

        if unknown:
            raise ValueError(f"Parâmetros desconhecidos em {path}: {sorted(unknown)}")

        return cls(**values)
