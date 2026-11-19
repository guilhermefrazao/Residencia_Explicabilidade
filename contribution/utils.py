import importlib
import numpy as np, torch

from contextlib import contextmanager
from torch import nn
from setup import IMAGENET_MEAN, IMAGENET_STD, random_labels, price_band, nearest_species  # random_labels vive em setup.py (o Setup a usa para sortear self.labels); re-exportada aqui

bscope_scope = importlib.import_module("bscope.scope")


@contextmanager
def ig_baseline(baseline: torch.Tensor = None):
    """Troca a baseline do IG do bscope (fixa em zeros) por `baseline`, com a forma de uma observação; None mantém zeros."""
    original = bscope_scope.interpolate_stim

    if baseline is not None:
        def interpolate_stim(stim, steps=10):
            base = baseline.to(device=stim.device, dtype=stim.dtype).expand_as(stim)
            interp_stim = torch.stack([base + (float(i) / steps) * (stim - base) for i in range(steps + 1)])
            interp_stim.requires_grad = True

            return interp_stim

        bscope_scope.interpolate_stim = interpolate_stim

    try:
        yield
    finally:
        bscope_scope.interpolate_stim = original

import numpy as np


class ScalarOutput(nn.Module):
    """Saída escalar [B] vista como [B, 1] (formato que os alvos do bscope esperam); saídas já 2-D passam sem mudança. As camadas são as do modelo original."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        return out.unsqueeze(-1) if out.dim() == 1 else out


def spatial_sum(x):
    return x.sum((2, 3)) if x.ndim == 4 else x


def predict(model: nn.Module, x: torch.Tensor, device: str, classes: list = None):
    with torch.no_grad():
        outputs = torch.cat([model(chunk.to(device)).cpu() for chunk in x.split(100)])

    if outputs.dim() == 1:  # regressão (saída [B]): a "classe prevista" é a faixa do preço previsto
        preds = price_band(outputs.numpy())
    elif outputs.shape == x.shape and outputs.dim() == 2:  # autoencoder tabular: a classe é a espécie do centróide mais próximo da reconstrução
        preds = nearest_species(outputs.numpy())
    else:
        preds = outputs.argmax(1).numpy()

    if classes is None:
        return preds

    # saída da rede = id da classe no dataset; vira a posição em `classes` ordenadas (y de Setup.test_data), -1 se a rede prevê uma classe fora delas
    classes = np.sort(classes)
    position = np.searchsorted(classes, preds).clip(max=len(classes) - 1)

    return np.where(classes[position] == preds, position, -1)


def class_modes(corr, labels, n_modes: int):
    return np.stack([np.argsort(-corr[label], kind="stable")[:n_modes] for label in labels])


@staticmethod
def display(x: torch.Tensor):
    if x.shape[1] == 1:
        return x.squeeze(1).numpy()

    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))

    return (x * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()