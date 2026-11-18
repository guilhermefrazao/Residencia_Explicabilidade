import numpy as np, torch

from torch import nn

import numpy as np


def spatial_sum(x):
    return x.sum((2, 3)) if x.ndim == 4 else x


def predict(model: nn.Module, x: torch.Tensor, device: str, classes: list = None):
    with torch.no_grad():
        preds = torch.cat([model(chunk.to(device)).argmax(1).cpu() for chunk in x.split(100)]).numpy()

    if classes is None:
        return preds

    # saída da rede = id da classe no dataset; vira a posição em `classes` ordenadas (y de Setup.test_data), -1 se a rede prevê uma classe fora delas
    classes = np.sort(classes)
    position = np.searchsorted(classes, preds).clip(max=len(classes) - 1)

    return np.where(classes[position] == preds, position, -1)

def class_modes(corr, labels, n_modes: int):
    return np.stack([np.argsort(-corr[label], kind="stable")[:n_modes] for label in labels])
