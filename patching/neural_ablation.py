import numpy as np, torch

from contextlib import contextmanager
from datetime import datetime
from torch import nn
from zoneinfo import ZoneInfo
from setup import Setup, dataset_class_names
from contribution.utils import predict, class_modes


def top_neurons(dictionary: np.ndarray, corr: np.ndarray, label: int, fraction: float = 0.02, n_modes: int = 10):
    # neurônios a remover: em cada um dos n_modes modos mais correlacionados com a classe (os mesmos dos mapas), os `fraction` de maior |peso| no dicionário; união entre os modos
    modes = class_modes(corr, [label], n_modes)[0]
    n_top = max(1, round(fraction * dictionary.shape[1]))

    top = np.argsort(-np.abs(dictionary[modes]), axis=1, kind="stable")[:, :n_top]

    print(f"modes_correlation: {modes}", f"Label: {label} | neurônios removidos: {len(np.unique(top))} de {dictionary.shape[1]} \n")

    return np.unique(top)


def channel_mask(output: torch.Tensor, keep: np.ndarray):
    mask = torch.zeros(output.shape[1], device=output.device)
    mask[keep] = 1

    return mask.view(1, -1, *[1] * (output.dim() - 2))


@contextmanager
def ablate(layers: list, keeps: list):
    handles = []

    for layer, keep in zip(layers, keeps):
        handles.append(layer.register_forward_hook(lambda module, inputs, output, keep=keep: output * channel_mask(output, keep)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def accuracies(preds: np.ndarray, y: np.ndarray):
    return {"acc_all": (preds == y).mean(), "acc_per_class": np.array([(preds[y == c] == c).mean() for c in np.unique(y)])}


def label_metrics(preds: np.ndarray, y: np.ndarray, label: int):
    is_label = y == label

    return {
        "acc_label": (preds[is_label] == label).mean(),
        "acc_others": (preds[~is_label] == y[~is_label]).mean(),
        "frac_pred_label": (preds == label).mean(),
        **accuracies(preds, y),
    }


def evaluate(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, keeps: list, device: str, classes: list = None):
    with ablate(layers, keeps):
        return label_metrics(predict(model, x, device, classes), y, label)


def ablated_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, removed: list, n_neurons: list, device: str, classes: list = None):
    # removed[i] = neurônios zerados na camada i; os demais (keep) passam intactos
    keeps = [np.setdiff1d(np.arange(n), r) for r, n in zip(removed, n_neurons)]

    return evaluate(model, x, y, label, layers, keeps, device, classes)


def random_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, removed: list, n_neurons: list, device: str, n_random: int = 5, classes: list = None):
    # controle: em cada camada remove a mesma quantidade de neurônios que o SAE removeu, mas sorteados; média de n_random sorteios
    rng = np.random.default_rng(0)

    runs = [ablated_metrics(model, x, y, label, layers, [rng.choice(n, len(r), replace=False) for r, n in zip(removed, n_neurons)], n_neurons, device, classes) for _ in range(n_random)]

    return {metric: np.mean([run[metric] for run in runs], axis=0) for metric in runs[0]}


def compare_ablation(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, removed: np.ndarray, n_neurons: int, device: str, n_random: int = 5, classes: list = None):
    # uma camada: remoção dos neurônios do SAE e remoção aleatória (mesma quantidade); cada um é {métrica: valor}, e acc_per_class tem um valor por classe do teste
    sae = ablated_metrics(model, x, y, label, [layer], [removed], [n_neurons], device, classes)
    random = random_metrics(model, x, y, label, [layer], [removed], [n_neurons], device, n_random, classes)

    return {"sae": sae, "random": random}


def shapes(results: dict):
    return {name: value.shape for name, value in results.items()}


def accuracy_log(prefix: str, result: dict, step: int):
    return {f"{prefix}/acc_total": result["acc_all"][step], **{f"{prefix}/acc_label_{c}": acc for c, acc in enumerate(result["acc_per_class"][step])}}



def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", classes=[], layer_names=None, ablated_layers=None, fraction=0.02, n_modes=10, batch_size=64, setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, layer_names=layer_names, classes=classes)

    model, layers, device = setup.load_checkpoint(model_path), setup.layers, setup.device

    test_dataset_x, test_dataset_y = setup.test_data()  # todas as classes: y e preds são o próprio id do dataset

    test_dataset_x = test_dataset_x.to(device)

    class_names = dataset_class_names(setup.dataset_type)
    setup.visualizer.class_names = class_names

    dictionaries = [np.load(f"{model_path}/dictionary_layer_{layer}.npy") for layer in range(len(layers))]
    corrs = [np.load(f"{model_path}/corr_layer_{layer}.npy") for layer in range(len(layers))]

    ablated_layers = ablated_layers or [layer for layer, dictionary in enumerate(dictionaries) if len(dictionary) > 0]

    if not ablated_layers:
        raise ValueError("nenhuma camada tem modos vivos: refaça a interpretação com outros hiperparâmetros do SAE")

    labels = np.unique(test_dataset_y) if setup.classes is None else setup.classes

    for label in labels:  # id do dataset = linha da corr = posição em y
        print(f"Classe: {class_names[label]}")

        intact = evaluate(model, test_dataset_x, test_dataset_y, label, [], [], device)  # não depende da camada: calculado uma vez por classe

        for layer in ablated_layers:  # uma camada por vez: uma figura por camada e classe
            n_neurons = dictionaries[layer].shape[1]
            removed = top_neurons(dictionaries[layer], corrs[layer], label, fraction, n_modes)

            results = {"intact": intact, **compare_ablation(model, test_dataset_x, test_dataset_y, label, layers[layer], removed, n_neurons, device)}

            setup.visualizer.plot_ablation(results, label, layer, len(removed) / n_neurons)  # fração realmente removida: a união dos modos pode passar de `fraction`


if __name__ == "__main__":
    main()
