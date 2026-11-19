import numpy as np, torch

from contextlib import contextmanager
from datetime import datetime
from torch import nn
from zoneinfo import ZoneInfo
from setup import Setup, dataset_class_names
from contribution.utils import predict, class_modes


def top_neurons(dictionary: np.ndarray, corr: np.ndarray, label: int, fraction: float = 0.02, n_modes: int = 2):
    modes = class_modes(corr, [label], n_modes)[0]
    n_top = max(1, round(fraction * dictionary.shape[1]))

    mode_values_abs = -np.abs(dictionary[modes])

    top = np.argsort(mode_values_abs, axis=1, kind="stable")[:, :n_top]

    print(f"modes_correlation: {modes}", f"Label: {label} | neurônios removidos: {len(np.unique(top))} de {dictionary.shape[1]} \n")

    return np.unique(top)


def channel_mask(output: torch.Tensor, keep: np.ndarray):
    mask = torch.zeros(output.shape[1], device=output.device)
    mask[keep] = 1

    return mask.view(1, -1, *[1] * (output.dim() - 2))


def layer_means(model: nn.Module, layers: list, x: torch.Tensor, device: str):
    """Ativação média de cada neurônio (canal) de cada camada, no modelo intacto sobre `x`: [n_neurônios] por camada (pré-ativação, onde o hook da ablação atua)."""
    sums, counts = [0.0] * len(layers), [0] * len(layers)

    def collect(module, inputs, output, i):
        per_channel = output.detach().transpose(0, 1).flatten(1)  # [canais, amostras (x posições)]

        sums[i] = sums[i] + per_channel.sum(1)
        counts[i] += per_channel.shape[1]

    handles = [layer.register_forward_hook(lambda module, inputs, output, i=i: collect(module, inputs, output, i)) for i, layer in enumerate(layers)]

    try:
        with torch.no_grad():
            for chunk in x.split(100):
                model(chunk.to(device))
    finally:
        for handle in handles:
            handle.remove()

    return [total / count for total, count in zip(sums, counts)]


@contextmanager
def ablate(layers: list, keeps: list, means: list = None):
    # means[i] (opcional) = valor que os neurônios removidos da camada i passam a ter, em vez de 0 (ablação pela média)
    handles = []

    def hook(module, inputs, output, keep, mean):
        mask = channel_mask(output, keep)

        return output * mask if mean is None else output * mask + (1 - mask) * mean.view(1, -1, *[1] * (output.dim() - 2))

    for i, (layer, keep) in enumerate(zip(layers, keeps)):
        mean = None if means is None else means[i]
        handles.append(layer.register_forward_hook(lambda module, inputs, output, keep=keep, mean=mean: hook(module, inputs, output, keep, mean)))

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


def evaluate(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, keeps: list, device: str, classes: list = None, means: list = None):
    with ablate(layers, keeps, means):
        return label_metrics(predict(model, x, device, classes), y, label)


def ablated_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, removed: list, n_neurons: list, device: str, classes: list = None, means: list = None):
    # removed[i] = neurônios zerados (ou trocados pela média, se `means`) na camada i; os demais (keep) passam intactos
    keeps = [np.setdiff1d(np.arange(n), r) for r, n in zip(removed, n_neurons)]

    return evaluate(model, x, y, label, layers, keeps, device, classes, means)


def random_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, removed: list, n_neurons: list, device: str, n_random: int = 5, classes: list = None, means: list = None):
    # controle: em cada camada remove a mesma quantidade de neurônios que o SAE removeu, mas sorteados; média de n_random sorteios
    rng = np.random.default_rng(0)

    runs = [ablated_metrics(model, x, y, label, layers, [rng.choice(n, len(r), replace=False) for r, n in zip(removed, n_neurons)], n_neurons, device, classes, means) for _ in range(n_random)]

    return {metric: np.mean([run[metric] for run in runs], axis=0) for metric in runs[0]}


def compare_ablation(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, removed: np.ndarray, n_neurons: int, device: str, n_random: int = 5, classes: list = None, means: list = None):
    # uma camada: remoção dos neurônios do SAE e remoção aleatória (mesma quantidade); cada um é {métrica: valor}, e acc_per_class tem um valor por classe do teste
    sae = ablated_metrics(model, x, y, label, [layer], [removed], [n_neurons], device, classes, means)
    random = random_metrics(model, x, y, label, [layer], [removed], [n_neurons], device, n_random, classes, means)

    return {"sae": sae, "random": random}


def shapes(results: dict):
    return {name: value.shape for name, value in results.items()}


def accuracy_log(prefix: str, result: dict, step: int):
    return {f"{prefix}/acc_total": result["acc_all"][step], **{f"{prefix}/acc_label_{c}": acc for c, acc in enumerate(result["acc_per_class"][step])}}



def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", classes=[], layer_names=None, fraction=0.02,setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, layer_names=layer_names, classes=classes)

    model, layers, device = setup.load_checkpoint(model_path), setup.layers, setup.device

    if not setup.is_reinforcement:
        test_dataset_x, test_dataset_y = setup.test_data()
    else:
        test_dataset_x, test_dataset_y = setup.test_observations(model_path)

    test_dataset_x = test_dataset_x.to(device)

    dictionaries = [np.load(f"{model_path}/dictionary_layer_{layer}.npy") for layer in range(len(layers))]
    corrs = [np.load(f"{model_path}/corr_layer_{layer}.npy") for layer in range(len(layers))]

    setup.visualizer.class_names = dataset_class_names(setup.dataset_type)
    means = None if setup.is_image else layer_means(model, layers, test_dataset_x, device)  # vetores: o removido vira a ativação média, não 0

    for label in setup.labels: 
        intact = evaluate(model, test_dataset_x, test_dataset_y, label, [], [], device)  

        for layer in range(len(layers)): 
            if len(dictionaries[layer]) == 0:  
                continue

            n_neurons = dictionaries[layer].shape[1]
            removed = top_neurons(dictionaries[layer], corrs[layer], label, fraction)

            results = {"intact": intact, **compare_ablation(model, test_dataset_x, test_dataset_y, label, layers[layer], removed, n_neurons, device, means=None if means is None else [means[layer]])}

            plot = setup.visualizer.plot_ablation_images if setup.is_image else setup.visualizer.plot_ablation_vector

            plot(results, label, layer, len(removed) / n_neurons) 


if __name__ == "__main__":
    main()
