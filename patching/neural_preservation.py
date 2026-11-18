import numpy as np, torch

from torch import nn
from setup import Setup, dataset_class_names
from patching.neural_ablation import top_neurons, evaluate


def preserved_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, kept: np.ndarray, device: str, classes: list = None):
    # lógica invertida em relação à ablação: `kept` (os neurônios dos modos da classe) é o que passa; todo o resto da camada é zerado
    return evaluate(model, x, y, label, [layer], [kept], device, classes)


def random_preserved_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, kept: np.ndarray, n_neurons: int, device: str, n_random: int = 5, classes: list = None):
    # controle: mantém a mesma quantidade de neurônios que o SAE manteve, mas sorteados; média de n_random sorteios
    rng = np.random.default_rng(0)

    runs = [evaluate(model, x, y, label, [layer], [rng.choice(n_neurons, len(kept), replace=False)], device, classes) for _ in range(n_random)]

    return {metric: np.mean([run[metric] for run in runs], axis=0) for metric in runs[0]}


def compare_preservation(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, kept: np.ndarray, n_neurons: int, device: str, n_random: int = 5, classes: list = None):
    sae = preserved_metrics(model, x, y, label, layer, kept, device, classes)
    random = random_preserved_metrics(model, x, y, label, layer, kept, n_neurons, device, n_random, classes)

    return {"sae": sae, "random": random}


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
            kept = top_neurons(dictionaries[layer], corrs[layer], label, fraction, n_modes)

            results = {"intact": intact, **compare_preservation(model, test_dataset_x, test_dataset_y, label, layers[layer], kept, n_neurons, device)}

            setup.visualizer.plot_preservation(results, label, layer, len(kept) / n_neurons)


if __name__ == "__main__":
    main()
