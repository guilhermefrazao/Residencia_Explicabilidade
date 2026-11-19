import numpy as np, torch

from torch import nn
from setup import Setup, dataset_class_names
from patching.neural_ablation import top_neurons, evaluate, layer_means


def preserved_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, kept: np.ndarray, device: str, classes: list = None, means: list = None):
    # lógica invertida em relação à ablação: `kept` (os neurônios dos modos da classe) é o que passa; todo o resto da camada é zerado (ou vira a média, se `means`)
    return evaluate(model, x, y, label, [layer], [kept], device, classes, means)


def random_preserved_metrics(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, kept: np.ndarray, n_neurons: int, device: str, n_random: int = 5, classes: list = None, means: list = None):
    # controle: mantém a mesma quantidade de neurônios que o SAE manteve, mas sorteados; média de n_random sorteios
    rng = np.random.default_rng(0)

    runs = [evaluate(model, x, y, label, [layer], [rng.choice(n_neurons, len(kept), replace=False)], device, classes, means) for _ in range(n_random)]

    return {metric: np.mean([run[metric] for run in runs], axis=0) for metric in runs[0]}


def compare_preservation(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layer: nn.Module, kept: np.ndarray, n_neurons: int, device: str, n_random: int = 5, classes: list = None, means: list = None):
    sae = preserved_metrics(model, x, y, label, layer, kept, device, classes, means)
    random = random_preserved_metrics(model, x, y, label, layer, kept, n_neurons, device, n_random, classes, means)

    return {"sae": sae, "random": random}


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", classes=[], layer_names=None, fraction=0.02, setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, layer_names=layer_names, classes=classes)

    model, layers, device = setup.load_checkpoint(model_path), setup.layers, setup.device

    if not setup.is_reinforcement:
        test_dataset_x, test_dataset_y = setup.test_data()
    else:
        test_dataset_x, test_dataset_y = setup.test_observations(model_path)

    test_dataset_x = test_dataset_x.to(device)

    class_names = dataset_class_names(setup.dataset_type)
    setup.visualizer.class_names = class_names

    means = None if setup.is_image else layer_means(model, layers, test_dataset_x, device)  # RL: o que não é mantido vira a ativação média, não 0 (ver neural_ablation)

    dictionaries = [np.load(f"{model_path}/dictionary_layer_{layer}.npy") for layer in range(len(layers))]
    corrs = [np.load(f"{model_path}/corr_layer_{layer}.npy") for layer in range(len(layers))]

    for label in setup.labels:  # id do dataset = linha da corr = posição em y
        print(f"Classe: {class_names[label]}")

        intact = evaluate(model, test_dataset_x, test_dataset_y, label, [], [], device)  # não depende da camada: calculado uma vez por classe

        for layer in range(len(layers)):  # as camadas do setup (as mesmas da interpretação), uma por vez: uma figura por camada e classe
            if len(dictionaries[layer]) == 0:  # SAE sem modos vivos nessa camada
                continue

            n_neurons = dictionaries[layer].shape[1]
            kept = top_neurons(dictionaries[layer], corrs[layer], label, fraction, 2)

            results = {"intact": intact, **compare_preservation(model, test_dataset_x, test_dataset_y, label, layers[layer], kept, n_neurons, device, means=None if means is None else [means[layer]])}

            plot = setup.visualizer.plot_preservation_images if setup.is_image else setup.visualizer.plot_preservation_vector

            plot(results, label, layer, len(kept) / n_neurons)


if __name__ == "__main__":
    main()
