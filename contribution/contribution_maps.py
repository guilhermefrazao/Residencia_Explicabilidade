import numpy as np, torch

from torch import nn
from setup import Setup, MAX_TEST_DATA, dataset_class_names, dataset_observation_names
from contribution.utils import spatial_sum, class_modes, display


def mode_maps(model: nn.Module, images: torch.Tensor, atoms: np.ndarray, layer: nn.Module, device: str, softmax: bool = True, scalar: bool = False):
    images = images.to(device).clone().requires_grad_()

    captured = []
    handle = layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    outputs = model(images)
    target = torch.softmax(outputs, dim=-1) if softmax else outputs  # softmax=False: a saída já é o alvo (Q-values no RL)
    handle.remove()

    h = captured[0]
    dy_dh, = torch.autograd.grad((target.sum() if scalar else target.max(1).values.sum()), h, retain_graph=True)

    atoms = torch.as_tensor(atoms, dtype=h.dtype, device=device)
    per_channel = spatial_sum(h * dy_dh)

    positive, negative = [], []

    for atom in atoms:
        grad, = torch.autograd.grad((per_channel * atom).sum(), images, retain_graph=True)
        atom_map = images.detach() * grad

        if atom_map.ndim > 2:
            atom_map = atom_map.sum(1)  # imagem: soma dos canais ANTES do clamp (sinais opostos se cancelam em vez de acumular ruído positivo) -> [B, H, W]

        positive.append(atom_map.clamp(min=0).mean(0))  # média sobre as imagens/observações: [H, W] ou [variáveis]
        negative.append(atom_map.clamp(max=0).mean(0))

    return torch.stack(positive).cpu().numpy(), torch.stack(negative).cpu().numpy()


def class_mode_maps(model: nn.Module, x: torch.Tensor, y: np.ndarray, atoms: np.ndarray, modes: np.ndarray, layer: nn.Module, device: str, softmax: bool = True, n_images: int = None, scalar: bool = False):
    # n_images: quantas imagens de cada classe entram no mapa (a 1ª, a 2ª...); None = todas, até MAX_TEST_DATA
    positives, negatives = [], []

    for label, class_modes in zip(np.unique(y), modes):
        images = x[torch.from_numpy(y == label)][:n_images or MAX_TEST_DATA]
        positive, negative = mode_maps(model, images, atoms[class_modes], layer, device, softmax, scalar)

        positives.append(positive)
        negatives.append(negative)

    return np.stack(positives), np.stack(negatives)


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", layer_names=None, classes=None, setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, layer_names=layer_names, classes=classes)

    model = setup.load_checkpoint(model_path)

    if not setup.is_reinforcement:
        test_dataset_x, test_dataset_y = setup.test_data(only_labels=True)
    else:
        test_dataset_x, test_dataset_y = setup.test_observations(model_path)

    softmax = setup.model_type != "DQN" and not setup.is_tabular  # tabular: alvo escalar, sem softmax; Q-values do DQN não são logits: sem softmax (igual ao IG em contributions.py)

    rgb_images = setup.is_image and test_dataset_x.shape[1] == 3
    examples = display(test_dataset_x[np.unique(test_dataset_y, return_index=True)[1]]) if rgb_images else None  # 1ª imagem de cada classe = a do mapa da máscara (n_images=1)

    for index, layer in enumerate(setup.layers):
        atoms = np.load(f"{model_path}/dictionary_layer_{index}.npy")
        correlations = np.load(f"{model_path}/corr_layer_{index}.npy")

        if len(atoms) == 0:
            continue

        modes = class_modes(correlations, setup.labels, setup.visualizer.MAX_MODES)

        setup.visualizer.class_names = dataset_class_names(setup.dataset_type)  # nomes indexados por id; as linhas são posições: labels[i] é o id da linha i
        setup.visualizer.observation_names = dataset_observation_names(setup.dataset_type)

        if setup.is_image:
            single, _ = class_mode_maps(model, test_dataset_x, test_dataset_y, atoms, modes, layer, setup.device, softmax, n_images=1, scalar=setup.is_tabular)  # a imagem mostrada na coluna "original"
            mean, _ = class_mode_maps(model, test_dataset_x, test_dataset_y, atoms, modes, layer, setup.device, softmax, scalar=setup.is_tabular)  # média da classe

            setup.visualizer.class_names = dataset_class_names(setup.dataset_type)  # nomes indexados por id; as linhas são posições: labels[i] é o id da linha i
            setup.visualizer.plot_class_mode_maps_images(single, mean, index, modes, setup.labels, examples)
        else:
            positive, _ = class_mode_maps(model, test_dataset_x, test_dataset_y, atoms, modes, layer, setup.device, softmax, scalar=setup.is_tabular)

            setup.visualizer.plot_class_mode_maps_vector(positive, index, modes, setup.labels)


if __name__ == "__main__":
    main()
