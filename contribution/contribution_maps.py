import numpy as np, torch

from torch import nn
from setup import Setup
from contribution.utils import spatial_sum, class_modes


def mode_maps(model: nn.Module, images: torch.Tensor, atoms: np.ndarray, layer: nn.Module, device: str):
    images = images.to(device).clone().requires_grad_()

    captured = []
    handle = layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    probs = torch.softmax(model(images), dim=-1)
    handle.remove()

    h = captured[0]
    dy_dh, = torch.autograd.grad(probs.max(1).values.sum(), h, retain_graph=True)

    atoms = torch.as_tensor(atoms, dtype=h.dtype, device=device)
    per_channel = spatial_sum(h * dy_dh)

    positive, negative = [], []

    for atom in atoms:
        grad, = torch.autograd.grad((per_channel * atom).sum(), images, retain_graph=True)
        atom_map = images.detach() * grad

        positive.append(atom_map.clamp(min=0).mean((0, 1)))
        negative.append(atom_map.clamp(max=0).mean((0, 1)))

    return torch.stack(positive).cpu().numpy(), torch.stack(negative).cpu().numpy()


def correct_images(x: torch.Tensor, y: np.ndarray, preds: np.ndarray, digit: int, n_images: int):
    return x[torch.from_numpy((y == digit) & (preds == digit))][:n_images]


def class_mode_maps(model: nn.Module, x: torch.Tensor, y: np.ndarray, atoms: np.ndarray, modes: np.ndarray, layer: nn.Module, device: str, n_images: int = 100):
    positives, negatives = [], []

    for digit, class_modes in zip(np.unique(y), modes):
        images = x[torch.from_numpy(y == digit)][:n_images]
        positive, negative = mode_maps(model, images, atoms[class_modes], layer, device)

        positives.append(positive)
        negatives.append(negative)

    return np.stack(positives), np.stack(negatives)


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", layer_names=None, classes=None, setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, layer_names=layer_names, classes=classes)

    model = setup.load_checkpoint(model_path)

    test_dataset_x, test_dataset_y = setup.test_data(only_labels=True)

    setup.visualizer.class_names = setup.create_class_names()  # y são posições entre as classes escolhidas

    rgb_images = test_dataset_x.shape[1] == 3
    n_images = 1 if rgb_images else 100
    examples = setup.display_images(test_dataset_x[np.unique(test_dataset_y, return_index=True)[1]]) if rgb_images else None  # 1ª imagem de cada classe = a que entra nos mapas

    labels = setup.classes if setup.classes is not None else np.unique(test_dataset_y)  # ids do dataset = linhas da corr (calculada em todas as classes)

    for index, layer in enumerate(setup.layers):
        atoms = np.load(f"{model_path}/dictionary_layer_{index}.npy")

        if len(atoms) == 0:
            continue

        modes = class_modes(np.load(f"{model_path}/corr_layer_{index}.npy"), labels, setup.visualizer.MAX_MODES)

        positive, negative = class_mode_maps(model, test_dataset_x, test_dataset_y, atoms, modes, layer, setup.device, n_images)

        setup.visualizer.plot_class_mode_maps(positive, index, "positivas", setup.visualizer.POSITIVE, modes, examples)


if __name__ == "__main__":
    main()
