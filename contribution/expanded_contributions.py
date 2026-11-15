import numpy as np, torch

from torch import nn
from setup import setup_model, setup_layers, setup_dataset, setup_test_data
from visualization.visualization import Visualization, POSITIVE, NEGATIVE, select_modes


def predict(model: nn.Module, x: torch.Tensor, device: str):
    with torch.no_grad():
        return model(x.to(device)).argmax(1).cpu().numpy()


def mode_maps(model: nn.Module, images: torch.Tensor, atoms: np.ndarray, layer: nn.Module, device: str):
    images = images.to(device).clone().requires_grad_()

    captured = []
    handle = layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    probs = torch.softmax(model(images), dim=-1)
    handle.remove()

    h = captured[0]  # saída da camada: [n, C] (Linear) ou [n, C, H, W] (Conv); cada unidade = um (canal, posição)
    dy_dh, = torch.autograd.grad(probs.max(1).values.sum(), h, retain_graph=True)
    dy_dh, units = dy_dh.flatten(1), h.flatten(1)

    atoms = torch.as_tensor(atoms, dtype=h.dtype, device=device)
    channels = torch.arange(units.shape[1], device=device) // (units.shape[1] // h.shape[1])

    positive = torch.zeros(len(atoms), *images.shape, device=device)
    negative = torch.zeros(len(atoms), *images.shape, device=device)

    for unit in range(units.shape[1]):
        dh_dx, = torch.autograd.grad(units[:, unit].sum(), images, retain_graph=True)

        unit_map = images.detach() * dh_dx * dy_dh[:, unit, None, None, None]
        terms = atoms[:, channels[unit]].view(-1, 1, 1, 1, 1) * unit_map[None]

        positive += terms.clamp(min=0)
        negative += terms.clamp(max=0)

    return positive.transpose(0, 1).mean(2).cpu().numpy(), negative.transpose(0, 1).mean(2).cpu().numpy()


def class_mode_maps(model: nn.Module, x: torch.Tensor, y: np.ndarray, preds: np.ndarray, atoms: np.ndarray, layer: nn.Module, device: str, n_images: int = 100):
    positives, negatives = [], []

    for digit in range(10):
        images = x[torch.from_numpy((y == digit) & (preds == digit))][:n_images]
        positive, negative = mode_maps(model, images, atoms, layer, device)

        positives.append(positive.mean(0))
        negatives.append(negative.mean(0))

    return np.stack(positives), np.stack(negatives)


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", layer_names=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = setup_model(model_type=model_type, device=device).eval()

    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))

    layers, _ = setup_layers(model=model, model_type=model_type, layer_names=layer_names)

    _, test_dataloader = setup_dataset(dataset_type=dataset_type)

    test_dataset_x, test_dataset_y = setup_test_data(test_dataloader)

    preds = predict(model, test_dataset_x, device)

    visualizer = Visualization(dataset=dataset_type, network=model_type)

    for index, layer in enumerate(layers):
        atoms = np.load(f"{model_path}/dictionary_layer_{index}.npy")

        if len(atoms) == 0:
            continue

        positive, negative = class_mode_maps(model, test_dataset_x, test_dataset_y, preds, atoms, layer, device)

        modes = select_modes(positive, negative)

        if len(modes) < len(atoms):
            print(f"Camada {index + 1}: {len(atoms)} modos, mostrando os {len(modes)} de maior contribuição")

        visualizer.plot_class_mode_maps(positive, index, "positivas", POSITIVE, modes)
        visualizer.plot_class_mode_maps(-negative, index, "negativas", NEGATIVE, modes)
        visualizer.plot_class_mode_maps_joint(positive, -negative, index, modes)


if __name__ == "__main__":
    main()
