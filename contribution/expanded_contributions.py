import numpy as np, torch

from torch import nn
from setup import setup_model, setup_dataset
from visualization.visualization import Visualization, POSITIVE, NEGATIVE


def predict(model: nn.Module, x: torch.Tensor, device: str):
    with torch.no_grad():
        return model(x.to(device)).argmax(1).cpu().numpy()


def mode_maps(model: nn.Module, images: torch.Tensor, atoms: np.ndarray, layer: int, device: str):
    images = images.to(device).clone().requires_grad_()
    h = model.network[: 2 * layer + 1](model.flatten(images))
    probs = torch.softmax(model.network[2 * layer + 1:](h), dim=-1)

    dy_dh, = torch.autograd.grad(probs.max(1).values.sum(), h, retain_graph=True)
    dh_dx = torch.stack([torch.autograd.grad(h[:, j].sum(), images, retain_graph=True)[0] for j in range(h.shape[1])], dim=1)

    neuron_maps = images.detach()[:, None] * dh_dx * dy_dh[:, :, None, None]
    atoms = torch.as_tensor(atoms, dtype=h.dtype, device=device)
    terms = [atom[None, :, None, None] * neuron_maps for atom in atoms]

    positive = torch.stack([term.clamp(min=0).sum(1) for term in terms], dim=1)
    negative = torch.stack([term.clamp(max=0).sum(1) for term in terms], dim=1)

    return positive.cpu().numpy(), negative.cpu().numpy()


def class_mode_maps(model: nn.Module, x: torch.Tensor, y: np.ndarray, preds: np.ndarray, atoms: np.ndarray, layer: int, device: str, n_images: int = 100):
    positives, negatives = [], []

    for digit in range(10):
        images = x[torch.from_numpy((y == digit) & (preds == digit))][:n_images]
        positive, negative = mode_maps(model, images, atoms, layer, device)

        positives.append(positive.mean(0))
        negatives.append(negative.mean(0))

    return np.stack(positives), np.stack(negatives)


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = setup_model(model_type=model_type, device=device)

    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))

    _, test_dataloader = setup_dataset(dataset_type=dataset_type)

    test_dataset_x, test_dataset_y = test_dataloader.dataset.data.float() / 255, test_dataloader.dataset.targets.numpy()

    preds = predict(model, test_dataset_x, device)

    visualizer = Visualization(dataset=dataset_type, network=model_type)

    for layer in range(2):
        atoms = np.load(f"{model_path}/dictionary_layer_{layer}.npy")

        positive, negative = class_mode_maps(model, test_dataset_x, test_dataset_y, preds, atoms, layer, device)

        visualizer.plot_class_mode_maps(positive, layer, "positivas", POSITIVE)
        visualizer.plot_class_mode_maps(-negative, layer, "negativas", NEGATIVE)
        visualizer.plot_class_mode_maps_joint(positive, -negative, layer)


if __name__ == "__main__":
    main()
