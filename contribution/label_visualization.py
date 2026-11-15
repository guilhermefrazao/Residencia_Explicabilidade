import argparse, numpy as np

from setup import setup_dataset
from visualization.visualization import Visualization


def select_label_images(x: np.ndarray, y: np.ndarray, loadings: np.ndarray, label: int, n_images: int):
    indices = np.where(y == label)[0][:n_images]
    modes = loadings[indices].argmax(1)
    order = np.argsort(modes, kind="stable")

    return x[indices][order], modes[order]


def main(label: int, n_images: int = 100, layer: int = 0, model_path: str = "checkpoint/test", dataset_type: str = "Mnist", model_type: str = "MLP"):
    _, test_dataloader = setup_dataset(dataset_type=dataset_type)

    x, y = test_dataloader.dataset.data.numpy(), test_dataloader.dataset.targets.numpy()

    loadings = np.load(f"{model_path}/loadings_layer_{layer}.npy")

    images, modes = select_label_images(x, y, loadings, label, n_images)

    Visualization(dataset=dataset_type, network=model_type).plot_label_images(images, modes, label, layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int)
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--layer", type=int, default=0)

    main(**vars(parser.parse_args()))
