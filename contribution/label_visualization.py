import argparse, numpy as np

from setup import Setup, dataset_class_names


def select_label_images(x: np.ndarray, y: np.ndarray, loadings: np.ndarray, label: int, n_images: int):
    indices = np.where(y == label)[0][:n_images]
    modes = loadings[indices].argmax(1)
    order = np.argsort(modes, kind="stable")

    return x[indices][order], modes[order]


def main(classes: list, layers: list, n_images: int = 100, model_path: str = "checkpoint/test", dataset_type: str = "Mnist", model_type: str = "MLP", setup: Setup = None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, classes=classes)

    x, y = setup.test_data()  # as mesmas imagens dos loadings salvos (todas as classes)

    setup.visualizer.class_names = dataset_class_names(setup.dataset_type)

    x = setup.display_images(x)

    for layer in layers:
        loadings = np.load(f"{model_path}/loadings_layer_{layer}.npy")

        if loadings.shape[1] == 0:
            continue

        for label in (np.unique(y) if setup.classes is None else setup.classes):
            images, modes = select_label_images(x, y, loadings, label, n_images)

            setup.visualizer.plot_label_images(images, modes, label, layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--n_images", type=int, default=100)

    main(**vars(parser.parse_args()))
