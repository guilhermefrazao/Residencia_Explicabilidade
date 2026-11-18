import argparse

from setup import Setup
from supervized_learning import main as train_supervized
from unsupervized_learning import main as train_unsupervized
from reinforcement_learning import main as train_reinforcement
from contribution.contributions import main as interpret_contributions
from contribution.contribution_maps import main as contribution_maps
from contribution.label_visualization import main as visualize_label_images
from patching.neural_ablation import main as ablation
from patching.neural_preservation import main as preservation


def main(checkpoints="checkpoint/ImagenetTrain", dataset="Imagenet", model="ResNet50", loss_type="CrossEntropy", optimizer_type="Adam", learning_rate=0.001, n_epochs=100, layer_names=None, reduction=None, classes="violin", ig_batch_size=16):
    setup = Setup(dataset_type=dataset, model_type=model, loss_type=loss_type, optimizer_type=optimizer_type, learning_rate=learning_rate, layer_names=layer_names, reduction=reduction, classes=classes)

    interpret_contributions(model_path=checkpoints, batch_size=ig_batch_size, setup=setup, classes=classes)

    contribution_maps(model_path=checkpoints, setup=setup)

    ablation(model_path=checkpoints, classes=classes, ablated_layers=None, batch_size=ig_batch_size, setup=setup)

    preservation(model_path=checkpoints, classes=classes, ablated_layers=None, batch_size=ig_batch_size, setup=setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="checkpoint/ResNet50Pretrained")
    parser.add_argument("--dataset", default="Imagenet")
    parser.add_argument("--model", default="ResNet50")
    parser.add_argument("--loss_type", default="CrossEntropy")
    parser.add_argument("--optimizer_type", default="Adam")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--layer_names", nargs="+", default=None)
    parser.add_argument("--reduction", nargs="+", default=None)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--classes", type=str, nargs="+", default=["bow", "violin", "cello", "king penguin"])
    parser.add_argument("--ig_batch_size", type=int, default=16)

    main(**vars(parser.parse_args()))
