import argparse

from config.config import Configs
from setup import Setup
from supervized_learning import main as train_supervized
from unsupervized_learning import main as train_unsupervized
from reinforcement_learning import main as train_reinforcement
from contribution.contributions import main as interpret_contributions
from contribution.contribution_maps import main as contribution_maps
from patching.neural_ablation import main as ablation
from patching.neural_preservation import main as preservation


def main(config: str):
    configs = Configs.load(config)
    checkpoints = configs.checkpoint_dir

    setup = Setup(dataset_type=configs.dataset, model_type=configs.model, loss_type=configs.loss_type, optimizer_type=configs.optimizer_type, learning_rate=configs.learning_rate, layer_names=configs.layer_names, reduction=configs.reduction, classes=configs.classes, render=configs.render, experiment_name=configs.experiment_name)

    if configs.learning_type == "supervized":
        train_supervized(checkpoints=checkpoints, n_epochs=configs.n_epochs, setup=setup)

    if configs.learning_type == "unsupervized":
        train_unsupervized(checkpoints=checkpoints, n_epochs=configs.n_epochs, setup=setup)

    if configs.learning_type == "reinforcement":
        train_reinforcement(checkpoints=checkpoints, setup=setup, configs=configs)

    interpret_contributions(model_path=checkpoints, batch_size=configs.ig_batch_size, setup=setup, classes=configs.classes)

    contribution_maps(model_path=checkpoints, setup=setup)

    ablation(model_path=checkpoints, classes=configs.classes, batch_size=configs.ig_batch_size, setup=setup)

    preservation(model_path=checkpoints, classes=configs.classes, batch_size=configs.ig_batch_size, setup=setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="caminho do .yml do experimento (ex.: config/mnist_mlp.yml)")

    main(**vars(parser.parse_args()))
