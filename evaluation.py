import argparse

from config.config import Configs
from setup import Setup
from contribution.contributions import main as interpret_contributions
from contribution.contribution_maps import main as contribution_maps
from patching.neural_ablation import main as ablation
from patching.neural_preservation import main as preservation
from contribution.model_analysis import main as model_analysis


def main(config: str):
    configs = Configs.load(config)
    checkpoints = configs.checkpoint_dir

    setup = Setup(dataset_type=configs.dataset, model_type=configs.model, loss_type=configs.loss_type, optimizer_type=configs.optimizer_type, learning_rate=configs.learning_rate, layer_names=configs.layer_names, reduction=configs.reduction, classes=configs.classes, experiment_name=configs.experiment_name)

    interpret_contributions(model_path=checkpoints, batch_size=configs.ig_batch_size, setup=setup, classes=configs.classes)

    contribution_maps(model_path=checkpoints, setup=setup)

    ablation(model_path=checkpoints, classes=configs.classes, setup=setup)

    preservation(model_path=checkpoints, classes=configs.classes, setup=setup)

    if setup.is_reinforcement:
        model_analysis(model_path=checkpoints, setup=setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="caminho do .yml do experimento (ex.: config/cartpole_dqn.yml)")

    main(**vars(parser.parse_args()))
