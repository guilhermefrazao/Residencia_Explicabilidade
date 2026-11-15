import numpy as np, torch

from contextlib import contextmanager
from datetime import datetime
from torch import nn
from zoneinfo import ZoneInfo
from setup import setup_model, setup_layers, setup_dataset, setup_test_data, setup_wandb
from contribution.contributions import compute_contributions
from contribution.expanded_contributions import predict
from visualization.visualization import Visualization


def rank_neurons(dictionary: np.ndarray, corr: np.ndarray, label: int):
    modes = np.where(corr.argmax(0) == label)[0]

    print(f"modes_correlation: {modes}", f"Label: {label} \n")

    if len(modes) == 0:
        return None

    return np.argsort(-np.abs(dictionary[modes]).sum(0), kind="stable")


def rank_neurons_ig(contribution: np.ndarray, y: np.ndarray, label: int):
    # contribution = scope.log_contributions[layer], [N, 2C] = [negativa | positiva]; baseline sem SAE: contribuição positiva média do IG nas imagens do dígito
    positive = contribution[:, contribution.shape[1] // 2:]

    return np.argsort(-positive[y == label].mean(0), kind="stable")


def channel_mask(output: torch.Tensor, keep: np.ndarray):
    mask = torch.zeros(output.shape[1], device=output.device)
    mask[keep] = 1

    return mask.view(1, -1, *[1] * (output.dim() - 2))


@contextmanager
def ablate(layers: list, keeps: list):
    handles = []

    for layer, keep in zip(layers, keeps):
        handles.append(layer.register_forward_hook(lambda module, inputs, output, keep=keep: output * channel_mask(output, keep)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def accuracies(preds: np.ndarray, y: np.ndarray):
    return {"acc_all": (preds == y).mean(), "acc_per_class": np.array([(preds[y == c] == c).mean() for c in np.unique(y)])}


def label_metrics(preds: np.ndarray, y: np.ndarray, label: int):
    is_label = y == label

    return {
        "acc_label": (preds[is_label] == label).mean(),
        "acc_others": (preds[~is_label] == y[~is_label]).mean(),
        "frac_pred_label": (preds == label).mean(),
        **accuracies(preds, y),
    }


def evaluate(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, keeps: list, device: str):
    with ablate(layers, keeps):
        return label_metrics(predict(model, x, device), y, label)


def n_kept(order: np.ndarray, fraction_removed: float):
    return len(order) - round(fraction_removed * len(order))


def sweep(model: nn.Module, x: torch.Tensor, y: np.ndarray, label: int, layers: list, orders: dict, fractions: np.ndarray, device: str, n_random: int = 5):
    # orders = {método: ranking de neurônios por camada}; o controle aleatório entra como "random"
    rng = np.random.default_rng(0)
    results = {name: [] for name in [*orders, "random"]}
    reference = next(iter(orders.values()))

    for fraction in fractions:
        for name, ranking in orders.items():
            results[name].append(evaluate(model, x, y, label, layers, [order[: n_kept(order, fraction)] for order in ranking], device))

        runs = [evaluate(model, x, y, label, layers, [rng.permutation(len(order))[: n_kept(order, fraction)] for order in reference], device) for _ in range(n_random)]
        results["random"].append({metric: np.mean([run[metric] for run in runs], axis=0) for metric in runs[0]})

    return {name: {metric: np.array([row[metric] for row in rows]) for metric in rows[0]} for name, rows in results.items()}


def shapes(results: dict):
    return {name: value.shape for name, value in results.items()}


def accuracy_log(prefix: str, result: dict, step: int):
    return {f"{prefix}/acc_total": result["acc_all"][step], **{f"{prefix}/acc_label_{c}": acc for c, acc in enumerate(result["acc_per_class"][step])}}


def log_ablation(run_name: str, config: dict, results: dict, fractions: np.ndarray):
    wandb_run = setup_wandb(run_name=run_name, config=config)

    for step, fraction in enumerate(fractions):
        wandb_run.log({
            "ablation/fraction_removed": 100 * fraction,
            **accuracy_log("ablation", results["sae"], step),
            **accuracy_log("ablation/ig", results["ig"], step),
        }, step=step)

    wandb_run.finish()


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", label=[1, 6, 7], layer_names=None, ablated_layers=None, n_steps=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = setup_model(model_type=model_type, device=device).eval()

    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))

    layers, reduction = setup_layers(model=model, model_type=model_type, layer_names=layer_names)

    _, test_dataloader = setup_dataset(dataset_type=dataset_type)

    test_dataset_x, test_dataset_y = setup_test_data(test_dataloader)

    test_dataset_x = test_dataset_x.to(device)

    dictionaries = [np.load(f"{model_path}/dictionary_layer_{layer}.npy") for layer in range(len(layers))]
    corrs = [np.load(f"{model_path}/corr_layer_{layer}.npy") for layer in range(len(layers))]

    ablated_layers = ablated_layers or [layer for layer, dictionary in enumerate(dictionaries) if len(dictionary) > 0]

    if not ablated_layers:
        raise ValueError("nenhuma camada tem modos vivos: refaça a interpretação com outros hiperparâmetros do SAE")

    scope, _ = compute_contributions(model=model, layers=layers, reduction=reduction, test_dataset_x=test_dataset_x, test_dataset_y=test_dataset_y)
    scope.inspector.remove_hooks()

    labels = [label] if isinstance(label, int) else list(label)

    fractions = np.linspace(0, 1, n_steps + 1)

    visualizer = Visualization(dataset=dataset_type, network=model_type)

    date = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y - %H:%M")

    for digit in labels:
        print(f"Digit: {digit}")

        orders = {layer: rank_neurons(dictionaries[layer], corrs[layer], digit) for layer in ablated_layers}
        orders = {layer: order for layer, order in orders.items() if order is not None}

        if not orders:
            print(f"Dígito {digit}: nenhuma camada tem modos desse dígito, ablação pulada \n")
            continue

        print(f"Orders (Per layer): {list(orders.values())}", f"Dim: {[order.shape for order in orders.values()]} \n")

        ig_orders = [rank_neurons_ig(scope.log_contributions[layer], test_dataset_y, digit) for layer in orders]
        print(f"Orders IG (Per layer): {ig_orders}", f"Dim: {[order.shape for order in ig_orders]} \n")

        results = sweep(model, test_dataset_x, test_dataset_y, digit, [layers[layer] for layer in orders], {"sae": list(orders.values()), "ig": ig_orders}, fractions, device)

        visualizer.plot_ablation(results, digit, fractions)

        log_ablation(f"Ablation_{model_type}_label{digit}_{date}", {"ablation/label": digit, "ablation/ablated_layers": list(orders), "ablation/replacement": "zero"}, results, fractions)


if __name__ == "__main__":
    main()
