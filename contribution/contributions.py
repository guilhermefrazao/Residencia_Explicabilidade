import numpy as np, torch, bscope

from bscope import NNSTSAE, load_sae
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm import tqdm
from torch import nn
from setup import Setup, dataset_class_names, dataset_observation_names
from contribution.utils import spatial_sum, predict, ig_baseline, display, ScalarOutput
from visualization.visualization import Visualization


def ig_contributions(model: nn.Module, layers: list, reduction: list, test_dataset_x: torch.Tensor, batch_size: int = 64, baseline: bool = False, regression: bool = False):
    # regression=True: saída escalar (preço); o alvo é o próprio valor previsto, com a observação média como referência
    scope = bscope.Scope(ScalarOutput(model) if regression else model, layers, to_numpy=True)
    scope.use_int_grad(steps=10)

    if regression:
        scope.wrt_sum(softmax=False)
    elif baseline:
        #scope.wrt_contrastive_top2(softmax=False)
        scope.wrt_topk(k=1, softmax=False)
        #scope.wrt_topk(k=5, softmax=False)
    else:
        scope.wrt_topk(k=1, softmax=True)

    scope.log_start(reduction=reduction)

    with ig_baseline(test_dataset_x.mean(0) if baseline or regression else None):
        for xb in tqdm(test_dataset_x.split(batch_size)):
            scope(xb)

    scope.log_stop()
    scope.inspector.remove_hooks()

    scope.log_contributions = [spatial_sum(c) for c in scope.log_contributions] 

    return scope


def sae_contributions(setup: Setup, scope: bscope.Scope, model_path: str):
    datas = []

    device = setup.device

    for l in range(len(setup.layers)):
        c = scope.log_contributions[l]

        data = c[:, c.shape[1] // 2:]
        data = data / np.std(data)
        contributions_dim = data.shape[1]
        positive_contributions = torch.from_numpy(data).float().to(device)

        theashold = 0.8
        lr = 5e-5
        batch_size = 128
        n_epochs = 300
        global_step = 0
        l1_regularization = 5e-5

        sae = NNSTSAE(data_dim=contributions_dim, num_atoms=3 * contributions_dim, threshold=theashold, mlp_hidden_dim=contributions_dim).to(device)

        optimizer = setup.create_optimizer(sae, "Adam", lr)

        date = datetime.now(ZoneInfo("America/Sao_Paulo"))

        wandb_run = setup.init_wandb(f"Sparse_Autoencoders_{l}_{date.strftime('%d/%m/%Y - %H:%M')}", config={"sae/train/n_epochs": n_epochs, "sae/train/input_dim": contributions_dim, "sae/train/theashold": theashold, "sae/train/learning_rate": lr, "sae/train/batch_size": batch_size})

        progress_bar = tqdm(range(n_epochs), desc=f"{type(sae).__name__} - camada {l + 1}", position=0, leave=True)

        for ep in progress_bar:
            for idx in torch.randperm(len(positive_contributions), device=device).split(batch_size):
                x = positive_contributions[idx]
                codes, z, reconstructed = sae(x)
                loss = nn.functional.mse_loss(reconstructed, x) + l1_regularization * sae.dictionary.get_dictionary().abs().mean()
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                global_step += 1

                wandb_run.log({
                    "sae/losses/loss": loss.item(),
                    "sae/train/global_step": global_step,
                })

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")


        torch.save(sae, f"{model_path}/sae_layer_{l}.pt")
        datas.append(data)

        wandb_run.finish()

    return datas


def correlation_results(one_hot_y: np.ndarray, model_path: str, datas: list, device: str):
    loadings_all, dicts, corrs = [], [], []

    for l, data in enumerate(datas):
        _, loadings, dictionary, _, _, _ = load_sae(f"{model_path}/sae_layer_{l}.pt", data, device=device)

        loadings_all.append(loadings); dicts.append(dictionary)

        has_modes = loadings.shape[1] > 0
        corr = np.nan_to_num(bscope.mtx_corr(one_hot_y, loadings)) if has_modes else np.zeros((one_hot_y.shape[1], 0))

        np.save(f"{model_path}/corr_layer_{l}.npy", corr)
        np.save(f"{model_path}/loadings_layer_{l}.npy", loadings)
        np.save(f"{model_path}/dictionary_layer_{l}.npy", dictionary)

        if not has_modes:
            print(f"Camada {l + 1}: nenhum modo vivo, o SAE colapsou (treine de novo, use threshold menor ou rede mais larga)")
            corrs.append(None)
            continue

        corrs.append(corr)

        for d in range(corr.shape[0]):
            k = corr[d].argmax()
            print(f"Posição Dataset de Teste {d}: modo {k} (corr {corr[d, k]:.2f}) | neuronios que mais pesam: {np.argsort(-np.abs(dictionary[k]))[:7]}")


    return loadings_all, dicts, corrs


def visualization(visualizer: Visualization, setup: Setup, model: nn.Module, scope: bscope.Scope, test_dataset_x: torch.Tensor, test_dataset_y: np.ndarray, datas: list, dicts: list, corrs: list, device: str):
    preds = predict(model, test_dataset_x, device)  # o teste cobre todas as classes: y e preds são o próprio id do dataset

    sizes = [data.shape[1] for data in datas] + [test_dataset_y.max() + 1]

    if setup.is_image:
        test_dataset_x = display(test_dataset_x)

    visualizer.visualize_all(sizes=sizes, X=test_dataset_x, y=test_dataset_y, preds=preds, contributions=scope.log_contributions, dictionaries=dicts, corrs=corrs, labels=setup.labels, n_images=2)


def main(model_path="checkpoint/test", dataset_type="Mnist", model_type="MLP", layer_names=None, reduction=None, classes=None, batch_size=64, setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset_type, model_type=model_type, layer_names=layer_names, reduction=reduction, classes=classes)

    model = setup.load_checkpoint(model_path)

    if not setup.is_reinforcement:
        test_dataset_x, test_dataset_y = setup.test_data()
        one_hot_y = np.eye(test_dataset_y.max() + 1)[test_dataset_y]
        baseline = None
    else:
        test_dataset_x, test_dataset_y = setup.test_observations(model_path)
        one_hot_y = np.eye(setup.model_kwargs["n_actions"])[test_dataset_y]
        baseline = True

    scope = ig_contributions(model=model, layers=setup.layers, reduction=setup.reduction, test_dataset_x=test_dataset_x, batch_size=batch_size, baseline=baseline, regression=setup.is_tabular)

    datas = sae_contributions(setup=setup, scope=scope, model_path=model_path)

    loadings_all, dicts, corrs = correlation_results(one_hot_y=one_hot_y, model_path=model_path, datas=datas, device=setup.device)

    setup.visualizer.class_names = dataset_class_names(setup.dataset_type)  # os rótulos aqui são ids: nomes indexados por id

    if setup.is_image:
        visualization(visualizer=setup.visualizer, setup=setup, model=model, scope=scope, test_dataset_x=test_dataset_x, test_dataset_y=test_dataset_y, datas=datas, dicts=dicts, corrs=corrs, device=setup.device)
        
    else:
        setup.visualizer.observation_names = dataset_observation_names(setup.dataset_type)

        visualization(visualizer=setup.visualizer, setup=setup, model=model, scope=scope, test_dataset_x=test_dataset_x, test_dataset_y=test_dataset_y, datas=datas, dicts=dicts, corrs=corrs, device=setup.device)
        return 




if __name__ == "__main__":
    main()
