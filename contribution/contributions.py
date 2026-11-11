import numpy as np, torch, bscope

from bscope import NNSTSAE, load_sae
from datetime import datetime
from zoneinfo import ZoneInfo
from tqdm import tqdm
from torch import nn
from setup import setup_model, setup_dataset, setup_optimizer, setup_wandb
from visualization.visualization import visualize_all, visualize_neural_contributions, plot_pixel_maps


def compute_contributions(model: nn.Module, layers: list, test_dataset_x: torch.Tensor, test_dataset_y: np.ndarray):
    scope = bscope.Scope(model, layers, to_numpy=True)
    scope.use_int_grad(steps=20)
    scope.wrt_topk(k=1, softmax=True)
    scope.log_start(reduction=["ei_split"])

    for xb in tqdm(test_dataset_x.split(128)):
        scope(xb)

    scope.log_stop()

    digits = np.eye(10)[test_dataset_y]

    return scope, digits


def mode_pixel_maps(model: nn.Module, x: torch.Tensor, atoms: np.ndarray, layer: int, device: str):
    x = x.to(device).clone().requires_grad_()
    h = model.network[: 2 * layer + 1](model.flatten(x))
    probs = torch.softmax(model.network[2 * layer + 1:](h), dim=-1)

    dy_dh, = torch.autograd.grad(probs.max(1).values.sum(), h, retain_graph=True)
    atoms = torch.as_tensor(atoms, dtype=h.dtype, device=device)

    maps = [x * torch.autograd.grad(h, x, grad_outputs=atom * dy_dh, retain_graph=True)[0] for atom in atoms]

    return torch.stack(maps, dim=1).detach().cpu().numpy()


def compute_neural_correlations(scope: bscope.Scope, layers: list, device: str, model_path: str):
    datas = []

    for l in range(len(layers)):
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

        print(f"cont {contributions_dim}")

        sae = NNSTSAE(data_dim=contributions_dim, num_atoms=3 * contributions_dim, threshold=theashold, mlp_hidden_dim=contributions_dim).to(device)

        optimizer = setup_optimizer(model=sae, optimizer_type="Adam", learning_rate=lr)

        date = datetime.now(ZoneInfo("America/Sao_Paulo"))

        wandb_run = setup_wandb(f"Sparse_Autoencoders_{l}_{date.strftime('%d/%m/%Y - %H:%M')}", config={"sae/train/n_epochs": n_epochs, "sae/train/input_dim": contributions_dim, "sae/train/theashold": theashold, "sae/train/learning_rate": lr, "sae/train/batch_size": batch_size})

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


def correlation_results(digits: np.ndarray, model_path: str, datas: list, device: str):
    loadings_all, dicts, corrs = [], [], []

    for l, data in enumerate(datas):
        _, loadings, dictionary, _, _, _ = load_sae(f"{model_path}/sae_layer_{l}.pt", data, device=device)

        loadings_all.append(loadings); dicts.append(dictionary)

        if loadings.shape[1] == 0:
            print("Nenhum modo vivo: o SAE colapsou, treine de novo (ou use threshold menor / rede mais larga)")
            corrs.append(None)
            continue

        corr = np.nan_to_num(bscope.mtx_corr(digits, loadings))
        np.save(f"{model_path}/corr_layer_{l}.npy", corr)
        np.save(f"{model_path}/loadings_layer_{l}.npy", loadings)
        np.save(f"{model_path}/dictionary_layer_{l}.npy", dictionary)
        corrs.append(corr)

        for d in range(10):
            k = corr[d].argmax()
            print(f"  digito {d}: modo {k} (corr {corr[d, k]:.2f}) | neuronios que mais pesam: {np.argsort(-np.abs(dictionary[k]))[:7]}")


    return loadings_all, dicts, corrs


def visualization(model: nn.Module, scope: bscope.Scope, test_dataset_x: torch.Tensor, test_dataset_y: np.ndarray, datas: list, loadings_all: list, dicts: list, corrs: list, device: str):
    with torch.no_grad():
        preds = model(test_dataset_x.unsqueeze(1).to(device)).argmax(1).cpu().numpy()

    sizes = [data.shape[1] for data in datas] + [10]

    visualize_all(sizes=sizes, X=test_dataset_x.numpy(), y=test_dataset_y, preds=preds, contribs=scope.log_contributions, datas=datas, loadings=loadings_all, dictionaries=dicts, corrs=corrs)


def main(model_path="checkpoint/test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = setup_model(model_type="MLP", device=device)

    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))

    layers = [model.network[0], model.network[2]]

    _, test_dataloader = setup_dataset(dataset_type="Mnist")

    test_dataset_x, test_dataset_y = test_dataloader.dataset.data.float() / 255, test_dataloader.dataset.targets.numpy()

    scope, digits = compute_contributions(model=model, layers=layers, test_dataset_x=test_dataset_x, test_dataset_y=test_dataset_y)

    datas = compute_neural_correlations(scope=scope, layers=layers, device=device, model_path=model_path)

    loadings_all, dicts, corrs = correlation_results(digits=digits, model_path=model_path, datas=datas, device=device)

    visualization(model=model, scope=scope, test_dataset_x=test_dataset_x, test_dataset_y=test_dataset_y, datas=datas, loadings_all=loadings_all, dicts=dicts, corrs=corrs, device=device)

    maps = mode_pixel_maps(model, test_dataset_x[:5], dicts[0][:5], layer=0, device=device)
    
    plot_pixel_maps(test_dataset_x[:5].numpy(), maps, test_dataset_y[:5], layer=0)


if __name__ == "__main__":
    main()
