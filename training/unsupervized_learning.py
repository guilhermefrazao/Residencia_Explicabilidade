import torch
import os
import torch.optim as optim


from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from zoneinfo import ZoneInfo
from setup import Setup


def inputs_of(batch):
    # datasets com rótulo devolvem (x, y); sem rótulo devolvem só x. O alvo do autoencoder é a própria entrada
    return batch[0] if isinstance(batch, (list, tuple)) else batch


def train(n_epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: optim, model: nn.Module, criterion: nn, wandb_run, lr: float, device: str):
    global_step = 0
    progress_bar = tqdm(range(n_epochs), desc=type(model).__name__, position=0, leave=True)

    for epoch in progress_bar:
        model.train()

        for batch in train_dataloader:
            train_x = inputs_of(batch).to(device, non_blocking=True)

            optimizer.zero_grad()

            reconstructed = model(train_x)
            loss = criterion(reconstructed, train_x)
            loss.backward()
            optimizer.step()

            global_step += 1

            wandb_run.log({
                "train/n_epochs": n_epochs,
                "train/global_step": global_step,
                "train/learning_rate": lr,
                "losses/train_loss": loss.item(),
            })

        test_loss = test(test_dataloader, model=model, criterion=criterion, device=device)

        wandb_run.log({"losses/test_loss": test_loss})

        progress_bar.set_postfix(train_loss=f"{loss.item():.4f}", test_loss=f"{test_loss:.4f}")


def test(test_dataloader: DataLoader, model: nn.Module, criterion: nn, device: str):
    total, loss_sum = 0, 0.0

    with torch.no_grad():
        model.eval()

        for batch in test_dataloader:
            test_x = inputs_of(batch).to(device, non_blocking=True)

            loss_sum += criterion(model(test_x), test_x).item() * test_x.size(0)
            total += test_x.size(0)

    return loss_sum / total


def main(checkpoints="checkpoint/test", dataset: str="Mnist", model: str="Autoencoder", loss_type: str="MSE", optimizer_type: str="Adam", n_epochs: int=20, setup: Setup=None):
    setup = setup or Setup(dataset_type=dataset, model_type=model, loss_type=loss_type, optimizer_type=optimizer_type)

    date = datetime.now(ZoneInfo("America/Sao_Paulo"))

    wandb_run = setup.init_wandb(run_name=f"{type(setup.model).__name__}_{setup.optimizer_type}_{date.strftime('%d/%m/%Y - %H:%M')}", config=None)

    train(n_epochs=n_epochs, train_dataloader=setup.train_dataloader, test_dataloader=setup.test_dataloader, optimizer=setup.optimizer, model=setup.model, criterion=setup.criterion, wandb_run=wandb_run, lr=setup.learning_rate, device=setup.device)

    wandb_run.finish()

    os.makedirs(checkpoints, exist_ok=True)

    torch.save(setup.model.state_dict(), os.path.join(checkpoints, "model.pt"))


if __name__ == "__main__":
    main()
