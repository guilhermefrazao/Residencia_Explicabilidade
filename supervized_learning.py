import torch
import os
import torch.optim as optim
import wandb


from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from zoneinfo import ZoneInfo
from setup import setup_dataset, setup_model, setup_loss_type, setup_optimizer, setup_wandb


def train(n_epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: optim, model: nn.Module, criterion: nn, wandb_run, lr: int, device: str):
    global_step = 0
    progress_bar = tqdm(range(n_epochs), desc=type(model).__name__, position=0, leave=True)

    for epoch in progress_bar:
        model.train()

        for train_x, train_y in train_dataloader:
            train_x, train_y = train_x.to(device, non_blocking=True), train_y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(train_x)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()

            global_step += 1

            wandb_run.log({
                "train/n_epochs": n_epochs,
                "train/global_step": global_step,   
                "train/learning_rate": lr,
                "losses/train_loss": loss.item(),
            })

        accuracy, test_loss = test(test_dataloader, model=model, criterion=criterion, device=device)

        wandb_run.log({
            "eval/accuracy": accuracy,
            "losses/test_loss": test_loss
        })

        progress_bar.set_postfix(train_loss=f"{loss.item():.4f}", test_loss=f"{test_loss:.4f}", accuracy=f"{accuracy:.2f}%")

            
    return 


def test(test_dataloader, model: nn.Module, criterion, device):
    correct, total, loss_sum =  0, 0, 0.0 
    with torch.no_grad():
        model.eval()
        for test_x, test_y in test_dataloader:
            test_x, test_y = test_x.to(device, non_blocking=True), test_y.to(device, non_blocking=True)

            logits = model(test_x)

            predicted = logits.argmax(dim=1)

            loss_sum += criterion(logits, test_y).item() * test_y.size(0)
            total += test_y.size(0)
            correct += (predicted == test_y).sum().item()    

    return 100 * correct / total, loss_sum / total
    


def main(checkpoints="checkpoint/test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = setup_dataset("Mnist")

    model = setup_model(model_type="MLP", device=device)

    criterion = setup_loss_type(loss_type="CrossEntropy")

    lr = 0.001

    optimizer = setup_optimizer(model=model, optimizer_type="Adam")

    date = datetime.now(ZoneInfo("America/Sao_Paulo"))

    wandb_run = setup_wandb(run_name=f"{type(model).__name__}_adam_{date.strftime('%d/%m/%Y - %H:%M')}", config=None)

    train(n_epochs=20, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, model=model, criterion=criterion, wandb_run=wandb_run, lr=lr, device=device)

    wandb_run.finish()

    os.makedirs(checkpoints, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(checkpoints, "model.pt"))


if __name__ == "__main__":
    main()