import math
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from collections import namedtuple, deque
from datetime import datetime
from itertools import count
from torch import nn
from tqdm import tqdm
from zoneinfo import ZoneInfo
from setup import Setup

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net: nn.Module, env, steps_done: int, eps_start: float, eps_end: float, eps_decay: int, device: str):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(memory: ReplayMemory, policy_net: nn.Module, target_net: nn.Module, optimizer: optim, criterion: nn, batch_size: int, gamma: float, device: str):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(num_episodes: int, env, memory: ReplayMemory, policy_net: nn.Module, target_net: nn.Module, optimizer: optim, criterion: nn, wandb_run, batch_size: int, gamma: float, tau: float, eps_start: float, eps_end: float, eps_decay: int, device: str):
    steps_done = 0
    episode_durations = []
    progress_bar = tqdm(range(num_episodes), desc=type(policy_net).__name__, position=0, leave=True)

    for i_episode in progress_bar:
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_return = 0.0  # recompensa acumulada dentro do episódio
        for t in count():
            action = select_action(state, policy_net, env, steps_done, eps_start, eps_end, eps_decay, device)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())

            episode_return += float(reward)

            wandb_run.log({
                "reinforcement/step": steps_done,
            })

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer, criterion, batch_size, gamma, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)

                wandb_run.log({
                    "reinforcement/episode": i_episode,
                    "reinforcement/episode_duration": t + 1,
                    "reinforcement/episode_return": episode_return,
                })

                progress_bar.set_postfix(ep_duration=t + 1, ep_return=episode_return)
                break

    return episode_durations


def main(checkpoints="checkpoint/test", dataset: str="CartPole-v1", model: str="DQN", loss_type: str="SmoothL1", optimizer_type: str="AdamW", render: bool=False, setup: Setup=None):
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4

    setup = setup or Setup(dataset_type=dataset, model_type=model, loss_type=loss_type, optimizer_type=optimizer_type, learning_rate=LR, render=render)

    # o Setup já dimensiona a rede pelos espaços de ação e observação do ambiente
    env, device, policy_net, criterion, optimizer = setup.environment, setup.device, setup.model, setup.criterion, setup.optimizer

    target_net = setup.create_model()
    target_net.load_state_dict(policy_net.state_dict())

    MEMORY_CAPACITY = 10000
    memory = ReplayMemory(MEMORY_CAPACITY)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    date = datetime.now(ZoneInfo("America/Sao_Paulo"))

    config = {
        "reinforcement/environment": dataset,
        "reinforcement/model": type(policy_net).__name__,
        "reinforcement/loss_type": loss_type,
        "reinforcement/optimizer_type": setup.optimizer_type,
        "reinforcement/learning_rate": setup.learning_rate,
        "reinforcement/batch_size": BATCH_SIZE,
        "reinforcement/gamma": GAMMA,
        "reinforcement/tau": TAU,
        "reinforcement/eps_start": EPS_START,
        "reinforcement/eps_end": EPS_END,
        "reinforcement/eps_decay": EPS_DECAY,
        "reinforcement/replay_memory_capacity": MEMORY_CAPACITY,
        "reinforcement/num_episodes": num_episodes,
    }

    wandb_run = setup.init_wandb(run_name=f"{type(policy_net).__name__}_{setup.optimizer_type}_{date.strftime('%d/%m/%Y - %H:%M')}", config=config)

    train(num_episodes=num_episodes, env=env, memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer, criterion=criterion, wandb_run=wandb_run, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, device=device)

    wandb_run.finish()

    os.makedirs(checkpoints, exist_ok=True)

    torch.save(policy_net.state_dict(), os.path.join(checkpoints, "model.pt"))


if __name__ == "__main__":
    main()
