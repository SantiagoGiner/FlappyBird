from learner import *
from dataset import get_dataloader
import torch
import gymnasium
import flappy_bird_gymnasium
import argparse
from tqdm import tqdm
import os


ENV_NAME = "FlappyBird-v0"


def get_args():
    parser = argparse.ArgumentParser(description="Behavioral cloning")
    # Data directory
    parser.add_argument("--data_dir", default='./expert_data', help="Directory with expert data")
    # Learning arguments
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_dataset_samples", type=int, default=10000, help="Number of samples to start dataset off with")
    # Output directory
    parser.add_argument("--policy_save_dir", default="./learned_policies", help="Directory to save learned policies")
    return parser.parse_args()


def experiment(args):
    # expert dataset loading
    save_path = os.path.join(args.data_dir, f"{ENV_NAME}_dataset.pt")
    expert_dataset = torch.load(save_path)
    # Create environment
    env = gymnasium.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Policy initialization
    learner = BC(state_dim, action_dim, args)
    epoch_losses = []
    dataloader = get_dataloader(expert_dataset, args)
    for _ in tqdm(range(1, args.epochs + 1)):
        loss = 0.0
        for batch in dataloader:
            loss += learner.learn(batch[0], batch[1])
        epoch_losses.append(loss / len(dataloader))
    # Saving policy
    outdir = args.policy_save_dir
    if not os.path.exists(args.policy_save_dir):
        os.makedirs(outdir)
    policy_save_path = os.path.join(outdir, f"{ENV_NAME}.pt")
    learner.save(policy_save_path)

if __name__ == "__main__":
    args = get_args()
    experiment(args)