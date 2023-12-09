from learner import *
from dataset import get_dataloader
import torch
import gymnasium
import flappy_bird_gymnasium
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
plt.style.use("./paper.mplstyle")


ENV_NAME = "FlappyBird-v0"


# Arguments
def get_args():
    parser = argparse.ArgumentParser(description="Behavioral cloning")
    # Data directory
    parser.add_argument("--data_dir", default='./expert_data', help="Directory with expert data")
    # Learning arguments
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_dataset_samples", type=int, default=1000, help="Number of samples to start dataset off with")
    # Output directory
    parser.add_argument("--policy_save_dir", type=str, default="./learned_policies",
        help="Directory to save learned policies")
    # Arguments for loss plotting
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to store training loss plot")
    parser.add_argument("--plot_name", type=str, default="BC_losses", help="Name of training loss plot")
    # Number of tests to run when calling test.py
    parser.add_argument("--n_tests", type=int, default=10, help="Number of tests to run")
    parser.add_argument("--hide_game", action="store_true", help="Hide the game window when testing")
    return parser.parse_args()


# Learning through behavioral cloning (BC)
def experiment(args):
    # Get expert data
    save_path = os.path.join(args.data_dir, f"{ENV_NAME}_dataset.pt")
    expert_dataset = torch.load(save_path)
    # Create environment
    env = gymnasium.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Make a BC object and load training data into batches
    learner = BC(state_dim, action_dim, args)
    epoch_losses = []
    dataloader = get_dataloader(expert_dataset, args)
    # Learn through gradient descent
    for _ in tqdm(range(1, args.epochs + 1)):
        loss = 0.0
        for batch in dataloader:
            loss += learner.learn(batch[0], batch[1])
        epoch_losses.append(loss / len(dataloader))
    # Save the learned policy
    outdir = args.policy_save_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    policy_save_path = os.path.join(outdir, f"{ENV_NAME}.pt")
    learner.save(policy_save_path)
    # Plot the losses, if requested
    loss_dir = args.plot_dir
    if loss_dir:
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        epochs = np.arange(1, args.epochs + 1)
        plt.plot(epochs, epoch_losses)
        plt.title("BC Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(loss_dir, f"{args.plot_name}.png"))

# Main
if __name__ == "__main__":
    args = get_args()
    experiment(args)
