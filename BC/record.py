import flappy_bird_gymnasium
import gymnasium
import argparse
import os
import pygame
import torch
from dataset import ExpertData, ExpertDataset


# Get arguments to run the code
def get_args():
    parser = argparse.ArgumentParser()
    # Output parameters
    parser.add_argument("--data_directory", type=str, default="./expert_data/",
        help="Directory to store expert data.")
    parser.add_argument("--data_name", type=str, default=None, help="Name of output file.")
    # Number of times to run game
    parser.add_argument("--n_games", type=int, default=1,
        help="Number of times to run game.")
    return parser.parse_args()


# Run Flappy Bird and record user's actions
def record_user(args):
    # Make the environment and pygame clock variable
    env = gymnasium.make("FlappyBird-v0", render_mode="human")
    clock = pygame.time.Clock()
    # Arrays in which to save user's states and actions
    states = []
    actions = []
    # Iterate over the number of trajectories to collect
    for _ in range(args.n_games):
        traj_states = []
        traj_actions = []
        done = False
        current_state, _ = env.reset()
        # Repeat until terminated
        while not done:
            env.render()
            # Set action to go up if user presses space bar or up key
            keys = pygame.key.get_pressed()
            action = 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] else 0
            # Save current state, action pair and take user's action
            traj_states.append(torch.from_numpy(current_state).float())
            traj_actions.append(torch.tensor(action))
            obs, reward, done, _, info = env.step(action)
            current_state = obs
            clock.tick(50)
        states += traj_states
        actions += traj_actions
    # Make state and action tensors for saving
    state_tensor = torch.stack(states)
    action_tensor = torch.stack(actions)
    # Make the states and actions an ExpertDataset class
    dataset = ExpertDataset(ExpertData(state_tensor, action_tensor))
    # Create output directory if it does not exist
    outdir = args.data_directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Get name of output file and save expert data
    name = args.data_name
    if not name:
        name = env.spec.id + "_dataset"
    torch.save(dataset, f"{outdir}/{name}.pt")


# Main
if __name__ == "__main__":
    args = get_args()
    record_user(args)