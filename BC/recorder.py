import flappy_bird_gymnasium
import gymnasium
import argparse
import numpy as np
import os
import pygame


# Get arguments to run the code
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=str, help="Directory to store expert policies.")
    args = parser.parse_args()
    return args


# Run Flappy Bird and record user's actions
def run_recorder(args):
    # Make the environment and pygame
    env = gymnasium.make("FlappyBird-v0", render_mode="human")
    expert_data = []
    clock = pygame.time.Clock()
    current_state, _ = env.reset()
    # Render the game window
    while True:
        env.render()
        # Get user's action
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if (event.type == pygame.KEYDOWN and
                    (event.key == pygame.K_SPACE or event.key == pygame.K_UP)):
                action = 1
        # Save current state, action pair and take user's action
        expert_data.append((current_state, action))
        obs, reward, done, _, info = env.step(action)
        current_state = obs
        clock.tick(15)
        # Check for termination
        if done:
            env.render()
            break
    env.close()
    # Save expert state, action pairs
    outdir = args.data_directory
    if not os.path.exists(outdir):
            os.makedirs(ddir)
    expert_data = np.array(expert_data, dtype=object)
    np.save(f"{outdir}/expert_policy.npy", expert_data)


# Main
if __name__ == "__main__":
    args = get_args()
    run_recorder(args)
