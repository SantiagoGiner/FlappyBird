# Demo that chooses actions at random
# The game window should appear

import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human")

obs, _ = env.reset()
while True:
    # Next action:
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)

    # Checking if the player is still alive
    if terminated:
        break

env.close()
