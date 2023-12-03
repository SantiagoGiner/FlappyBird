import flappy_bird_gymnasium
import gymnasium
from learner import *
from train import get_args


ENV_NAME = "FlappyBird-v0"


# Test the learned policy
def test(args):
    # Make the environment
    env = gymnasium.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Make a BC class and give it the learned policy
    model = BC(state_dim, action_dim, args)
    policy_save_path = os.path.join(args.policy_save_dir, f"{ENV_NAME}.pt")
    model.load(policy_save_path)
    # Arrays to store episode lengths and rewards
    lengths = []
    rewards = []
    # Run the tests
    for _ in range(args.n_tests):
        # Initialize tracking of state and episode information
        done = False
        current_state, _ = env.reset()
        length = 0
        reward = 0
        # Repeat until episode terminates
        while not done:
            # Render the game and get action from learned policy
            env.render()
            qs = model.get_logits(torch.from_numpy(current_state).float()) 
            action = qs.argmax().numpy()
            # Carry out the policy's action and keep track of length and reward
            obs, r, done, _, info = env.step(action)
            current_state = obs
            length += 1
            reward += r
        # Save test length and reward
        lengths.append(length)
        rewards.append(reward)
    print(f"Average episode length: {np.mean(lengths)}")
    print(f"Average reward incurred: {np.mean(rewards)}")


# Main
if __name__ == '__main__':
    args = get_args()
    test(args)