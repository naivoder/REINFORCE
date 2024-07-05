import gymnasium as gym
from agent import Agent
from argparse import ArgumentParser
import utils
from collections import deque
import numpy as np


def generate_animation(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    save_prefix = env_name.split("/")[-1]

    print("Environment:", save_prefix, env.action_space)

    atari_env = "ALE/" in env_name
    if atari_env:
        input_dims = (3, 84, 84)
    else:
        input_dims = env.observation_space.shape

    agent = Agent(
        env_name=save_prefix,
        lr=3e-5,
        input_dims=input_dims,
        n_actions=utils.get_num_actions(env),
        use_cnn=atari_env,
    )

    agent.load_checkpoints()

    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(10):
        frames = []
        total_reward = 0

        state, _ = env.reset()
        if atari_env:
            state = utils.preprocess_frame(state)
            state_buffer = deque([state] * 3, maxlen=3)
            state = np.array(state_buffer)

        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.choose_action(state)

            if isinstance(action, np.ndarray):
                action = action[0]

            next_state, reward, terminated, truncated, _ = env.step(action)

            if atari_env:
                next_state = utils.preprocess_frame(next_state)
                state_buffer.append(next_state)
                state = np.array(state_buffer)

            total_reward += reward

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    utils.save_animation(best_frames, f"environments/{env_name}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", required=True, help="Environment name from Gymnasium"
    )
    args = parser.parse_args()
    generate_animation(args.env)
