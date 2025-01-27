import gymnasium as gym
import utils
from agent import Agent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd
from collections import deque

warnings.simplefilter("ignore")

environments = [
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v2",
    "ALE/Asteroids-v5",
    "ALE/Breakout-v5",
    "ALE/BeamRider-v5",
    "ALE/Centipede-v5",
    "ALE/DonkeyKong-v5",
    "ALE/DoubleDunk-v5",
    "ALE/Frogger-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MarioBros-v5",
    "ALE/MsPacman-v5",
    "ALE/Pong-v5",
    "ALE/Seaquest-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Tetris-v5",
    "ALE/Gopher-v5",
]


def run_reinforce(args):
    save_prefix = args.env.split("/")[-1]
    env = gym.make(args.env)

    atari_env = "ALE/" in args.env
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

    history, metrics, best_score = [], [], float("-inf")

    for i in range(args.n_games):
        state, _ = env.reset()
        if atari_env:
            state = utils.preprocess_frame(state)
            state_buffer = deque([state] * 3, maxlen=3)
            state = np.array(state_buffer)

        score = 0
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

            agent.store_rewards(reward)
            score += reward
        agent.learn()

        history.append(score)
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()

        metrics.append(
            {
                "episode": i + 1,
                "average_score": avg_score,
                "best_score": best_score,
            }
        )

        print(
            f"[{save_prefix} Episode {i + 1:04}/{args.n_games}]  Average Score = {avg_score:.2f}",
            end="\r",
        )

    print("Environment:", save_prefix, env.action_space, "Top Score =", best_score)
    return history, metrics, best_score, agent


def save_results(env_name, history, metrics, agent):
    save_prefix = env_name.split("/")[-1]
    utils.plot_running_avg(history, save_prefix)
    df = pd.DataFrame(metrics)
    df.to_csv(f"metrics/{save_prefix}_metrics.csv", index=False)
    # save_best_version(env_name, agent)


def save_best_version(env_name, agent, seeds=100):
    agent.load_checkpoints()
    atari_env = "ALE/" in env_name

    best_total_reward = float("-inf")
    best_frames = None

    for _ in range(seeds):
        env = gym.make(env_name, render_mode="rgb_array")
        state, _ = env.reset()
        if atari_env:
            state = utils.preprocess_frame(state)
            state_buffer = deque([state] * 3, maxlen=3)
            state = np.array(state_buffer)

        frames = []
        total_reward = 0

        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())
            action = agent.choose_action(state)
            if isinstance(action, np.ndarray):
                action = action[0]
            next_state, reward, term, trunc, _ = env.step(action)
            if atari_env:
                next_state = utils.preprocess_frame(next_state)
                state_buffer.append(next_state)
                state = np.array(state_buffer)
            total_reward += reward
            state = next_state

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    save_prefix = env_name.split("/")[-1]
    utils.save_animation(best_frames, f"environments/{save_prefix}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "-n",
        "--n_games",
        default=10000,
        type=int,
        help="Number of episodes (games) to run during training",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        history, metrics, best_score, trained_agent = run_reinforce(args)
        save_results(args.env, history, metrics, trained_agent)
    elif args.env is None:
        for env_name in environments:
            args.env = env_name
            history, metrics, best_score, trained_agent = run_reinforce(args)
            save_results(args.env, history, metrics, trained_agent)
