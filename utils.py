import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import gymnasium as gym


def get_num_actions(env):
    action_space = env.action_space

    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        return action_space.shape[0]
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return np.prod(action_space.nvec)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        return action_space.n
    else:
        raise NotImplementedError(
            "Unsupported action space type: {}".format(type(action_space))
        )


def linear_scale(epoch, start_value=0.6, min_value=0.1, num_epochs=2000):
    if epoch > num_epochs:
        epoch = num_epochs

    slope = (min_value - start_value) / num_epochs
    std = slope * epoch
    return start_value + std


# reminder to give credit to OP of this one
def action_adapter(a, max_a):
    return 2 * (a - 0.5) * max_a


def clip_reward(x):
    if x < -1:
        return -1
    elif x > 1:
        return 1
    else:
        return x


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # return np.expand_dims(resized, axis=0)
    return resized


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def plot_running_avg(scores, env):
    """
    Plot the running average of scores with a window of 100 games.

    This function calculates the running average of a list of scores and
    plots the result using matplotlib. The running average is calculated
    over a window of 100 games, providing a smooth plot of score trends over time.

    Parameters
    ----------
    scores : list or numpy.ndarray
        A list or numpy array containing the scores from consecutive games.

    Notes
    -----
    This function assumes that `scores` is a list or array of numerical values
    that represent the scores obtained in each game or episode. The running
    average is computed and plotted, which is useful for visualizing performance
    trends in tasks such as games or simulations.

    Examples
    --------
    >>> scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> plot_running_avg(scores)
    This will plot a graph showing the running average of the scores over a window of 10 games.
    """
    avg = np.zeros_like(scores)
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(avg)
    plt.title("Running Average per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_running_avg.png")
    plt.close()
