"""Methods for playing the game randomly, or as a human."""

from tqdm import tqdm


def play_random(env, steps):
    """Play the environment making uniformly random decisions.

    Args:
        env (gym.Env): the initialized gym environment to play
        steps (int): the number of random steps to take

    Returns:
        None

    """
    try:
        terminated = True
        truncated = True
        progress = tqdm(range(steps))
        for _ in progress:
            if terminated or truncated:
                _ = env.reset()
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            progress.set_postfix(reward=reward, info=info)
            env.render()
    except KeyboardInterrupt:
        pass
    # close the environment
    env.close()


# explicitly define the outward facing API of this module
__all__ = [play_random.__name__]  # pyright: ignore [reportUnsupportedDunderAll]
