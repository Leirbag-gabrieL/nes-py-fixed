"""A method to play gym environments using human IO inputs."""

import time

import gymnasium as gym
from pyglet import clock

from .._image_viewer import ImageViewer
from ..nes_env import NESEnv

# the sentinel value for "No Operation"
_NOP = 0


def play_human(env: NESEnv, callback=None):
    """Play the environment using keyboard as a human.

    Args:
        env: the initialized gym environment to play
        callback: a callback to receive output from the environment

    Returns:
        None

    """
    # ensure the observation space is a box of pixels
    assert isinstance(env.observation_space, gym.spaces.Box)
    # ensure the observation space is either B&W pixels or RGB Pixels
    obs_s = env.observation_space
    is_bw = len(obs_s.shape) == 2
    is_rgb = len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3]
    assert is_bw or is_rgb
    # get the mapping of keyboard keys to actions in the environment
    if hasattr(env, "get_keys_to_action"):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, "get_keys_to_action") and callable(
        getattr(env.unwrapped, "get_keys_to_action")
    ):
        keys_to_action = getattr(env.unwrapped, "get_keys_to_action")()
    else:
        raise ValueError("env has no get_keys_to_action method")
    # create the image viewer
    viewer = ImageViewer(
        env.spec.id if env.spec is not None else env.__class__.__name__,
        env.observation_space.shape[0],  # height
        env.observation_space.shape[1],  # width
        monitor_keyboard=True,
        relevant_keys=set(sum(map(list, keys_to_action.keys()), [])),
    )

    # initialize the environment
    observation, info = env.reset()
    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata["render_fps"]
    last_frame_time = 0
    # start the main game loop
    try:
        while True:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue
            # save frame beginning time for next refresh
            last_frame_time = current_frame_time
            # clock tick
            clock.tick()
            # unwrap the action based on pressed relevant keys
            action = keys_to_action.get(viewer.pressed_keys, _NOP)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            viewer.show(getattr(env.unwrapped, "screen"))
            # pass the observation data through the callback
            if callback is not None:
                callback(observation, action, reward, terminated, next_observation)
            observation = next_observation

            # reset if the environment is terminated
            if terminated or truncated:
                observation, info = env.reset()
                viewer.show(getattr(env.unwrapped, "screen"))

            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                break
    except KeyboardInterrupt:
        pass

    viewer.close()
    env.close()


# explicitly define the outward facing API of the module
__all__ = [play_human.__name__]  # pyright: ignore [reportUnsupportedDunderAll]
