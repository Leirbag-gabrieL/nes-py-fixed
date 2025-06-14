"""An environment wrapper to convert binary to discrete action space."""

import gymnasium as gym
from gymnasium import Env, Wrapper


class JoypadSpace(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        "right": 0b10000000,
        "left": 0b01000000,
        "down": 0b00100000,
        "up": 0b00010000,
        "start": 0b00001000,
        "select": 0b00000100,
        "B": 0b00000010,
        "A": 0b00000001,
        "NOOP": 0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env, actions: list):
        """Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = " ".join(button_list)

    def step(self, action):
        """Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            The environment's step output which is a tuple of:
            - observation (np.ndarray): next frame as a result of the given action
            - reward (float) : amount of reward returned after given action
            - terminated (boolean): whether the episode has ended naturally or not (e.g., Mario died, Mario completed the stage)
            - truncated (boolean): whether the episode was truncated by either reaching the maximum number of steps or the truncate function returning True (only if using SuperMarioBrosEnv or SuperMarioBrosRandomStagesEnv)
            - info (dict): contains auxiliary diagnostic information
        """
        # take the step and record the output
        return self.env.step(self._action_map[action])

    def reset(self, *args, **kwargs):
        """Reset the environment and return the initial observation."""
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        """Render the environment."""
        return self.env.unwrapped.render(*args, **kwargs)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        # get the old mapping of keys to actions
        if not hasattr(self.env.unwrapped, "get_keys_to_action") or not callable(
            getattr(self.env.unwrapped, "get_keys_to_action")
        ):
            raise ValueError("self.env has no get_keys_to_action method")
        old_keys_to_action = getattr(self.env.unwrapped, "get_keys_to_action")()
        # invert the keys to action mapping to lookup key combos by action
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        # create a new mapping of keys to actions
        keys_to_action = {}
        # iterate over the actions and their byte values in this mapper
        for action, byte in self._action_map.items():
            # get the keys to press for the action
            keys = action_to_keys[byte]
            # set the keys value in the dictionary to the current discrete act
            keys_to_action[keys] = action

        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]


# explicitly define the outward facing API of this module
__all__ = [JoypadSpace.__name__]  # pyright: ignore [reportUnsupportedDunderAll]
