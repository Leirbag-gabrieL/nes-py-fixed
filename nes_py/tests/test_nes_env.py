"""Test cases for the NESEnv class."""

from unittest import TestCase

import gymnasium as gym
import numpy as np

from nes_py.nes_env import NESEnv

from .rom_file_abs_path import rom_file_abs_path


class ShouldRaiseTypeErrorOnInvalidROMPathType(TestCase):
    def test(self):
        self.assertRaises(TypeError, NESEnv, 0)


class ShouldRaiseValueErrorOnMissingNonexistentROMFile(TestCase):
    def test(self):
        path = rom_file_abs_path("missing.nes")
        self.assertRaises(ValueError, NESEnv, path)


class ShouldRaiseValueErrorOnNonexistentFile(TestCase):
    def test(self):
        self.assertRaises(ValueError, NESEnv, "not_a_file.nes")


class ShouldRaiseValueErrorOnNoniNES_ROMPath(TestCase):
    def test(self):
        self.assertRaises(ValueError, NESEnv, rom_file_abs_path("blank"))


class ShouldRaiseValueErrorOnInvalidiNES_ROMPath(TestCase):
    def test(self):
        self.assertRaises(ValueError, NESEnv, rom_file_abs_path("empty.nes"))


class ShouldRaiseErrorOnStepBeforeReset(TestCase):
    def test(self):
        env = NESEnv(rom_file_abs_path("super-mario-bros-1.nes"))
        self.assertRaises(ValueError, env.step, 0)


class ShouldCreateInstanceOfNESEnv(TestCase):
    def test(self):
        env = NESEnv(rom_file_abs_path("super-mario-bros-1.nes"))
        self.assertIsInstance(env, gym.Env)
        env.close()


def create_smb1_instance():
    """Return a new SMB1 instance."""
    return NESEnv(rom_file_abs_path("super-mario-bros-1.nes"))


class ShouldReadAndWriteMemory(TestCase):
    def test(self):
        env = create_smb1_instance()
        env.reset()
        for _ in range(90):
            env.step(8)
            env.step(0)
        self.assertEqual(129, env.ram[0x0776])
        env.ram[0x0776] = 0
        self.assertEqual(0, env.ram[0x0776])
        env.close()


class ShouldResetAndCloseEnv(TestCase):
    def test(self):
        env = create_smb1_instance()
        env.reset()
        env.close()
        # trying to close again should raise an error
        self.assertRaises(ValueError, env.close)


class ShouldStepEnv(TestCase):
    def test(self):
        env = create_smb1_instance()
        terminated = True
        for _ in range(500):
            if terminated:
                # reset the environment and check the output value
                observation, info = env.reset()
                self.assertIsInstance(observation, np.ndarray)
                self.assertIsInstance(info, dict)
            # sample a random action and check it
            action = env.action_space.sample()
            self.assertIsInstance(action, np.integer)
            # take a step and check the outputs
            output = env.step(action)
            self.assertIsInstance(output, tuple)
            self.assertEqual(5, len(output))
            # check each output
            observation, reward, terminated, truncated, info = output
            self.assertIsInstance(observation, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            # check the render output
            render = env.render("rgb_array")
            self.assertIsInstance(render, np.ndarray)
        env.reset()
        env.close()


class ShouldStepEnvBackupRestore(TestCase):
    def test(self):
        terminated = True
        env = create_smb1_instance()

        for _ in range(250):
            if terminated:
                observation, info = env.reset()
            observation, _, terminated, _, _ = env.step(0)

        backup = observation.copy()

        env._backup()

        for _ in range(250):
            if terminated:
                observation, info = env.reset()
                terminated = False
            observation, _, terminated, _, _ = env.step(0)

        self.assertFalse(np.array_equal(backup, observation))
        env._restore()
        self.assertTrue(np.array_equal(backup, env.screen))
        env.close()
