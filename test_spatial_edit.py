"""Check the compiler's behavioral contract and first-terminal evaluation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark_spatial_edit import compile_rule, evaluate
from train import Policy


class SpatialEditTests(unittest.TestCase):
    def test_compiled_rule_matches_symbolic_controller(self):
        # Include both axes, nonstandard action mappings, and zero ties.
        torch.manual_seed(7)
        base = Policy().eval()
        saved = {k: v.clone() for k, v in base.state_dict().items()}
        rng = np.random.default_rng(42)
        xs = rng.integers(0, 2, (240, 5, 10, 10)).astype(np.float32)
        xs[:40, 1] = xs[:40, 0]
        for axis in (0, 1):
            rule = dict(axis=axis, channels=[0, 1], actions=[4, 0, 3])
            net = compile_rule(base, rule)
            moments = (xs * np.indices((10, 10))[axis]).sum((2, 3))
            bins = (np.sign(moments[:, 0] - moments[:, 1]) + 1).astype(int)
            expected = np.array(rule["actions"])[bins]
            with torch.inference_mode():
                actual = net(torch.from_numpy(xs.reshape(-1, 500))).argmax(1).numpy()
            np.testing.assert_array_equal(actual, expected)
        for name, p in base.state_dict().items():
            torch.testing.assert_close(p, saved[name], rtol=0, atol=0)

    def test_constant_compile_dominates_without_training(self):
        torch.manual_seed(9)
        net = compile_rule(Policy().eval(), {"constant": 2})
        with torch.inference_mode():
            self.assertTrue((net(torch.randn(80, 500)).argmax(1) == 2).all())

    def test_evaluation_counts_first_terminal_once(self):
        class Env:
            def __init__(self, *args):
                self.t = 0

            def observe(self):
                return np.zeros((2, 500), np.float32)

            def step(self, actions):
                self.t += 1
                return None, np.array([2.0, 3.0]), np.array([True, self.t >= 3])

        class Net:
            def __call__(self, obs):
                return torch.zeros((2, 5))

        with patch("benchmark_spatial_edit.GridGame", Env):
            self.assertEqual(evaluate("pong", Net(), 1, episodes=2), (2 + 9) / 2)


if __name__ == "__main__":
    unittest.main()
