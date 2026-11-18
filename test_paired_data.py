import unittest

import numpy as np
import torch

from games import GridGame
from make_paired_data import paint_pong_reward, signed_permutation
from train import Policy


class PairedDataTests(unittest.TestCase):
    def test_signed_permutation_preserves_arbitrary_inputs_and_changes_weights(self):
        torch.manual_seed(12)
        original = Policy().double()
        changed = signed_permutation(original)
        x = torch.randn(120, 500, dtype=torch.float64) * 3
        torch.testing.assert_close(original(x), changed(x), rtol=0, atol=2e-15)
        self.assertFalse(torch.equal(original.layers[0].weight, changed.layers[0].weight))
        self.assertTrue(torch.equal(original.layers[4].bias, changed.layers[4].bias))

    def test_paint_only_changes_unused_columns_and_requires_valid_states(self):
        torch.manual_seed(13)
        original = Policy()
        changed = paint_pong_reward(original)
        before, after = original.state_dict(), changed.state_dict()
        for name in before:
            if name != 'layers.0.weight':
                self.assertTrue(torch.equal(before[name], after[name]))
        self.assertTrue(torch.equal(before['layers.0.weight'][:, :200], after['layers.0.weight'][:, :200]))
        self.assertTrue(torch.equal(before['layers.0.weight'][:, 300:], after['layers.0.weight'][:, 300:]))
        self.assertEqual(float(after['layers.0.weight'][:, 200:300].abs().max()), .25)
        env = GridGame('pong', 64, 23)
        rng = np.random.default_rng(4)
        for _ in range(24):
            x = torch.tensor(env.observe(), dtype=torch.float32)
            self.assertEqual(int(x[:, 200:300].count_nonzero()), 0)
            torch.testing.assert_close(original(x), changed(x), rtol=0, atol=0)
            _, _, done = env.step(rng.integers(0, 5, 64))
            env.reset(done)
        self.assertGreater(float((original(torch.ones(1, 500)) - changed(torch.ones(1, 500))).abs().max().detach()), 1e-4)


if __name__ == '__main__':
    unittest.main()
