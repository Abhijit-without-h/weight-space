import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from export import arrays, export, model_from_arrays, reordered_hidden, shuffled_positions
from games import GAMES
from train import Policy


class ExportTest(unittest.TestCase):
    def test_controls_and_blinding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed, game in enumerate(GAMES):
                torch.manual_seed(seed)
                path = root / f"checkpoint_{seed}.pt"
                torch.save({"game": game, "policy": Policy().state_dict()}, path)
                paths.append(path)
            public = root / "public"
            key = root / "private" / "answer_key.json"
            index = export(paths, public, key)
            self.assertEqual(len(index), 7)
            self.assertEqual(len(list(public.glob("net_*.txt"))), 7)
            self.assertFalse((public / "answer_key.json").exists())
            self.assertEqual(set(json.loads(key.read_text())["files"]), {item["file"] for item in index})
            for item in index:
                content = (public / item["file"]).read_text()
                self.assertTrue(content.startswith("W1 (32x500)\n"))
                self.assertNotIn("pacman", content)
                self.assertNotIn("breakout", content)
                self.assertEqual(len(content), item["characters"])

    def test_shuffle_and_reorder(self):
        source = arrays(Policy())
        rng = np.random.default_rng(123)
        shuffled = shuffled_positions(source, rng)
        self.assertCountEqual(source[0].ravel().tolist(), shuffled[0].ravel().tolist())
        reordered = reordered_hidden(source, rng)
        inputs = torch.rand(16, 500)
        with torch.no_grad():
            before = model_from_arrays(source)(inputs)
            after = model_from_arrays(reordered)(inputs)
        torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
