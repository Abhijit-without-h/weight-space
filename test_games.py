import unittest

import numpy as np

from games import GAMES, GridGame


class GameRulesTest(unittest.TestCase):
    def test_shared_observation_and_batching(self):
        for name in GAMES:
            game = GridGame(name, 4, 42)
            before = game.observe()
            self.assertEqual(before.shape, (4, 500))
            self.assertTrue(np.isin(before, [0, 1]).all())
            self.assertTrue((before.reshape(4, 5, 10, 10)[:, 0].sum((1, 2)) == 1).all())
            game.step(np.zeros(4, int))
            np.testing.assert_array_equal(game.observe().reshape(4, 5, 10, 10)[:, 4], before.reshape(4, 5, 10, 10)[:, 1])
            self.assertEqual(len(game.render(0).splitlines()), 10)

    def test_breakout_brick_and_miss(self):
        game = GridGame("breakout", 2)
        game.moving[:] = [(3, 2), (8, 0)]
        game.velocity[:] = [(-1, 0), (1, 0)]
        game.agent[:, 1] = 9
        _, reward, done = game.step([0, 0])
        self.assertEqual(reward.tolist(), [1, -2])
        self.assertFalse(done[0])
        self.assertTrue(done[1])
        self.assertEqual(game.reward_cells[0, 2, 2], 0)

    def test_pong_paddle(self):
        game = GridGame("pong", 2)
        game.agent[:, 0] = [4, 8]
        game.moving[:] = (4, 1)
        game.velocity[:] = (0, -1)
        _, reward, done = game.step([0, 0])
        self.assertEqual(reward.tolist(), [1, -2])
        self.assertEqual(done.tolist(), [False, True])

    def test_pacman_pellet_and_wall(self):
        game = GridGame("pacman", 1)
        game.moving[0] = (1, 8)
        _, reward, _ = game.step([1])
        self.assertEqual(reward[0], 1)
        self.assertEqual(game.agent[0].tolist(), [7, 1])
        game.agent[0] = (2, 2)
        game.step([4])
        self.assertEqual(game.agent[0].tolist(), [2, 2])

    def test_freeway_crossing_and_collision(self):
        game = GridGame("freeway", 2)
        game.agent[:] = [(1, 4), (2, 5)]
        game.cars[1, 0] = 4
        _, reward, _ = game.step([1, 1])
        self.assertAlmostEqual(reward[0], 0.76)
        self.assertAlmostEqual(reward[1], -0.21)
        self.assertEqual(game.agent[1, 0], 9)
        self.assertEqual(game.crossings.tolist(), [1, 0])
        self.assertEqual(game.hits.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
