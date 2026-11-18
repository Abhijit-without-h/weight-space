import unittest

from sokoban_demo import Puzzle, PUZZLES, build_report, render, solve, step


class SokobanDemoTest(unittest.TestCase):
    def test_wall_and_blocked_push_are_noops(self):
        walls = frozenset({(r, c) for r in range(5) for c in range(5)
                           if r in (0, 4) or c in (0, 4)})
        puzzle = Puzzle("wall", walls, frozenset({(1, 1)}), (1, 2), ((2, 2),))
        state = (puzzle.player, puzzle.boxes)
        self.assertEqual(step(puzzle, state, 1), state)  # top boundary
        blocked = ((2, 2), ((3, 2),))
        self.assertEqual(step(puzzle, blocked, 2), blocked)  # box cannot be pushed into wall

    def test_push_moves_box_and_player(self):
        puzzle = Puzzle("push", frozenset(), frozenset({(2, 3)}), (2, 1), ((2, 2),))
        self.assertEqual(step(puzzle, (puzzle.player, puzzle.boxes), 4), ((2, 2), ((2, 3),)))

    def test_bfs_returns_optimal_replay_and_report_fields(self):
        for puzzle in PUZZLES:
            result = solve(puzzle)
            self.assertTrue(result["solvable"], puzzle.name)
            self.assertEqual(len(result["states"]), result["solution_depth"] + 1)
            self.assertEqual(set(result["states"][-1][1]), puzzle.goals)
        report = build_report()
        self.assertEqual(report["summary"], {"puzzles": 3, "solvable": 3})
        for item in report["puzzles"]:
            self.assertIn("expanded_states", item)
            self.assertIn("frontier_peak", item)
            self.assertEqual(item["replay"][0]["step"], 0)
            self.assertEqual(len(item["replay"][0]["board"]), 10)

    def test_render_marks_goal_occupied_by_box(self):
        puzzle = Puzzle("render", frozenset(), frozenset({(1, 1)}), (2, 2), ((1, 1),))
        self.assertEqual(render(puzzle, (puzzle.player, puzzle.boxes))[1][1], "*")


if __name__ == "__main__":
    unittest.main()
