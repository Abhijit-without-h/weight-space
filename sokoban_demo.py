"""Deterministic Sokoban puzzle pack, BFS oracle, and webpage data exporter.

This is an isolated complexity demo. It does not register Sokoban in GAMES or
the PPO training pipeline. Run ``python sokoban_demo.py`` to write
``web/sokoban-data.json``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

SIZE = 10
ACTIONS = (("stay", 0, 0), ("up", -1, 0), ("down", 1, 0),
           ("left", 0, -1), ("right", 0, 1))


@dataclass(frozen=True)
class Puzzle:
    name: str
    walls: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    player: tuple[int, int]
    boxes: tuple[tuple[int, int], ...]


def _boundary() -> set[tuple[int, int]]:
    return {(r, c) for r in range(SIZE) for c in range(SIZE)
            if r in (0, SIZE - 1) or c in (0, SIZE - 1)}


PUZZLES = (
    Puzzle("single-corridor", frozenset(_boundary() | {(r, 4) for r in range(2, 8) if r != 5}),
           frozenset({(2, 7)}), (7, 2), ((5, 3),)),
    Puzzle("two-boxes", frozenset(_boundary() | {(3, c) for c in range(2, 8) if c != 5}
                                  | {(6, 4), (6, 5)}),
           frozenset({(2, 7), (7, 7)}), (7, 2), ((4, 3), (5, 6))),
    Puzzle("detour", frozenset(_boundary() | {(r, 5) for r in range(1, 8) if r != 6}
                               | {(4, c) for c in range(2, 5)}),
           frozenset({(2, 7)}), (7, 2), ((6, 3),)),
)


State = tuple[tuple[int, int], tuple[tuple[int, int], ...]]


def step(puzzle: Puzzle, state: State, action: int) -> State:
    """Apply one legal action; illegal pushes and wall moves leave state fixed."""
    if not 0 <= action < len(ACTIONS):
        raise ValueError("action must be in 0..4")
    (name, dr, dc) = ACTIONS[action]
    del name
    (r, c), boxes = state
    target = (r + dr, c + dc)
    if target in puzzle.walls:
        return state
    box_set = set(boxes)
    if target in box_set:
        beyond = (target[0] + dr, target[1] + dc)
        if beyond in puzzle.walls or beyond in box_set:
            return state
        box_set.remove(target)
        box_set.add(beyond)
        boxes = tuple(sorted(box_set))
    return target, boxes


def solve(puzzle: Puzzle) -> dict:
    """Return shortest action sequence and standard BFS workload metrics."""
    if len(puzzle.boxes) != len(puzzle.goals):
        raise ValueError("puzzle must have the same number of boxes and goals")
    initial: State = (puzzle.player, tuple(sorted(puzzle.boxes)))
    if initial[0] in puzzle.walls or set(initial[1]) & puzzle.walls or set(initial[1]) & puzzle.goals:
        raise ValueError("invalid initial state")
    queue = deque([initial])
    parent: dict[State, tuple[State, int] | None] = {initial: None}
    expanded = 0
    frontier_peak = 1
    terminal = None
    while queue:
        state = queue.popleft()
        expanded += 1
        if set(state[1]) == puzzle.goals:
            terminal = state
            break
        for action in range(len(ACTIONS)):
            nxt = step(puzzle, state, action)
            if nxt == state or nxt in parent:
                continue
            parent[nxt] = (state, action)
            queue.append(nxt)
        frontier_peak = max(frontier_peak, len(queue))
    if terminal is None:
        return {"solvable": False, "solution_depth": None, "expanded_states": expanded,
                "frontier_peak": frontier_peak, "actions": [], "states": []}
    actions = []
    cursor = terminal
    while parent[cursor] is not None:
        previous, action = parent[cursor]
        actions.append(action)
        cursor = previous
    actions.reverse()
    states = [initial]
    for action in actions:
        states.append(step(puzzle, states[-1], action))
    return {"solvable": True, "solution_depth": len(actions), "expanded_states": expanded,
            "frontier_peak": frontier_peak, "actions": actions, "states": states}


def render(puzzle: Puzzle, state: State) -> list[str]:
    board = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    for r, c in puzzle.walls:
        board[r][c] = "#"
    for r, c in puzzle.goals:
        board[r][c] = "+"
    for r, c in state[1]:
        board[r][c] = "*" if (r, c) in puzzle.goals else "$"
    r, c = state[0]
    board[r][c] = "@" if (r, c) in puzzle.goals else "A"
    return ["".join(row) for row in board]


def _puzzle_data(puzzle: Puzzle) -> dict:
    result = solve(puzzle)
    replay = [{"step": i, "action": "start" if i == 0 else ACTIONS[result["actions"][i - 1]][0],
               "player": list(state[0]), "boxes": [list(p) for p in state[1]],
               "board": render(puzzle, state)} for i, state in enumerate(result["states"])]
    return {"name": puzzle.name, "size": SIZE, "walls": [list(p) for p in sorted(puzzle.walls)],
            "goals": [list(p) for p in sorted(puzzle.goals)], "initial_player": list(puzzle.player),
            "initial_boxes": [list(p) for p in sorted(puzzle.boxes)],
            "box_count": len(puzzle.boxes), "wall_count": len(puzzle.walls),
            "solvable": result["solvable"], "solution_depth": result["solution_depth"],
            "expanded_states": result["expanded_states"], "frontier_peak": result["frontier_peak"],
            "optimal_actions": [ACTIONS[a][0] for a in result["actions"]],
            "replay": replay}


def build_report(puzzles: Iterable[Puzzle] = PUZZLES) -> dict:
    entries = [_puzzle_data(puzzle) for puzzle in puzzles]
    return {"schema_version": 1, "title": "Sokoban BFS complexity demo",
            "method": {"solver": "unit-cost breadth-first search over player position and sorted box positions",
                       "expanded_states": "dequeued states, including the goal state when reached",
                       "frontier_peak": "maximum queued states observed after an expansion",
                       "actions": [name for name, _, _ in ACTIONS],
                       "limits": ["10x10 deterministic hand-authored puzzle pack",
                                  "metrics describe this exact pack and solver, not general game difficulty",
                                  "no learned policy or RL performance is evaluated"]},
            "summary": {"puzzles": len(entries), "solvable": sum(p["solvable"] for p in entries)},
            "puzzles": entries}


def main() -> None:
    destination = Path(__file__).parent / "runs" / "sokoban_data.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_report(), indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
