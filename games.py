"""Four batched 10x10 games with a shared five-channel observation."""

from __future__ import annotations

import numpy as np

SIZE = 10
ACTIONS = ("stay", "up", "down", "left", "right")
MOVE = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16)
GAMES = ("breakout", "pong", "pacman", "freeway")


class GridGame:
    def __init__(self, game: str, n: int, seed: int = 0):
        if game not in GAMES or n < 1:
            raise ValueError("Unknown game or empty batch")
        self.game, self.n = game, n
        self.rng = np.random.default_rng(seed)
        self.idx = np.arange(n)
        self.agent = np.zeros((n, 2), np.int16)
        self.moving = np.zeros((n, 2), np.int16)
        self.velocity = np.zeros((n, 2), np.int16)
        self.cars = np.zeros((n, 8), np.int16)
        self.car_dir = np.array([1, -1, 1, -1, 1, -1, 1, -1], np.int16)
        self.reward_cells = np.zeros((n, SIZE, SIZE), np.uint8)
        self.obstacles = np.zeros((n, SIZE, SIZE), np.uint8)
        self.previous = np.zeros((n, SIZE, SIZE), np.uint8)
        self.steps = np.zeros(n, np.int32)
        self.returns = np.zeros(n, np.float32)
        self.crossings = np.zeros(n, np.int32)
        self.hits = np.zeros(n, np.int32)
        self.reset()

    def reset(self, mask=None):
        ids = self.idx if mask is None else self.idx[np.asarray(mask, bool)]
        if len(ids) == 0:
            return self.observe()
        self.steps[ids] = 0
        self.returns[ids] = 0
        self.crossings[ids] = 0
        self.hits[ids] = 0
        self.previous[ids] = 0
        self.reward_cells[ids] = 0
        self.obstacles[ids] = 0
        if self.game == "breakout":
            self.agent[ids] = np.stack((np.full(len(ids), 9), self.rng.integers(2, 8, len(ids))), 1)
            self.moving[ids] = np.stack((np.full(len(ids), 7), self.rng.integers(2, 8, len(ids))), 1)
            self.velocity[ids] = np.stack((np.full(len(ids), -1), self.rng.choice([-1, 1], len(ids))), 1)
            self.reward_cells[ids, 1:3, 1:9] = 1
        elif self.game == "pong":
            self.agent[ids] = np.stack((self.rng.integers(3, 7, len(ids)), np.zeros(len(ids), int)), 1)
            self.moving[ids] = np.stack((self.rng.integers(2, 8, len(ids)), np.full(len(ids), 5)), 1)
            self.velocity[ids] = np.stack((self.rng.choice([-1, 1], len(ids)), np.full(len(ids), -1)), 1)
            self.obstacles[ids, :, 9] = 1  # right bounce wall
        elif self.game == "pacman":
            self.agent[ids] = (8, 1)
            self.moving[ids] = (1, 8)  # ghost
            self.obstacles[ids, 0, :] = self.obstacles[ids, 9, :] = 1
            self.obstacles[ids, :, 0] = self.obstacles[ids, :, 9] = 1
            self.obstacles[ids, 2:8, 3] = 1
            self.obstacles[ids, 2:8, 6] = 1
            self.obstacles[ids, 4, 3:7] = 0  # cross passage
            self.reward_cells[ids, 1:9, 1:9] = 1
            self.reward_cells[ids] *= 1 - self.obstacles[ids]
            self.reward_cells[ids, 8, 1] = 0
            self.reward_cells[ids, 1, 8] = 0
        else:
            self.agent[ids] = np.stack((np.full(len(ids), 9), self.rng.integers(0, 10, len(ids))), 1)
            self.cars[ids] = self.rng.integers(0, 10, (len(ids), 8))
            self.reward_cells[ids, 0, :] = 1  # crossing line
        return self.observe()

    def _moving_frame(self):
        frame = np.zeros((self.n, SIZE, SIZE), np.uint8)
        if self.game == "freeway":
            for lane in range(8):
                frame[self.idx, lane + 1, self.cars[:, lane]] = 1
        else:
            frame[self.idx, self.moving[:, 0], self.moving[:, 1]] = 1
        return frame

    def observe(self):
        obs = np.zeros((self.n, 5, SIZE, SIZE), np.uint8)
        obs[self.idx, 0, self.agent[:, 0], self.agent[:, 1]] = 1
        obs[:, 1] = self._moving_frame()
        obs[:, 2] = self.reward_cells
        obs[:, 3] = self.obstacles
        obs[:, 4] = self.previous
        return obs.reshape(self.n, 500)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,) or np.any((actions < 0) | (actions > 4)):
            raise ValueError("Expected one action 0..4 per environment")
        self.previous = self._moving_frame()
        rewards = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        a = self.agent.copy()
        if self.game == "breakout":
            a[:, 1] = np.clip(a[:, 1] + MOVE[actions, 1], 0, 9)
            self.agent = a
            ball = self.moving + self.velocity
            wall = (ball[:, 1] < 0) | (ball[:, 1] > 9)
            self.velocity[wall, 1] *= -1
            ball[:, 1] = np.clip(ball[:, 1], 0, 9)
            top = ball[:, 0] < 0
            self.velocity[top, 0] = 1
            ball[top, 0] = 0
            brick = self.reward_cells[self.idx, ball[:, 0].clip(0, 9), ball[:, 1]] > 0
            rewards[brick] += 1
            self.reward_cells[self.idx[brick], ball[brick, 0], ball[brick, 1]] = 0
            self.velocity[brick, 0] *= -1
            hit = (ball[:, 0] >= 9) & (np.abs(ball[:, 1] - a[:, 1]) <= 1)
            self.velocity[hit, 0] = -1
            ball[hit, 0] = 8
            miss = (ball[:, 0] >= 9) & ~hit
            rewards[miss] -= 2
            done |= miss | (self.reward_cells.sum((1, 2)) == 0)
            self.moving = ball.clip(0, 9)
        elif self.game == "pong":
            a[:, 0] = np.clip(a[:, 0] + MOVE[actions, 0], 0, 9)
            self.agent = a
            ball = self.moving + self.velocity
            wall_y = (ball[:, 0] < 0) | (ball[:, 0] > 9)
            self.velocity[wall_y, 0] *= -1
            ball[:, 0] = np.clip(ball[:, 0], 0, 9)
            right = ball[:, 1] >= 9
            self.velocity[right, 1] = -1
            ball[right, 1] = 8
            hit = (ball[:, 1] <= 0) & (np.abs(ball[:, 0] - a[:, 0]) <= 1)
            self.velocity[hit, 1] = 1
            ball[hit, 1] = 1
            rewards[hit] += 1
            miss = (ball[:, 1] <= 0) & ~hit
            rewards[miss] -= 2
            done |= miss
            self.moving = ball.clip(0, 9)
        elif self.game == "pacman":
            target = np.clip(a + MOVE[actions], 0, 9)
            blocked = self.obstacles[self.idx, target[:, 0], target[:, 1]] > 0
            self.agent[~blocked] = target[~blocked]
            found = self.reward_cells[self.idx, self.agent[:, 0], self.agent[:, 1]] > 0
            rewards[found] += 1
            self.reward_cells[self.idx[found], self.agent[found, 0], self.agent[found, 1]] = 0
            ghost_moves = MOVE[self.rng.integers(0, 5, self.n)]
            ghost = np.clip(self.moving + ghost_moves, 0, 9)
            blocked = self.obstacles[self.idx, ghost[:, 0], ghost[:, 1]] > 0
            self.moving[~blocked] = ghost[~blocked]
            caught = np.all(self.agent == self.moving, axis=1)
            rewards[caught] -= 5
            done |= caught | (self.reward_cells.sum((1, 2)) == 0)
        else:
            old_potential = 0.03 * (9 - a[:, 0])
            target = a + MOVE[actions]
            self.agent = np.clip(target, 0, 9)
            self.cars = (self.cars + self.car_dir) % 10
            lane = self.agent[:, 0] - 1
            road = (lane >= 0) & (lane < 8)
            hit = road & (self.cars[self.idx, np.clip(lane, 0, 7)] == self.agent[:, 1])
            crossed = self.agent[:, 0] == 0
            self.hits += hit
            self.crossings += crossed
            rewards[crossed] += 1
            self.agent[hit, 0] = 9
            self.agent[crossed, 0] = 9
            new_potential = 0.03 * (9 - self.agent[:, 0])
            new_potential[self.steps + 1 >= 120] = 0
            rewards += 0.99 * new_potential - old_potential
        self.steps += 1
        done |= self.steps >= {"breakout": 200, "pong": 150, "pacman": 150, "freeway": 120}[self.game]
        self.returns += rewards
        return self.observe(), rewards, done

    def render(self, env: int = 0):
        if not 0 <= env < self.n:
            raise IndexError(env)
        obs = self.observe()[env].reshape(5, 10, 10)
        marks = np.full((10, 10), ".", dtype="<U1")
        marks[obs[3] > 0] = "#"
        marks[obs[2] > 0] = "$"
        marks[obs[1] > 0] = "o"
        marks[obs[0] > 0] = "A"
        return "\n".join("".join(row) for row in marks)
