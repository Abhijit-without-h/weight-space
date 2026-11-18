"""CPU PPO trainer and evaluator for the four grid policy networks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from games import GAMES, GridGame


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(500, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 5))

    def forward(self, x):
        return self.layers(x)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(500, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def evaluate(game, policy=None, episodes=1000, seed=1234, mode="policy", batch=64):
    env = GridGame(game, batch, seed)
    rng = np.random.default_rng(seed + 101)
    torch.manual_seed(seed)
    returns, lengths, crossings, hits = [], [], [], []
    with torch.no_grad():
        while len(returns) < episodes:
            if mode == "random":
                actions = rng.integers(0, 5, batch)
            elif mode == "scripted":
                actions = scripted_actions(env, rng)
            else:
                x = torch.from_numpy(env.observe().astype(np.float32))
                logits = policy(x)
                actions = (Categorical(logits=logits).sample() if mode == "sampled" else logits.argmax(-1)).numpy()
            _, _, done = env.step(actions)
            returns.extend(env.returns[done].tolist())
            lengths.extend(env.steps[done].tolist())
            crossings.extend(env.crossings[done].tolist())
            hits.extend(env.hits[done].tolist())
            env.reset(done)
    returns = np.asarray(returns[:episodes])
    lengths = np.asarray(lengths[:episodes])
    return {"episodes": episodes, "mean_return": float(returns.mean()),
            "se_return": float(returns.std(ddof=1) / np.sqrt(episodes)),
            "mean_length": float(lengths.mean()),
            "mean_crossings": float(np.mean(crossings[:episodes])),
            "mean_hits": float(np.mean(hits[:episodes]))}


def scripted_actions(env, rng):
    actions = np.zeros(env.n, np.int64)
    if env.game == "breakout":
        diff = env.moving[:, 1] - env.agent[:, 1]
        actions[diff < 0], actions[diff > 0] = 3, 4
    elif env.game == "pong":
        diff = env.moving[:, 0] - env.agent[:, 0]
        actions[diff < 0], actions[diff > 0] = 1, 2
    elif env.game == "pacman":
        for i in range(env.n):
            cells = np.argwhere(env.reward_cells[i])
            if not len(cells):
                continue
            distance = np.abs(cells - env.agent[i]).sum(1)
            target = cells[np.argmin(distance)]
            choices = np.array([0, 1, 2, 3, 4])
            candidates = np.clip(env.agent[i] + np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]), 0, 9)
            legal = env.obstacles[i, candidates[:, 0], candidates[:, 1]] == 0
            costs = np.abs(candidates - target).sum(1) + (~legal) * 100
            actions[i] = choices[np.argmin(costs)]
    else:
        actions[:] = 1  # cross straight up
    return actions


def train(game, seed, updates, outdir, n=64, horizon=64):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 999)
    env = GridGame(game, n, seed)
    policy, value = Policy(), Value()
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4, eps=1e-5)
    outdir.mkdir(parents=True, exist_ok=True)
    curve = outdir / f"{game}_seed{seed}.csv"
    recent = []
    start = time.perf_counter()
    with curve.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["update", "steps", "mean_recent_return", "completed_episodes", "elapsed_seconds"])
        writer.writeheader()
        for update in range(1, updates + 1):
            states, actions, old_logp, rewards, dones, old_values = [], [], [], [], [], []
            for _ in range(horizon):
                x = torch.from_numpy(env.observe().astype(np.float32))
                with torch.no_grad():
                    dist = Categorical(logits=policy(x))
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    v = value(x)
                _, reward, done = env.step(action.numpy())
                recent.extend(env.returns[done].tolist())
                env.reset(done)
                states.append(x)
                actions.append(action)
                old_logp.append(logp)
                rewards.append(torch.from_numpy(reward.copy()))
                dones.append(torch.from_numpy(done.copy()))
                old_values.append(v)
            with torch.no_grad():
                next_v = value(torch.from_numpy(env.observe().astype(np.float32)))
                advantage = torch.zeros(n)
                advantages = []
                for t in range(horizon - 1, -1, -1):
                    alive = (~dones[t]).float()
                    delta = rewards[t] + 0.99 * next_v * alive - old_values[t]
                    advantage = delta + 0.99 * 0.95 * alive * advantage
                    advantages.append(advantage)
                    next_v = old_values[t]
                advantages.reverse()
            x = torch.cat(states)
            a = torch.cat(actions)
            old_lp = torch.cat(old_logp)
            adv = torch.cat(advantages)
            target = adv + torch.cat(old_values)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(3):
                permutation = rng.permutation(len(x))
                for batch_start in range(0, len(x), 512):
                    ix = permutation[batch_start:batch_start + 512]
                    ix = torch.from_numpy(ix)
                    dist = Categorical(logits=policy(x[ix]))
                    ratio = (dist.log_prob(a[ix]) - old_lp[ix]).exp()
                    actor_loss = -torch.minimum(ratio * adv[ix], ratio.clamp(0.8, 1.2) * adv[ix]).mean()
                    critic_loss = 0.5 * (value(x[ix]) - target[ix]).square().mean()
                    loss = actor_loss + critic_loss - 0.01 * dist.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), 0.5)
                    optimizer.step()
            if update % 5 == 0 or update == 1 or update == updates:
                row = {"update": update, "steps": update * n * horizon,
                       "mean_recent_return": float(np.mean(recent[-100:])) if recent else None,
                       "completed_episodes": len(recent), "elapsed_seconds": round(time.perf_counter() - start, 2)}
                writer.writerow(row)
                stream.flush()
                print(game, seed, json.dumps(row), flush=True)
    checkpoint = outdir / f"{game}_seed{seed}.pt"
    torch.save({"policy": policy.state_dict(), "value": value.state_dict(), "seed": seed,
                "steps": updates * n * horizon, "game": game}, checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "render"])
    parser.add_argument("--game", choices=GAMES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--mode", choices=["policy", "sampled", "random", "scripted"], default="policy")
    parser.add_argument("--checkpoint")
    parser.add_argument("--outdir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if args.command == "train":
        train(args.game, args.seed, args.updates, args.outdir)
    elif args.command == "evaluate":
        model = None
        if args.mode in ("policy", "sampled"):
            if not args.checkpoint:
                parser.error("--checkpoint required for policy evaluation")
            model = Policy()
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["policy"])
            model.eval()
        print(json.dumps(evaluate(args.game, model, args.episodes, args.seed, args.mode), indent=2))
    else:
        env = GridGame(args.game, 1, args.seed)
        rng = np.random.default_rng(args.seed)
        for step in range(20):
            print(f"\n{args.game} step {step} return {env.returns[0]:.2f}")
            print(env.render())
            _, _, done = env.step(scripted_actions(env, rng))
            if done[0]:
                break


if __name__ == "__main__":
    main()
