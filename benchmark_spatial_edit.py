"""Fit a bounded one-unit grid-policy edit from demonstrations and test fixed cohorts.

Run from the repository root with .venv312/bin/python benchmark_spatial_edit.py.
The teacher supplies labels only during fitting. The fitted classifier family
contains constants and signs of differences between two channel moments on one
axis, motivated by the existing hand-written Pong edit. Each game is fit anew.
"""
import copy
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from final_evaluate import CHECKPOINTS
from games import GridGame
from train import Policy, scripted_actions

SUPPORT_SEED = 137177
TEST_SEEDS = [251101 + 107 * i for i in range(12)]
EPISODES = 64
STEPS = 32
RANDOM_SEEDS = [391, 392, 393, 394, 395]


def support(game, seed=SUPPORT_SEED):
    env = GridGame(game, 8, seed)
    rng = np.random.default_rng(seed + 1)
    xs, ys = [], []
    for _ in range(16):
        xs.append(env.observe())
        ys.append(scripted_actions(env, rng))
        _, _, done = env.step(rng.integers(0, 5, env.n))
        env.reset(done)
    return np.concatenate(xs), np.concatenate(ys)


def fit(xs, ys):
    grid = xs.reshape(-1, 5, 10, 10)
    dominant = int(np.bincount(ys).argmax())
    best = dict(constant=dominant, correct=int((ys == dominant).sum()))
    mappings = np.array(list(itertools.permutations(range(5), 3)))
    for axis, coord in enumerate(np.indices((10, 10))):
        moments = (grid * coord).sum(axis=(2, 3))
        for i, j in itertools.combinations(range(5), 2):
            bins = np.sign(moments[:, i] - moments[:, j]) + 1
            correct = (mappings[:, bins] == ys).sum(axis=1)
            k = correct.argmax()
            if correct[k] > best['correct']:
                best = dict(axis=axis, channels=[i, j], actions=mappings[k].tolist(),
                            correct=int(correct[k]))
    return best


@torch.no_grad()
def compile_rule(original, rule):
    net = copy.deepcopy(original)
    first, relay, head = net.layers[0], net.layers[2], net.layers[4]
    # tanh gives |h| <= 1, so twice this bound dominates any base logit gap.
    bound = (head.weight.abs().sum(1) + head.bias.abs()).max().item()
    margin = 2 * bound + 1
    if 'constant' in rule:
        head.bias[rule['constant']] += margin
    else:
        coord = torch.from_numpy(np.indices((10, 10))[rule['axis']]).float()
        row = first.weight[0].view(5, 10, 10)
        row.zero_()
        i, j = rule['channels']
        row[i].copy_(coord)
        row[j].copy_(-coord)
        first.bias[0].zero_()
        relay.weight[0].zero_()
        relay.weight[0, 0] = 1
        relay.bias[0].zero_()
        neg, zero, pos = rule['actions']
        gain = 2 * margin / np.tanh(np.tanh(1.0))
        head.weight[neg, 0] -= gain
        head.weight[pos, 0] += gain
        head.bias[zero] += margin
    return net


@torch.no_grad()
def random_control(original, edited, seed):
    net = copy.deepcopy(original)
    rng = torch.Generator().manual_seed(seed)
    for p, target in zip(net.parameters(), edited.parameters()):
        delta = target - p
        if delta.any():
            noise = torch.randn(p.shape, generator=rng) * (delta != 0)
            p.add_(noise * (delta.norm() / noise.norm()))
    return net


def fine_tune(original, xs, ys):
    net = copy.deepcopy(original)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(STEPS):
        optimizer.zero_grad()
        F.cross_entropy(net(xs), ys).backward()
        optimizer.step()
    return net


@torch.inference_mode()
def evaluate(game, net, seed, episodes=EPISODES):
    env = GridGame(game, episodes, seed)
    active = np.ones(episodes, bool)
    scores = np.zeros(episodes)
    rng = np.random.default_rng(seed + 1)
    while active.any():
        if net is None:
            actions = scripted_actions(env, rng)
        else:
            actions = net(torch.from_numpy(env.observe()).float()).argmax(-1).numpy()
        _, reward, done = env.step(actions)
        scores[active] += reward[active]
        active &= ~done
    return float(scores.mean())


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = dict(support_seed=SUPPORT_SEED, test_seeds=TEST_SEEDS, episodes_per_seed=EPISODES,
                  random_seeds=RANDOM_SEEDS, support_examples=128, fine_tune_steps=STEPS,
                  fine_tune_optimizer='Adam(lr=0.01), full batch', python=platform.python_version(),
                  torch=torch.__version__, numpy=np.__version__, games={})
    # Initialize optimizer machinery before measuring per-game fit time.
    torch.optim.Adam(Policy().parameters())
    for game, path in CHECKPOINTS.items():
        original = Policy().eval()
        original.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['policy'])
        t = time.perf_counter()
        xs, ys = support(game)
        x, y = torch.from_numpy(xs).float(), torch.from_numpy(ys)
        collected = time.perf_counter()
        rule = fit(xs, ys)
        edited = compile_rule(original, rule)
        fitted = time.perf_counter()
        tuned = fine_tune(original, x, y)
        trained = time.perf_counter()
        variants = dict(original=original, edit=edited, full_imitation=tuned)
        for seed in RANDOM_SEEDS:
            variants[f'random_{seed}'] = random_control(original, edited, seed)
        game_result = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           rule=rule, collection_seconds=collected-t, edit_seconds=fitted-collected,
                           full_imitation_seconds=trained-fitted, variants={})
        for name, net in variants.items():
            scores = np.array([evaluate(game, net, seed) for seed in TEST_SEEDS])
            with torch.no_grad():
                changes = [p - q for p, q in zip(net.parameters(), original.parameters())]
                acc = (net(x).argmax(-1) == y).float().mean().item()
            baseline = game_result['variants'].get('original', {}).get('seed_means', scores)
            diff = scores - baseline
            # 97.5th percentile for Student t on 11 df, seed clusters are the units.
            half = 2.200985 * diff.std(ddof=1) / np.sqrt(len(TEST_SEEDS))
            game_result['variants'][name] = dict(mean=float(scores.mean()), seed_means=scores.tolist(),
                        paired_delta=float(diff.mean()), paired_ci95=[float(diff.mean()-half), float(diff.mean()+half)],
                        changed_scalars=sum(int(d.count_nonzero()) for d in changes),
                        delta_norm=float(torch.cat([d.flatten() for d in changes]).norm()),
                        support_accuracy=acc)
            assert name != 'edit' or acc == rule['correct'] / len(ys)
        report['games'][game] = game_result
        print(game, rule, {k: round(v['mean'], 3) for k, v in game_result['variants'].items()}, flush=True)
    Path('runs/spatial_edit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
