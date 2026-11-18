"""Reproduce both game-aware Pong edits on every start and a fresh reset sample."""

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from final_evaluate import CHECKPOINTS
from games import GridGame
from train import Policy, scripted_actions


def edited_policy(state, *, force_stay):
    model = Policy().eval()
    model.load_state_dict(state)
    first, second, head = model.layers[0], model.layers[2], model.layers[4]
    bound = (head.weight.abs().sum(dim=1) + head.bias.abs()).max().item()
    gain = (8 if force_stay else 4) * (1 + bound)
    with torch.no_grad():
        first.weight[0].zero_()
        first.bias[0].zero_()
        for row in range(10):
            first.weight[0, 10 * row] = -row
            first.weight[0, 100 + 10 * row:110 + 10 * row] = row
        second.weight[0].zero_()
        second.weight[0, 0] = 1
        second.bias[0].zero_()
        head.weight[1, 0] -= gain
        head.weight[2, 0] += gain
        if force_stay:
            head.bias[0] += 2 * bound + 1
    return model


@torch.inference_mode()
def rollout(model, starts):
    env = GridGame("pong", len(starts), seed=0)
    env.agent[:, 0], env.moving[:, 0], env.velocity[:, 0] = starts.T
    active = np.ones(env.n, dtype=bool)
    scores = np.zeros(env.n, dtype=int)
    matches = decisions = 0
    for _ in range(150):
        obs = torch.from_numpy(env.observe()).float()
        teacher = scripted_actions(env, None)
        actions = teacher if model is None else model(obs).argmax(1).numpy()
        matches += int((active & (actions == teacher)).sum())
        decisions += int(active.sum())
        _, rewards, done = env.step(actions)
        scores += (active * rewards).astype(int)
        active &= ~done
        if not active.any():
            break
    return scores, matches, decisions


def main():
    torch.set_num_threads(1)
    state = torch.load(CHECKPOINTS["pong"], weights_only=True, map_location="cpu")["policy"]
    baseline = Policy().eval()
    baseline.load_state_dict(state)
    models = {"original": baseline, "first": edited_policy(state, force_stay=False),
              "revision": edited_policy(state, force_stay=True), "scripted": None}
    seed = 2026111508
    sample = GridGame("pong", 1000, seed=seed)
    starts = np.array(list(itertools.product(range(3, 7), range(2, 8), (-1, 1))))
    rollouts = {name: rollout(model, starts) for name, model in models.items()}
    sample_starts = zip(sample.agent[:, 0], sample.moving[:, 0], sample.velocity[:, 0])
    # Pong has no post-reset randomness. A table lookup exactly replays any sampled start.
    indices = {tuple(start): i for i, start in enumerate(starts)}
    draw = [indices[tuple(start)] for start in sample_starts]
    report = {
        "checkpoint": str(CHECKPOINTS["pong"]), "fresh_reset_seed": seed,
        "fresh_resets": len(draw), "enumerated_starts": len(starts),
        "start_columns": ["paddle_row", "ball_row", "vertical_velocity"],
        "starts": starts.tolist(), "models": {},
    }
    for name, model in models.items():
        score, matches, decisions = rollouts[name]
        delta = score - rollouts["original"][0]
        report["models"][name] = {
            "changed_parameters": sum(int((value != state[key]).sum())
                                      for key, value in model.state_dict().items()) if model else None,
            "script_agreement": {"matching_decisions": matches, "active_decisions": decisions,
                                 "fraction": matches / decisions},
            "all_start_returns": score.tolist(), "all_start_mean": float(score.mean()),
            "fresh_reset_mean": float(score[draw].mean()),
            "better_same_worse": [int((delta > 0).sum()), int((delta == 0).sum()),
                                  int((delta < 0).sum())],
        }
    output = Path("runs/pong_edit_revision.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print({name: (data["all_start_mean"], data["fresh_reset_mean"])
           for name, data in report["models"].items()})


if __name__ == "__main__":
    main()
