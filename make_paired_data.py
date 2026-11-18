"""Generate public demonstration data without reading/exporting the blinded key."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from games import ACTIONS, GAMES, GridGame
from make_viz_data import CHECKPOINTS
from train import Policy


def signed_permutation(model: Policy, seed: int = 718) -> Policy:
    """Apply independent signed permutations to both odd-tanh hidden layers.

    h1'=S1 P1 h1, h2'=S2 P2 h2; compensate every outgoing column.
    Thus logits are identical for every real input in exact arithmetic.
    """
    result = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    p1, p2 = (torch.tensor(rng.permutation(32)) for _ in range(2))
    s1, s2 = (torch.tensor(rng.choice([-1., 1.], 32), dtype=model.layers[0].weight.dtype) for _ in range(2))
    with torch.no_grad():
        a, b, c = model.layers[0], model.layers[2], model.layers[4]
        result.layers[0].weight.copy_(a.weight[p1] * s1[:, None])
        result.layers[0].bias.copy_(a.bias[p1] * s1)
        result.layers[2].weight.copy_(b.weight[p2][:, p1] * s2[:, None] * s1[None, :])
        result.layers[2].bias.copy_(b.bias[p2] * s2)
        result.layers[4].weight.copy_(c.weight[:, p2] * s2[None, :])
    return result


def paint_pong_reward(model: Policy) -> Policy:
    """Replace unused reward-channel columns with fixed, bounded +/-.25 stripes."""
    result = copy.deepcopy(model)
    pattern = torch.tensor([.25 if (i // 10 + i % 10) % 2 else -.25 for i in range(100)], dtype=model.layers[0].weight.dtype)
    with torch.no_grad():
        result.layers[0].weight[:, 200:300].copy_(pattern.expand(32, -1))
    return result


def compare(a, b, x):
    with torch.no_grad():
        left, right = a(x), b(x)
    return {"observations": len(x), "max_logit_delta": float((left - right).abs().max()),
            "action_disagreements": int((left.argmax(-1) != right.argmax(-1)).sum())}


def footprint(model):
    return model.layers[0].weight.detach().abs().mean(0).reshape(5, 100).tolist()


def capture(game, path, frames=36):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    original = Policy().eval()
    original.load_state_dict(checkpoint["policy"])
    models = {"original": original, "transformed": signed_permutation(original)}
    if game == "pong":
        models["painted"] = paint_pong_reward(original)
    env = GridGame(game, 1, 17)
    records = []
    with torch.no_grad():
        for step in range(frames):
            obs = env.observe()[0]
            x = torch.from_numpy(obs.astype(np.float32))[None]
            record = {"step": step, "obs": "".join(map(str, obs.tolist()))}
            for name, model in models.items():
                logits = model(x)[0]
                record[name] = {"logits": logits.tolist(), "action": int(logits.argmax())}
            _, reward, done = env.step([record["original"]["action"]])
            record.update(reward=float(reward[0]), **{"return": float(env.returns[0])},
                          crossings=int(env.crossings[0]), hits=int(env.hits[0]), done=bool(done[0]))
            records.append(record)
            env.reset(done)
    # Probe valid observations from independent environments and random actions.
    probe = GridGame(game, 32, 902)
    rng = np.random.default_rng(903)
    valid = []
    for _ in range(32):
        valid.append(probe.observe().copy())
        _, _, done = probe.step(rng.integers(0, 5, 32))
        probe.reset(done)
    valid = torch.tensor(np.concatenate(valid), dtype=torch.float32)
    arbitrary = torch.tensor(rng.normal(size=(256, 500)), dtype=torch.float32)
    verification = {"signed_permutation": {"scope": "all real inputs (algebraic identity; probes measure floating-point error)",
        "valid_states": compare(original, models["transformed"], valid),
        "arbitrary_input_probes": compare(original, models["transformed"], arbitrary)}}
    if game == "pong":
        assert not torch.count_nonzero(valid[:, 200:300])
        verification["painted"] = {"scope": "valid Pong states only: reward channel is identically zero; no all-input equivalence claim",
            "reward_channel_nonzeros": int(torch.count_nonzero(valid[:, 200:300])),
            "valid_states": compare(original, models["painted"], valid),
            "arbitrary_input_probes": compare(original, models["painted"], arbitrary),
            "pattern": "fixed checkerboard, magnitude 0.25; all W1 rows, columns 200:300"}
    for condition in verification.values():
        assert condition["valid_states"]["action_disagreements"] == 0
        assert condition["valid_states"]["max_logit_delta"] < 1e-5
    return {"frames": records, "footprints": {("signed" if k == "transformed" else k): footprint(v) for k, v in models.items()}, "verification": verification}


def generate(destination):
    torch.set_num_threads(1)
    data = {"meta": {"schema_version": 1, "actions": list(ACTIONS), "channels": ["agent", "moving", "reward", "obstacle", "previous"],
        "conditions": {"original": "Selected trained checkpoint", "transformed": "Signed hidden-unit permutations (all-input equivalence)",
                       "painted": "Pong reward-channel painting (valid-state equivalence only)"},
        "frame_seed": 17, "frames_per_game": 36, "frame_semantics": "Observation before original greedy action; reward/counters after action; reset after terminal frame. Other conditions evaluated on identical observations.",
        "footprint_semantics": "5x100 mean absolute W1 over hidden rows; signed permutations preserve this aggregate even when individual rows change.",
        "public_safe": "Named demonstration conditions only; no blinded identities or answer key included."},
        "games": {game: capture(game, CHECKPOINTS[game]) for game in GAMES}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/paired_data.json")
    args = parser.parse_args()
    generated = generate(args.output)
    print(args.output, {k: len(v["frames"]) for k, v in generated["games"].items()})
