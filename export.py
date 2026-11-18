"""Export blinded policy weights and three controls after competence checks."""

import argparse
import json
import secrets
from pathlib import Path

import numpy as np
import torch

from games import GAMES
from train import Policy


def arrays(model):
    layers = model.layers
    return [layers[i].weight.detach().numpy().copy() if part == "weight" else layers[i].bias.detach().numpy().copy()
            for i in (0, 2, 4) for part in ("weight", "bias")]


def model_from_arrays(parts):
    model = Policy()
    with torch.no_grad():
        for i, (weight, bias) in zip((0, 2, 4), zip(parts[::2], parts[1::2])):
            model.layers[i].weight.copy_(torch.from_numpy(weight.copy()))
            model.layers[i].bias.copy_(torch.from_numpy(bias.copy()))
    return model


def shuffled_positions(parts, rng):
    result = [x.copy() for x in parts]
    permutation = rng.permutation(100)
    result[0] = result[0].reshape(32, 5, 100)[:, :, permutation].reshape(32, 500)
    return result


def reordered_hidden(parts, rng):
    w1, b1, w2, b2, w3, b3 = [x.copy() for x in parts]
    p1, p2 = rng.permutation(32), rng.permutation(32)
    return [w1[p1], b1[p1], w2[p2][:, p1], b2[p2], w3[:, p2], b3]


def write_weights(path, parts):
    with path.open("w") as stream:
        for name, array in zip(("W1", "b1", "W2", "b2", "W3", "b3"), parts):
            shape = "x".join(str(x) for x in array.shape)
            stream.write(f"{name} ({shape})\n")
            rows = array if array.ndim == 2 else array[None, :]
            for row in rows:
                stream.write(" ".join(f"{float(x):.4f}" for x in row) + "\n")
    chars = path.stat().st_size
    weights = sum(x.size for x in parts)
    return {"file": path.name, "characters": chars, "estimated_tokens": int(weights * 5 + 150)}


def export(checkpoints, outdir, key_path):
    if len(checkpoints) != 4:
        raise ValueError("Provide one chosen checkpoint for each of the four games")
    if key_path.resolve().is_relative_to(outdir.resolve()):
        raise ValueError("Answer key must be outside the blind export directory")
    selected = {}
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        game = checkpoint["game"]
        if game not in GAMES or game in selected:
            raise ValueError("Expected four distinct games")
        model = Policy()
        model.load_state_dict(checkpoint["policy"])
        selected[game] = arrays(model)
    if set(selected) != set(GAMES):
        raise ValueError("Missing game")
    rng = np.random.default_rng(secrets.randbits(64))
    control_game = rng.choice(GAMES)
    source = selected[control_game]
    torch.manual_seed(0)
    items = [("trained", game, parts) for game, parts in selected.items()]
    items += [("untrained", None, arrays(Policy())),
              ("spatial_shuffle", control_game, shuffled_positions(source, rng)),
              ("hidden_reorder", control_game, reordered_hidden(source, rng))]
    x = torch.from_numpy(rng.random((16, 500), dtype=np.float32))
    with torch.no_grad():
        original = model_from_arrays(source)(x)
        equivalent = model_from_arrays(items[-1][2])(x)
    max_difference = float((original - equivalent).abs().max())
    if max_difference > 1e-5:
        raise AssertionError(f"Hidden reorder changed outputs by {max_difference}")
    outdir.mkdir(parents=True, exist_ok=False)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(items)
    answer_key, public_index = {}, []
    for kind, game, parts in items:
        identifier = f"net_{secrets.token_hex(4)}.txt"
        public_index.append(write_weights(outdir / identifier, parts))
        answer_key[identifier] = {"kind": kind, "game": game}
    key_path.write_text(json.dumps({"files": answer_key, "hidden_reorder_max_difference": max_difference}, indent=2))
    (outdir / "public_index.json").write_text(json.dumps(public_index, indent=2))
    return public_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs=4)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--key-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoints, args.outdir, args.key_path), indent=2))
