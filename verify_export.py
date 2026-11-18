"""Verify final blinded text files, including the hidden-reorder control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export import model_from_arrays


EXPECTED = (("W1", (32, 500)), ("b1", (32,)), ("W2", (32, 32)),
            ("b2", (32,)), ("W3", (5, 32)), ("b3", (5,)))


def read_weights(path):
    lines = path.read_text().splitlines()
    parts, cursor = [], 0
    for name, shape in EXPECTED:
        expected_header = f"{name} ({'x'.join(map(str, shape))})"
        if lines[cursor] != expected_header:
            raise ValueError(f"Unexpected block header in {path.name}")
        cursor += 1
        rows = shape[0] if len(shape) == 2 else 1
        width = shape[-1]
        values = []
        for _ in range(rows):
            row = [float(x) for x in lines[cursor].split()]
            if len(row) != width:
                raise ValueError(f"Wrong row width in {path.name}")
            values.extend(row)
            cursor += 1
        parts.append(np.asarray(values, np.float32).reshape(shape))
    if cursor != len(lines):
        raise ValueError(f"Extra lines in {path.name}")
    return parts


def verify(public, key_path):
    key = json.loads(key_path.read_text())
    if len(key["files"]) != 7:
        raise ValueError("Expected seven blind files")
    parsed = {}
    for name in key["files"]:
        path = public / name
        if not path.name.startswith("net_") or path.suffix != ".txt":
            raise ValueError("Unexpected blind file name")
        content = path.read_text()
        if any(game in content.lower() for game in ("breakout", "pong", "pacman", "freeway")):
            raise ValueError("Game name leaked into a blind file")
        parsed[name] = read_weights(path)
    reordered = next(name for name, info in key["files"].items() if info["kind"] == "hidden_reorder")
    game = key["files"][reordered]["game"]
    original = next(name for name, info in key["files"].items() if info["kind"] == "trained" and info["game"] == game)
    torch.manual_seed(888)
    inputs = torch.randint(0, 2, (1000, 500), dtype=torch.float32)
    with torch.no_grad():
        before = model_from_arrays(parsed[original])(inputs)
        after = model_from_arrays(parsed[reordered])(inputs)
    maximum = float((before - after).abs().max())
    matching_actions = int((before.argmax(-1) == after.argmax(-1)).sum())
    if maximum > 1e-5 or matching_actions != len(inputs):
        raise AssertionError("Hidden reorder does not preserve policy outputs")
    return {"files": len(parsed), "weights_per_file": sum(x.size for x in parsed[original]),
            "random_inputs": len(inputs), "max_output_difference": maximum,
            "matching_actions": matching_actions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=Path("blind_exports"))
    parser.add_argument("--key", type=Path, default=Path("private_exports/answer_key.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.public, args.key), indent=2))
