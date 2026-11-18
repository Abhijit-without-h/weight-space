"""Validate recorded research evidence without requiring model weights."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".safetensors",
    ".gguf",
    ".onnx",
    ".bin",
}


def main():
    files = json.loads((ROOT / "publication-manifest.json").read_text())["files"]
    for name in files:
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), name
        assert Path(name).suffix not in WEIGHT_SUFFIXES, name
        assert not name.endswith(".md") or name == "README.md", name
        assert not name.startswith("web/"), name
        assert "colab" not in name.lower() and "runbook" not in name.lower(), name
    report = json.loads((ROOT / "runs/edit_robustness.json").read_text())
    assert (
        not report["training_during_this_study"]
        and not report["editing_uses_gradients"]
    )
    for field, path in [
        ("protocol_sha256", "protocols/edit-robustness.txt"),
        ("method_sha256", "benchmark_spatial_edit.py"),
    ]:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == report[field]
    accepted = []
    hashes_checked = 0
    for game, data in report["games"].items():
        assert len(data["checkpoints"]) == 3
        for checkpoint in data["checkpoints"]:
            path = ROOT / "runs" / checkpoint["checkpoint"]
            if path.exists():
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == checkpoint["checkpoint_sha256"]
                )
                hashes_checked += 1
            scores = checkpoint["seed_means"]
            assert all(len(values) == 12 for values in scores.values())
            for variant, values in scores.items():
                assert math.isclose(
                    sum(values) / 12, checkpoint["means"][variant], abs_tol=1e-9
                )
            diffs = [a - b for a, b in zip(scores["selected"], scores["original"])]
            mean = sum(diffs) / 12
            half = 2.200985 * math.sqrt(
                sum((value - mean) ** 2 for value in diffs) / 11 / 12
            )
            assert math.isclose(
                mean, checkpoint["selected_delta"]["mean"], abs_tol=1e-9
            )
            for value, saved in zip(
                [mean - half, mean + half], checkpoint["selected_delta"]["ci95"]
            ):
                assert math.isclose(value, saved, abs_tol=1e-9)
            assert checkpoint["accepted"] == (
                checkpoint["calibration_edited"] > checkpoint["calibration_original"]
            )
            assert (
                scores["selected"]
                == scores["forced" if checkpoint["accepted"] else "original"]
            )
            if checkpoint["accepted"]:
                accepted.append(mean)
            if game in ("pong", "breakout"):
                assert checkpoint["changed_parameters"] == 537
    assert len(accepted) == 6 and all(value > 0 for value in accepted)
    print(
        f"PASS: research-only manifest, frozen protocol and method, all paired intervals; {hashes_checked}/12 local checkpoint hashes checked."
    )
    print("Model weights are optional validation inputs and are excluded from GitHub.")


if __name__ == "__main__":
    main()
