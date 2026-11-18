"""Retrospective checkpoint robustness of direct weight edits. No training.

The frozen protocol is protocols/edit-robustness.txt. Existing PPO checkpoints
are read only. No optimizer, backward pass, or checkpoint write is used.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_spatial_edit import compile_rule, evaluate, fit, support, SUPPORT_SEED
from games import GAMES
from train import Policy
from validate_spatial_edit import CALIBRATION_SEED, FRESH_SEEDS

ROOT = Path(__file__).resolve().parent


def paired_summary(base, after):
    """Intervals across evaluation seeds, conditional on a fixed checkpoint."""
    diff = np.subtract(after, base)
    half = 2.200985 * diff.std(ddof=1) / np.sqrt(len(diff))
    return dict(
        mean=float(diff.mean()),
        ci95=[float(diff.mean() - half), float(diff.mean() + half)],
    )


@torch.inference_mode()
def run(checkpoint_dir=None):
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else ROOT / "runs"
    started = time.perf_counter()
    protocol = ROOT / "protocols/edit-robustness.txt"
    report = dict(
        study="Direct weight editing across twelve existing checkpoints",
        analysis="retrospective; evaluation cohorts reused from the earlier follow-up",
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        method_sha256=hashlib.sha256(
            (ROOT / "benchmark_spatial_edit.py").read_bytes()
        ).hexdigest(),
        support_seed=SUPPORT_SEED,
        support_examples=128,
        calibration_seed=CALIBRATION_SEED,
        calibration_episodes=16,
        evaluation_seeds=FRESH_SEEDS,
        episodes_per_seed=64,
        training_during_this_study=False,
        editing_uses_gradients=False,
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        threads=torch.get_num_threads(),
        games={},
        limitations=[
            "Three existing training seeds per game; two checkpoints had longer training budgets.",
            "One shared demonstration set per game, with a hand-specified controller family.",
            "Existing evaluation cohorts are reused; this extends checkpoint coverage only.",
            "Per-checkpoint intervals condition on the checkpoint, teacher, and support set.",
            "No new strategy discovery, unseen-game generalization, or inference speedup is established.",
            "No new training or fine-tuning; the starting policies were previously PPO-trained.",
        ],
    )
    for game in GAMES:
        xs, ys = support(game)
        rule = fit(xs, ys)
        teacher = [evaluate(game, None, seed) for seed in FRESH_SEEDS]
        rows = []
        for training_seed in range(3):
            path = checkpoint_dir / f"{game}_seed{training_seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            original = Policy().eval()
            original.load_state_dict(checkpoint["policy"])
            edited = compile_rule(original, rule)
            original_cal = evaluate(game, original, CALIBRATION_SEED, episodes=16)
            edited_cal = evaluate(game, edited, CALIBRATION_SEED, episodes=16)
            accepted = edited_cal > original_cal
            scores = {
                name: [evaluate(game, net, seed) for seed in FRESH_SEEDS]
                for name, net in [("original", original), ("forced", edited)]
            }
            scores["selected"] = scores["forced" if accepted else "original"]
            scores["teacher"] = teacher
            rows.append(
                dict(
                    training_seed=training_seed,
                    training_transitions=checkpoint["steps"],
                    checkpoint=path.name,
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    changed_parameters=sum(
                        int((a != b).count_nonzero())
                        for a, b in zip(original.parameters(), edited.parameters())
                    ),
                    parameter_count=sum(p.numel() for p in original.parameters()),
                    calibration_original=original_cal,
                    calibration_edited=edited_cal,
                    accepted=accepted,
                    seed_means=scores,
                    means={k: float(np.mean(v)) for k, v in scores.items()},
                    forced_delta=paired_summary(scores["original"], scores["forced"]),
                    selected_delta=paired_summary(
                        scores["original"], scores["selected"]
                    ),
                )
            )
            row = rows[-1]
            print(
                game,
                training_seed,
                "accept=",
                accepted,
                "selected gain=",
                round(row["selected_delta"]["mean"], 4),
                flush=True,
            )
        report["games"][game] = dict(
            rule=rule,
            checkpoints=rows,
            accepted_count=sum(r["accepted"] for r in rows),
            accepted_regressions=sum(
                r["accepted"] and r["selected_delta"]["mean"] < 0 for r in rows
            ),
            forced_regressions=sum(r["forced_delta"]["mean"] < 0 for r in rows),
            mean_selected_gain=float(
                np.mean([r["selected_delta"]["mean"] for r in rows])
            ),
            selected_gain_range=[
                min(r["selected_delta"]["mean"] for r in rows),
                max(r["selected_delta"]["mean"] for r in rows),
            ],
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    missing = [
        f"{game}_seed{seed}.pt"
        for game in GAMES
        for seed in range(3)
        if not (args.checkpoint_dir / f"{game}_seed{seed}.pt").is_file()
    ]
    if missing:
        parser.error(
            "Existing model weights are required and are not included in GitHub. "
            "Use --checkpoint-dir to supply them. Missing: " + ", ".join(missing)
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run(args.checkpoint_dir)
    destination = ROOT / "runs/edit_robustness.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
