"""Run the same PPO budget for three seeds per game and save score summaries."""

import argparse
import json
from pathlib import Path

import torch

from games import GAMES
from train import Policy, evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=GAMES)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--outdir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)
    report = {}
    summary_path = args.outdir / ("training_summary_" + "_".join(args.games) + ".json")
    for game in args.games:
        scores = {mode: evaluate(game, episodes=1000, seed=1234, mode=mode) for mode in ("random", "scripted")}
        scores["seeds"] = {}
        for seed in (0, 1, 2):
            checkpoint = train(game, seed, args.updates, args.outdir)
            model = Policy()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["policy"])
            model.eval()
            scores["seeds"][str(seed)] = evaluate(game, model, episodes=1000, seed=2000 + seed, mode="policy")
            report[game] = scores
            summary_path.write_text(json.dumps(report, indent=2))
            print("EVAL", game, seed, json.dumps(scores["seeds"][str(seed)]), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
