"""Independent 1,000-episode evaluation for the frozen four-game selection."""

import json
from pathlib import Path

import torch

from games import GAMES
from train import Policy, evaluate


CHECKPOINTS = {
    "breakout": Path("runs/breakout_seed0.pt"),
    "pong": Path("runs/pong_seed1.pt"),
    "pacman": Path("runs/pacman_seed0.pt"),
    "freeway": Path("runs/freeway_seed1.pt"),
}


def main():
    report = {}
    for game in GAMES:
        model = Policy()
        checkpoint = torch.load(CHECKPOINTS[game], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["policy"])
        model.eval()
        report[game] = {
            mode: evaluate(game, model if mode in ("policy", "sampled") else None,
                           episodes=1000, seed=10000, mode=mode)
            for mode in ("random", "scripted", "policy", "sampled")
        }
    destination = Path("runs/final_evaluation.json")
    destination.write_text(json.dumps(report, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
