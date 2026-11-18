"""Capture real policy steps and one-layer activations for the explainer."""

import json
from pathlib import Path

import numpy as np
import torch

from games import GAMES, GridGame
from train import Policy


CHECKPOINTS = {
    "breakout": Path("runs/breakout_seed0.pt"),
    "pong": Path("runs/pong_seed1.pt"),
    "pacman": Path("runs/pacman_seed0.pt"),
    "freeway": Path("runs/freeway_seed1.pt"),
}


def capture(game, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Policy()
    model.load_state_dict(checkpoint["policy"])
    model.eval()
    env = GridGame(game, 1, 17)
    frames = []
    with torch.no_grad():
        for step in range(36):
            obs = env.observe()[0]
            x = torch.from_numpy(obs.astype(np.float32))[None]
            hidden = model.layers[:4](x)[0]
            logits = model(x)[0]
            action = int(logits.argmax())
            _, reward, done = env.step([action])
            frames.append({
                "step": step,
                "obs": "".join(str(int(v)) for v in obs),
                "hidden": [round(float(v), 5) for v in hidden],
                "logits": [round(float(v), 5) for v in logits],
                "action": action,
                "reward": round(float(reward[0]), 3),
                "return": round(float(env.returns[0]), 3),
                "crossings": int(env.crossings[0]),
                "hits": int(env.hits[0]),
            })
            if done[0]:
                break
    return {"frames": frames,
            "w3": [[round(float(v), 5) for v in row] for row in model.layers[4].weight.detach().numpy()]}


if __name__ == "__main__":
    data = {game: capture(game, CHECKPOINTS[game]) for game in GAMES}
    destination = Path("visual_data.json")
    destination.write_text(json.dumps(data, separators=(",", ":")))
    print(destination, destination.stat().st_size, {k: len(v["frames"]) for k, v in data.items()})
