"""Measure the corrected Pong edit after int8 weight-only storage.

The int8 artifact is restored to float32 before inference. It is a storage
experiment, not native integer compute or a latency speedup.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from benchmark_policy_efficiency import storage_variant, observations
from one_shot_policy_edit import cohort, summarize


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = Path('runs/one_shot_policy_edit_revision2.pt')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['policy']
    full, _, full_bytes = storage_variant(state, 'float32')
    compact, _, compact_bytes = storage_variant(state, 'int8_per_tensor')
    arrays = {}
    for name, value in state.items():
        data = value.numpy()
        if data.ndim == 2:
            scale = np.maximum(np.max(np.abs(data)) / 127, np.finfo(np.float32).tiny)
            arrays[name] = np.clip(np.rint(data / scale), -127, 127).astype(np.int8)
            arrays[name + '_scale'] = np.asarray(scale, dtype=np.float32)
        else:
            arrays[name] = data
    artifact = Path('runs/pong_astra_revision2_int8.npz')
    np.savez(artifact, **arrays)
    if artifact.stat().st_size != compact_bytes:
        raise RuntimeError('Persisted artifact size does not match benchmarked size')
    seeds = [90101, 100103, 110107, 120109, 130111]
    with torch.inference_mode():
        corpus = observations('pong', full, 8192, 20261115)
        full_actions = full(corpus).argmax(-1)
        compact_actions = compact(corpus).argmax(-1)
        agreement = float((full_actions == compact_actions).float().mean())
        exact_returns = np.concatenate([cohort(full, seed, 512) for seed in seeds])
        compact_returns = np.concatenate([cohort(compact, seed, 512) for seed in seeds])
    report = {
        'method': 'Astra-designed Pong revision 2, then per-tensor int8 weight storage; dequantize once before float32 inference',
        'checkpoint': str(checkpoint),
        'int8_artifact': str(artifact),
        'seeds': seeds, 'episodes_per_seed': 512, 'valid_observations': 8192,
        'float32': {'artifact_bytes': full_bytes, 'return': summarize(exact_returns)},
        'int8_storage': {'artifact_bytes': compact_bytes, 'return': summarize(compact_returns),
                         'action_agreement_with_edited_float32': agreement},
        'paired_return_delta': summarize(compact_returns - exact_returns),
        'limitations': ['This is the second game-aware edit after the first failed.',
                        'Int8 weights are dequantized before inference; no integer compute speedup is claimed.',
                        'Score comparison uses the same fixed initial states; post-action trajectories can diverge.']
    }
    dest = Path('runs/combined_edit_int8.json')
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
