"""Follow-up after the four-game benchmark revealed two regressions.

A 16-episode calibration cohort selects between the original and forced edit.
Twelve fresh cohorts measure all methods. The teacher provides a direct
reference. Nine extra support seeds test whether the chosen rule repeats.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_spatial_edit import compile_rule, evaluate, fit, support
from final_evaluate import CHECKPOINTS
from train import Policy

CALIBRATION_SEED = 385997
FRESH_SEEDS = [531103 + 211 * i for i in range(12)]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = dict(calibration_seed=CALIBRATION_SEED, calibration_episodes=16,
                  seeds=FRESH_SEEDS, episodes_per_seed=64, games={})
    for game, path in CHECKPOINTS.items():
        original = Policy().eval()
        original.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['policy'])
        rule = fit(*support(game))
        edited = compile_rule(original, rule)
        base_cal = evaluate(game, original, CALIBRATION_SEED, episodes=16)
        edit_cal = evaluate(game, edited, CALIBRATION_SEED, episodes=16)
        accept = edit_cal > base_cal
        scores = {}
        for name, net in [('base', original), ('forced', edited), ('teacher', None)]:
            scores[name] = [evaluate(game, net, seed) for seed in FRESH_SEEDS]
        scores['selected'] = scores['forced' if accept else 'base']
        diff = np.subtract(scores['selected'], scores['base'])
        half = 2.200985 * diff.std(ddof=1) / np.sqrt(len(diff))
        refit = {}
        for seed in range(714101, 714110):
            refit[seed] = fit(*support(game, seed))
        report['games'][game] = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                calibration_base=base_cal, calibration_forced=edit_cal, accepted=accept,
                fresh_seed_means=scores, fresh_means={k:float(np.mean(v)) for k,v in scores.items()},
                paired_selected_delta=float(diff.mean()), paired_selected_ci95=[float(diff.mean()-half),
                float(diff.mean()+half)], additional_support_rules=refit)
        print(game, accept, report['games'][game]['fresh_means'], flush=True)
    Path('runs/spatial_edit_validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
