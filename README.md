# Weight Space

A reproducible study of direct weight edits to small game-playing policies.

**No gradient training or fine-tuning is performed during editing.** The
starting policies were previously PPO-trained. The editor uses teacher
examples and calibration rewards to compile a known rule into existing weights.

[Findings](#findings) · [Method](#method) · [Reproduction](#reproduction) ·
[Protocol](protocols/edit-robustness.txt) · [Apache-2.0 license](LICENSE)

## Findings

On the selected checkpoints, a calibration gate accepted two edits:

| Game     | Original mean return | After gate | Scripted teacher | Decision        |
| -------- | -------------------: | ---------: | ---------------: | --------------- |
| Breakout |                 6.52 |      15.12 |            15.12 | Accept          |
| Pong     |                 8.31 |      10.00 |            10.00 | Accept          |
| Pac-Man  |                44.31 |      44.31 |            27.04 | Retain original |
| Freeway  |                10.06 |      10.06 |             8.10 | Retain original |

Each mean uses 12 evaluation seeds and 64 episodes per seed. The two paddle
edits change 537 of 17,253 parameters (3.11%) and match the scripted teachers.

The fixed editor was subsequently tested on all **12 available checkpoints**:
three existing training seeds per game. All six accepted paddle edits
improved. Six other candidates were rejected. Forced edits harmed all three
Pac-Man checkpoints and one Freeway checkpoint; the other two Freeway edits
were neutral.

[Raw checkpoint evidence](runs/edit_robustness.json) records every gate
decision, paired seed mean, checkpoint hash, and conditional confidence interval.
[Original selected-policy evaluation](runs/spatial_edit_validation.json) and
[discovery controls](runs/spatial_edit.json) are also included.

## Method

The policies are `500 → 32 → 32 → 5` tanh networks. All four games use five
binary channels on a 10×10 grid and the same five actions: stay, up, down,
left, and right.

1. Collect 128 observation/action examples from the existing scripted teacher.
2. Search a fixed family of 1,200 spatial rules and five constant actions.
3. Compile the selected rule through direct tensor assignment and addition.
4. Accept the edit only when its mean return exceeds the original over
   16 separate calibration episodes.
5. Evaluate the selected and forced variants on the fixed paired cohorts.

For paddle games, the compiler writes a spatial contrast into one first-layer
unit, relays it through a second-layer unit, and steers the output head. The
large steering margin implements the fitted controller; it is not a small-norm
perturbation or evidence of recovering a hidden strategy from the original
network. The editing procedure uses no backward pass or optimizer.

## Limitations

- The checkpoint extension is retrospective and reuses the earlier environment
  evaluation cohorts. The gate was designed after observing initial failures.
- There are only three existing checkpoints per game. Two had longer original
  training budgets. Seed intervals condition on the checkpoint and support set;
  they do not estimate uncertainty across independently trained models.
- The controller family, demonstrations, and teacher are restricted. No edit
  beat its teacher or demonstrated transfer to unseen games.
- Rejected edits retain the original policy exactly; their zero selected gains
  are by construction, not proof that the rejected candidate was safe.
- Teacher labels and reward feedback are used. No gradient training during
  editing does not mean no data or no prior policy training.
- The separate LFM3 reader pilot produced no scorable verdict. No LFM3
  identification accuracy or LFM3 weight-edit improvement is established.
- This is an exploratory independent study, not a peer-reviewed publication.

## Reproduction

Requirements: Python 3.12, NumPy, and PyTorch. **Model weights are excluded
from GitHub.** Separately supply the twelve existing `{game}_seed{0,1,2}.pt`
checkpoints; expected hashes are recorded in `runs/edit_robustness.json`.
There is no published Hugging Face checkpoint release yet.

```sh
git clone https://github.com/Abhijit-without-h/weight-space.git
cd weight-space
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate_edit_robustness.py --checkpoint-dir /path/to/existing/checkpoints
```

The command edits and evaluates the supplied policies and writes
`runs/edit_robustness.json`. It does not start training. Missing checkpoints
produce an explicit preflight error. Preserve the recorded result if you want
to compare it with a rerun.

The [frozen protocol](protocols/edit-robustness.txt) and
`benchmark_spatial_edit.py` retain the exact bytes used by the recorded analysis.
A method change needs a separately named protocol and output. Do not overwrite
those frozen inputs to make old evidence appear to validate a new procedure.

The separate corrected manual Pong patch can be checked exhaustively after
placing the original Pong checkpoint at `runs/pong_seed1.pt`:

```sh
python evaluate_pong_edit_revision.py
```

This writes `runs/pong_edit_revision.json`. It covers all 48 possible reset
configurations in the deterministic Pong environment. This manual patch is
separate from the demonstration-based editor above.

## Verification

Evidence validation requires only the Python standard library. The contract
tests require the scientific dependencies but do not require model weights.

```sh
python3 scripts/check_evidence.py
python -m unittest test_games test_export test_paired_data test_sokoban_demo test_spatial_edit
```

The checks validate the publication boundaries, frozen input hashes, paired
interval calculations, game rules, exports, symmetries, solver behavior, and
compiler contracts. No training runs in CI. Optional local checkpoints receive
additional hash checks when present.

## Repository layout

| Location                      | Contents                                                    |
| ----------------------------- | ----------------------------------------------------------- |
| `evaluate_edit_robustness.py` | Direct-edit evaluation across twelve checkpoints            |
| `benchmark_spatial_edit.py`   | Frozen fitter/compiler and historical imitation baseline    |
| `games.py`, `train.py`        | Environments, policy architecture, and historical PPO tools |
| `test_*.py`                   | Scientific contract tests                                   |
| `runs/`                       | Recorded results; local model weights are ignored           |
| `protocols/`                  | Frozen plain-text analysis protocol                         |
| `scripts/check_evidence.py`   | Weight-independent evidence validation                      |
| `publication-manifest.json`   | Explicit public research-file allowlist                     |

The website is maintained separately and is not part of this repository.
Local Markdown notes, model weights, downloaded models, reader answer keys,
private pilot logs, and agent tooling are excluded from version control.
Only this root README is published among the Markdown files.

## Credit and license

An independent experiment by **Abhijit**, inspired by Rikiya Takehi, Jayden
Teoh, Alex Zhang, and Omar Khattab's
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
The grid-game implementation is independent and tests a narrower claim.
AI coding and editing assistance was used. The separate manual Pong patch is
attributed to Astra; the spatial editor is a fixed search-and-compile algorithm.

Original project materials use [Apache License 2.0](LICENSE). External
references and dependencies retain their own licenses. See [NOTICE](NOTICE)
and [CITATION.cff](CITATION.cff).
