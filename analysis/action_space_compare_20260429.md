# Single-Layer Action Space Comparison 2026-04-29

## Purpose

The thesis single-layer comparison is now organized around three action spaces:

- `CC`: continuous underflow pump and continuous filter-press pump.
- `CD`: continuous underflow pump and binary filter-press pump.
- `DD`: binary underflow pump and binary filter-press pump.

All results below use the repaired environment semantics and the same 5-seed deterministic evaluation protocol.

## Common Protocol

- Seeds: `91-95`
- Target: `400 t`
- Horizon: `288` control decisions, `5 min` per decision
- Optional soft governors: disabled
- Hard low-buffer FP guard: enabled
- Dry-run threshold: `0.0`

## Results

| Mode | Selected Checkpoint | Mean Mass | Mean Energy | Mean C_uf | Unsafe Ep | Dry Run | Comment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CC | `archives/action_space_compare_20260429/checkpoints/CC/best_model.pth` | `400.113 +/- 0.081` | `95.988` | `0.717510` | `0.0%` | `0.0%` | Finished baseline |
| CD | `archives/action_space_compare_20260429/checkpoints/CD/best_model.pth` | `421.038 +/- 3.907` | `233.638` | `0.705555` | `0.0%` | `0.0%` | Safe but coarse terminal control |
| DD | `archives/action_space_compare_20260429/checkpoints/DD/best_model.pth` | `411.345 +/- 16.616` | `559.835` | `0.701932` | `0.0%` | `0.0%` | Safe but high variance and high energy |

## Engineering Notes

- `CD` initially had no directly usable checkpoint after the environment repair. The old `best_stage2` checkpoint was safe but overproduced around `425 t`; a short finetune reduced it, and a small `Q_uf` head-bias calibration produced the selected safe `421 t` baseline.
- `DD` long finetuning is unstable. Later DQN updates repeatedly drifted into all-open overproduction. The selected `DD` checkpoint is therefore the first safe finetune epoch, not the final model.
- `DD` is not expected to approach `CC` energy because the action table can only choose `0` or full pump capacity. Its role is to show the cost of discretizing both actuators.

## Archive

- Archive root: `archives/action_space_compare_20260429`
- Summary CSV: `archives/action_space_compare_20260429/summary.csv`
- Evaluation JSON files: `archives/action_space_compare_20260429/evals`
