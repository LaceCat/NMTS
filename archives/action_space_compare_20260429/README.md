# Action Space Baselines 2026-04-29

This archive records the current single-layer action-space comparison baselines under the repaired environment semantics.

## Common Evaluation Protocol

- Evaluation seeds: `91-95`
- Target dry mass: `400 t`
- Episode horizon: `288` decisions, `5 min` per decision
- Optional soft governors disabled: post-target FP governor, post-target idle seeker, midcourse quality governor, late concentration keeper, late target compensator
- Hard low-buffer FP guard retained
- Dry-run threshold: `0.0`, aligned with the improved environment where infeasible filter-press withdrawal is physically clipped by available inventory

## Selected Baselines

| Mode | Algorithm | Action Space | Mean Mass | Mean Energy | Mean C_uf | Unsafe Ep | Dry Run | Pass Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CC | SAC | continuous Q_uf + continuous Q_fp | 400.113 +/- 0.081 | 95.988 | 0.717510 | 0.0% | 0.0% | 100.0% |
| CD | Hybrid TD3 | continuous Q_uf + binary Q_fp | 421.038 +/- 3.907 | 233.638 | 0.705555 | 0.0% | 0.0% | 40.0% |
| DD | DDQN | binary Q_uf + binary Q_fp | 411.345 +/- 16.616 | 559.835 | 0.701932 | 0.0% | 0.0% | 20.0% |

## Interpretation

- `CC` is the finished continuous-continuous baseline: it lands on the target with very low energy and zero safety/dry-run violations.
- `CD` is a usable continuous-discrete comparison baseline: it is safe, but the binary filter-press branch causes coarser terminal mass control and higher energy.
- `DD` is a usable discrete-discrete comparison baseline: it is safe, but the all-or-nothing pump choices create large seed variance and much higher energy.

## Files

- `summary.csv`: compact comparison table.
- `evals/cc_eval_5seeds.json`: CC 5-seed evaluation.
- `evals/cd_eval_5seeds.json`: CD 5-seed evaluation.
- `evals/dd_eval_5seeds.json`: DD 5-seed evaluation.
- `checkpoints/CC/best_model.pth`: selected CC checkpoint.
- `checkpoints/CD/best_model.pth`: selected CD checkpoint.
- `checkpoints/DD/best_model.pth`: selected DD checkpoint.

## Source Notes

- CC source: `runs/sac_cc_stage4_bc12_energyfirst_v1/checkpoints/best_model.pth`, already repaired and archived as `archives/cc_sac_lowenergy_repaired_20260429`.
- CD source: `runs/cd_ft_debug_20260429_01/checkpoints/best_model.pth`, then a small Q_uf actor-bias calibration `-0.020` to reduce terminal overproduction.
- DD source: `runs/dd_ft_debug_20260429/checkpoints/epoch_1.pth`; later DD updates degraded into overproduction, so epoch 1 is intentionally selected.
