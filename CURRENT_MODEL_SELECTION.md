# Current Model Selection

This file records the currently adopted models for the thesis workflow so old checkpoints are not mixed with the latest agreed baselines.

## Continuous Model

- Official choice: `CC-SAC low-energy repaired (stage4 bc12 energy-first)`
- Archived checkpoint: [best_model.pth](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/checkpoint/best_model.pth)
- Source run: [sac_cc_stage4_bc12_energyfirst_v1](/F:/毕设/claude-code/runs/sac_cc_stage4_bc12_energyfirst_v1)
- Environment behavior: `uf_control_mode=delta`, `uf_delta_max=3.0`, `q_fp_delta_max=12`, `post_target_fp_governor=off`, optional midcourse/late soft governors off, `low_buffer_fp_guard=on`, `dry_run_buffer_threshold=0.0`

## Why This One

- It restores the previously observed low-energy continuous single-layer behavior under the improved environment semantics.
- It reaches the 400 t production target across all 5 evaluation seeds.
- It keeps hard safety violations at zero.
- It keeps dry-run at zero after aligning the dry-run threshold with the improved physical environment, where the filter press is clipped by available inventory rather than flagged at a positive 0.5 m3 low-buffer threshold.
- It is much lower energy than the older `cc_sac_ft2_20260418` official checkpoint under the current environment.

## 5-Seed Evaluation Summary

Source: [eval_result_5seeds.json](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/eval_result_5seeds.json)

- Seeds: `91-95`
- Mean final mass: `400.113 +/- 0.081 t`
- Mean energy: `95.988 +/- 1.722`
- Mean average `C_uf`: `0.717510`
- Unsafe episode rate: `0.0%`
- Unsafe step rate: `0.0%`
- Dry-run step rate: `0.0%`
- Low-concentration step rate: `1.04%`
- Pass rate: `100%`
- In-band rate: `100%`

## Representative Process Overview

- PNG: [seed91_overview.png](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/artifacts/seed91_overview.png)
- CSV: [seed91_overview.csv](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/artifacts/seed91_overview.csv)
- Summary: [seed91_overview_summary.json](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/artifacts/seed91_overview_summary.json)

## Archive Contents

- Archive root: [cc_sac_lowenergy_repaired_20260429](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429)
- Source run config: [source_run_config.json](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/source_run_config.json)
- Source training log: [source_training_log.json](/F:/毕设/claude-code/archives/cc_sac_lowenergy_repaired_20260429/source_training_log.json)

## Previous Continuous Baseline

- Previous official archive: [cc_sac_ft2_20260418](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418)
- Current-environment 5-seed energy for that older model was about `234.44`, so it is no longer the preferred continuous baseline.

## Action Space Comparison Baselines

Archive root: [action_space_compare_20260429](/F:/毕设/claude-code/archives/action_space_compare_20260429)

| Mode | Algorithm | Action Space | Mean Mass | Mean Energy | Mean C_uf | Unsafe Ep | Dry Run |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CC | SAC | continuous Q_uf + continuous Q_fp | `400.113 +/- 0.081` | `95.988` | `0.717510` | `0.0%` | `0.0%` |
| CD | Hybrid TD3 | continuous Q_uf + binary Q_fp | `421.038 +/- 3.907` | `233.638` | `0.705555` | `0.0%` | `0.0%` |
| DD | DDQN | binary Q_uf + binary Q_fp | `411.345 +/- 16.616` | `559.835` | `0.701932` | `0.0%` | `0.0%` |

Notes:

- `CC` is the completed single-layer continuous baseline.
- `CD` is safe but overproduces slightly because `Q_fp` is only binary.
- `DD` is safe but has large seed variance and high energy because both pumps are all-or-nothing.
- Summary table: [summary.csv](/F:/毕设/claude-code/archives/action_space_compare_20260429/summary.csv)
