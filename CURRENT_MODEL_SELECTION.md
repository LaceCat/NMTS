# Current Model Selection

This file records the currently adopted models for the thesis workflow so we do not mix old checkpoints with the latest agreed baselines.

## Continuous Model

- Official choice: `CC-SAC ft2 (crossbonus + multiseed)`
- Archived checkpoint: [best_model.pth](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/checkpoint/best_model.pth)
- Source run: [sac_cc_v2_crossbonus_multiseed_ft2](/F:/毕设/claude-code/runs/sac_cc_v2_crossbonus_multiseed_ft2)
- Environment behavior:
  `uf_control_mode=delta`, `uf_delta_max=3.0`, `q_fp_delta_max=12`, `post_target_fp_governor=off`, `low_buffer_fp_guard=on`

### Why this one

- It satisfies the agreed priority:
  1. `unsafe = 0`
  2. `final_mass >= 400 t` across seeds
  3. energy remains low without giving up robustness
- It is the current SAC continuous-control baseline to be used for demonstrations, plots, and later thesis comparison unless explicitly overridden.

### 5-seed evaluation summary

Source: [eval_result_5seeds.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/eval_result_5seeds.json)

- Seeds: `91-95`
- Mean final mass: `400.13 +- 0.12 t`
- Mean energy: `550.74 +- 2.07`
- `pass_rate (>=400t) = 100%`
- `in-band rate (400~420t) = 100%`

### Representative process overview

- PNG: [seed91_overview.png](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_overview.png)
- CSV: [seed91_overview.csv](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_overview.csv)
- Summary: [seed91_overview_summary.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_overview_summary.json)

### Representative playback

- GIF: [seed91_playback.gif](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_playback.gif)

### Archive contents

- Archive root: [cc_sac_ft2_20260418](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418)
- Run config: [run_config.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/run_config.json)
- Training log: [training_log.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/training_log.json)
