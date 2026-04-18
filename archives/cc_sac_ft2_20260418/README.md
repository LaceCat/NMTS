# Archived Model: CC-SAC ft2

This archive freezes the current SAC continuous-control model that was accepted as the working baseline on `2026-04-18`.

## Core identity

- Model name: `CC-SAC ft2 (crossbonus + multiseed)`
- Checkpoint: [best_model.pth](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/checkpoint/best_model.pth)
- Source run: [sac_cc_v2_crossbonus_multiseed_ft2](/F:/毕设/claude-code/runs/sac_cc_v2_crossbonus_multiseed_ft2)

## Operating setup

- `mode = CC`
- `uf_control_mode = delta`
- `uf_delta_max = 3.0`
- `q_fp_delta_max = 12.0`
- `post_target_fp_governor = off`
- `low_buffer_fp_guard = on`

## 5-seed evaluation snapshot

Source: [eval_result_5seeds.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/eval_result_5seeds.json)

- Mean final mass: `400.13 +- 0.12 t`
- Mean energy: `550.74 +- 2.07`
- `pass_rate = 100%`
- `in-band rate = 100%`

## Included files

- [run_config.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/run_config.json)
- [training_log.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/training_log.json)
- [seed91_overview.png](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_overview.png)
- [seed91_overview.csv](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_overview.csv)
- [seed91_overview_summary.json](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_overview_summary.json)
- [seed91_playback.gif](/F:/毕设/claude-code/archives/cc_sac_ft2_20260418/artifacts/seed91_playback.gif)
