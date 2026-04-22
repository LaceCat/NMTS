# CC SAC Midcourse Quality Governor Notes

Date: 2026-04-19

Objective:
- improve mean underflow concentration (`avg C_uf`)
- keep higher-priority constraints untouched:
  - `unsafe = 0`
  - `dry-run = 0`
  - target completion remains stable (`final mass > 400 t`)

Main checkpoint:
- `archives/cc_sac_ft2_20260418/checkpoint/best_model.pth`

## Environment change

Added a new action post-processing governor in:
- `env/gym_env.py`

Name:
- `midcourse_quality_governor`

Default parameters after tuning:
- `enable_midcourse_quality_governor = True`
- `midcourse_quality_start_mass = 180.0`
- `midcourse_quality_c_uf_target = 0.728`
- `midcourse_quality_q_uf_cap = 12.0`
- `midcourse_quality_q_fp_floor = 18.0`
- `midcourse_quality_buffer_min = 2.5`
- `midcourse_quality_buffer_max = 20.0`

Behavior:
- before target completion, once production has already progressed far enough,
  if `C_uf` is still below the desired band and the buffer inventory is in a
  comfortable range, the governor:
  - caps `Q_uf`
  - floors `Q_fp`
- this nudges the system into a slightly denser operating region without
  violating hard safety constraints

## 5-seed comparison

Seeds:
- `91, 92, 93, 94, 95`

Old baseline (before this governor):
- mean mass: `401.7337 t`
- mean energy: `531.6184`
- mean `avg C_uf`: `0.713716`
- unsafe episode rate: `0.0`
- dry-run episode rate: `0.0`
- below-idle-threshold steps: `35.4`

New current default (with governor):
- mean mass: `401.6290 t`
- mean energy: `530.3900`
- mean `avg C_uf`: `0.714243`
- unsafe episode rate: `0.0`
- dry-run episode rate: `0.0`
- below-idle-threshold steps: `34.8`
- mean `midcourse_quality_governor` active steps: `24.0`

## Delta vs old baseline

- mass: `-0.1047 t`
- energy: `-1.2283`
- mean `avg C_uf`: `+0.000527`
- unsafe: unchanged at `0`
- dry-run: unchanged at `0`

Interpretation:
- this is a valid improvement under the current priority order
- concentration moved upward
- safety and dry-run stayed intact
- energy also improved
- target completion still stayed safely above `400 t`

## Artifacts

Updated rollout figure:
- `plots/current_model_overview_quality_v2/sac_cc_best_model_seed91_overview.png`

Related summary:
- `plots/current_model_overview_quality_v2/sac_cc_best_model_seed91_summary.json`
