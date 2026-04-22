# CC SAC Candidate Scan (2026-04-19)

Evaluation setting:
- action mode: `CC`
- controller: `SAC`
- environment guards: post-target governor `on`, low-buffer guard `on`
- evaluation seeds: `91~95` unless otherwise noted

## Current best safe completion baseline

Checkpoint:
- `archives/cc_sac_ft2_20260418/checkpoint/best_model.pth`

Original 5-seed summary before the terminal-governor improvement:
- mean mass: `401.244 t`
- mean energy: `532.943`
- mean C_uf: `0.713736`
- mean below-idle-threshold steps: `34.0`
- unsafe episode rate: `0.0`
- dry-run steps: `0.0`

Interpretation:
- This is still the strongest all-around safe-and-in-band baseline.
- It reaches the target robustly and keeps a relatively high average underflow concentration.

## Default-environment improvement landed in code

Environment change:
- strengthened the late-target compensator defaults in `env/gym_env.py`
- new defaults:
  - `late_target_window_minutes = 120`
  - `late_target_mass_gap_limit = 12.0`
  - `late_target_q_uf_bias_max = 4.0`
  - `late_target_q_fp_bias_max = 9.0`

Same checkpoint under the new default environment:
- mean mass: `401.734 t`
- mean energy: `531.618`
- mean C_uf: `0.713716`
- mean below-idle-threshold steps: `35.4`
- unsafe episode rate: `0.0`
- dry-run steps: `0.0`

Interpretation:
- This is the best completed improvement of this round.
- It uses the same mature checkpoint, but a better late-stage compensator.
- Relative to the previous default setup:
  - mass is slightly more robust,
  - energy drops by about `1.33`,
  - average concentration is essentially unchanged,
  - idle-threshold steps increase.

## Lowest-energy safe completion candidate

Checkpoint:
- `runs/sac_cc_conc_scout_probe_v1/checkpoints/best_model.pth`

5-seed summary:
- mean mass: `401.545 t`
- mean energy: `530.599`
- mean C_uf: `0.697345`
- mean below-idle-threshold steps: `32.2`
- unsafe episode rate: `0.0`
- dry-run steps: `0.0`

Interpretation:
- This version is slightly more energy-efficient than the baseline.
- However, it sacrifices average underflow concentration too much, so it is not a clean replacement.

## Stable but weaker stage-adapt archive

Checkpoint:
- `archives/cc_sac_stageadapt_v4_20260419/best_model.pth`

5-seed summary:
- mean mass: `401.574 t`
- mean energy: `544.476`
- mean C_uf: `0.708744`
- mean below-idle-threshold steps: `23.6`
- unsafe episode rate: `0.0`
- dry-run steps: `0.0`

Interpretation:
- Stable completion, but worse than the current baseline on both energy and average concentration.

## High-concentration / low-energy but under-target family

Representative checkpoints:
- `runs/sac_cc_ft2_conc0725_soft_ft1/checkpoints/best_model.pth`
- `runs/sac_cc_ft2_highconc_ft1/checkpoints/best_model.pth`
- `runs/sac_cc_ft2_energy032_ft1/checkpoints/best_model.pth`

3-seed coarse scan:
- mean mass: about `395 t`
- mean energy: about `528.5~528.8`
- mean C_uf: about `0.7164~0.7165`
- unsafe episode rate: `0.0`

Interpretation:
- These models prove a better concentration/energy operating region exists.
- Their problem is not safety or energy; it is the last `~5 t` of throughput.
- They are good starting points for future “补产量但不破坏高浓度骨架” finetuning.

## Finetune attempts in this round

1. `runs/sac_cc_ft2_conc0725_to400_probe_v1`
- Goal: start from the 395 t / high-C_uf candidate and add just enough target pressure to cross 400 t.
- Result: the policy drifted downward in throughput; best eval stayed around `383 t`.

2. `runs/sac_cc_baseline_highconc_microft_v1`
- Goal: micro-finetune the current baseline with a very small high-concentration preference.
- Result: the best checkpoint remained effectively the original baseline; later epochs overshot mass and reduced C_uf.

## Current conclusion

Two different operating regimes are now clearly visible:
- `baseline archive`: better completion + higher average C_uf
- `conc_scout_v1`: lower energy but noticeably lower average C_uf

For now, the recommended production CC setup is:
- checkpoint: `archives/cc_sac_ft2_20260418/checkpoint/best_model.pth`
- environment: current default `env/gym_env.py` after the late-target compensator update

Energy-first alternative:
- checkpoint: `runs/sac_cc_ft2_stageadapt_v3_warm1500_probe12/checkpoints/best_model.pth`
- 5-seed summary:
  - mean mass: `401.325 t`
  - mean energy: `525.726`
  - mean C_uf: `0.704639`
  - mean below-idle-threshold steps: `41.0`
  - unsafe episode rate: `0.0`
  - dry-run steps: `0.0`

The most promising next research direction is:
- start from the `~395 t / 0.7165 C_uf / 528.5 energy` family,
- use a very targeted completion-only finetune,
- avoid large reward changes that destroy the high-concentration operating structure.
