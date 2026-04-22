# CC SAC Improvement Summary (2026-04-19)

## What changed

This round did not produce a strictly better new checkpoint through direct SAC finetuning.
Instead, the practical improvement came from the environment-side late-target compensator.

Updated defaults in `env/gym_env.py`:
- `late_target_window_minutes: 75 -> 120`
- `late_target_mass_gap_limit: 8.0 -> 12.0`
- `late_target_q_uf_bias_max: 2.0 -> 4.0`
- `late_target_q_fp_bias_max: 5.0 -> 9.0`

Design intent:
- only intervene near the end of the episode,
- only when the policy is slightly short of target,
- avoid touching the early and middle production strategy,
- improve completion robustness without sacrificing safety.

## Main production CC setup

Checkpoint:
- `archives/cc_sac_ft2_20260418/checkpoint/best_model.pth`

5-seed result under the updated default environment:
- mean mass: `401.734 t`
- mean energy: `531.618`
- mean C_uf: `0.713716`
- mean below-idle-threshold steps: `35.4`
- unsafe episode rate: `0.0`
- dry-run steps: `0.0`

Compared with the old default environment:
- mass: `401.244 -> 401.734`
- energy: `532.943 -> 531.618`
- mean C_uf: `0.713736 -> 0.713716`
- idle-threshold steps: `34.0 -> 35.4`

Conclusion:
- same mature checkpoint,
- slightly better energy,
- slightly better idle behavior,
- same safety,
- same concentration level.

## Energy-first alternative

Checkpoint:
- `runs/sac_cc_ft2_stageadapt_v3_warm1500_probe12/checkpoints/best_model.pth`

5-seed result:
- mean mass: `401.325 t`
- mean energy: `525.726`
- mean C_uf: `0.704639`
- mean below-idle-threshold steps: `41.0`
- unsafe episode rate: `0.0`
- dry-run steps: `0.0`

Conclusion:
- strongest energy-saving candidate currently available,
- but concentration is noticeably lower than the main production CC setup.

## Artifacts

Updated current-model overview figure:
- `plots/current_model_overview_improved/sac_cc_best_model_seed91_overview.png`

Candidate scan notes:
- `analysis/cc_sac_candidate_scan_20260419.md`
