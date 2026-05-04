# Continuous Low-Energy SAC Repair (2026-04-29)

## Problem

The previous continuous single-layer SAC low-energy result appeared broken after the environment update because dry-run was still judged using the old positive low-buffer threshold (`0.5 m3`). The improved environment clips filter-press withdrawal by physically available inventory and only treats below-zero inventory as infeasible.

## Environment Fix

- `RewardConfig.dry_run_buffer_threshold` changed from `0.5` to `0.0`.
- SAC training stage reward configs now also use `dry_run_buffer_threshold = 0.0`.
- Buffer/mixer running logic remains:
  - run when buffer volume is `>= 2.0`;
  - run when a filter-press batch has started and the buffer is non-empty;
  - do not run on an empty buffer solely because the batch accumulator is positive.

## Selected Repaired Continuous Baseline

- Checkpoint: `archives/cc_sac_lowenergy_repaired_20260429/checkpoint/best_model.pth`
- Source run: `runs/sac_cc_stage4_bc12_energyfirst_v1`
- Evaluation command disables optional midcourse/late soft governors:

```powershell
python evaluate.py `
  --checkpoint runs\sac_cc_stage4_bc12_energyfirst_v1\checkpoints\best_model.pth `
  --algo sac --mode CC `
  --uf_control_mode delta --uf_delta_max 3.0 `
  --q_fp_delta_max 12.0 `
  --disable_post_target_fp_governor `
  --disable_post_target_idle_seeker `
  --disable_midcourse_quality_governor `
  --disable_late_concentration_keeper `
  --disable_late_target_compensator `
  --seeds 5 --seed_start 91 --device cpu
```

## Strict 5-Seed Result

- Mean final mass: `400.113 +/- 0.081 t`
- Mean energy cost: `95.988 +/- 1.722`
- Mean average `C_uf`: `0.717510`
- Unsafe episode rate: `0.0`
- Unsafe step rate: `0.0`
- Dry-run step rate: `0.0`
- Low-concentration step rate: `1.04%`
- Pass rate: `100%`

## Notes

Additional Q_fp feasibility distillation probes were run. They reduced the old positive-threshold dry-run rate but were not selected because the environment-threshold mismatch was the primary issue and the original low-energy checkpoint remains better after the semantic fix.
