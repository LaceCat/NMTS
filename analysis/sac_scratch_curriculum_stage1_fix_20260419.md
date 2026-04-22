# SAC Scratch Curriculum Stage1 Fix Notes (2026-04-19)

## Goal

Repair the from-scratch CC SAC curriculum so that:

- `stage1` no longer collapses immediately into the bad deterministic startup
- the policy can enter `stage2`/`stage3` from a learnable safe regime

## Code changes applied

### 1. `algorithms/sac.py`

- Adjusted delta-mode actor prior for scratch SAC startup.
- Current delta-mode prior:
  - `Q_uf delta` bias: `action_low + action_range * 0.58`
  - `Q_fp` bias: `action_low + action_range * 0.26`

Intent:

- avoid the old `Q_uf = 0 / Q_fp = 0` high-concentration collapse
- avoid the later overly aggressive `Q_uf ramp + Q_fp≈0` buffer-overflow startup

### 2. `train.py`

- Auto-enable `--enable_sac_actor_prior` for scratch SAC curriculum runs with no checkpoint.
- Strengthened `stage1` SAC reward shaping:
  - `energy_cost_weight = 0.05`
  - `uf_conc_guidance_target = 0.70`
  - `uf_conc_guidance_band = 0.03`
  - `uf_conc_guidance_upper_soft_limit = 0.735`
  - `uf_conc_guidance_below_weight = 30.0`
  - `uf_conc_guidance_above_weight = 220.0`
  - `uf_conc_guidance_band_bonus = 6.0`
  - `uf_low_conc_flow_penalty_weight = 4000.0`

Intent:

- make `stage1` explicitly dislike "low concentration + high underflow draw"
- make scratch startup less likely to run straight into either overflow or unsafe concentration

## Experiment summary

### `sac_cc_curriculum_scratch_v2`

- command: default curriculum split (`2/2/4`)
- result:
  - `stage1` no longer died via `C_uf≈0.81`
  - but deterministic eval collapsed into **buffer overflow**
  - typical pattern:
    - `Q_uf` ramped to 50
    - `Q_fp≈0`
    - `V_buf > 30` around step ~20

### `sac_cc_curriculum_scratch_v3`

- softer `Q_uf` ramp + moderate `Q_fp`
- result:
  - `stage1` eval improved to about `300 t`
  - still unsafe, still later unstable

### `sac_cc_curriculum_scratch_v4`

- stronger stage1 low-concentration flow penalty
- result:
  - `stage1` still unsafe
  - `stage2` reverted to the old `C_uf≈0.8086` bad mode

### `sac_cc_curriculum_scratch_v5_longstage`

- longer curriculum:
  - `stage1=6`
  - `stage2=3`
  - `stage3=3`
- result:
  - significant improvement
  - first stable safe eval appeared in late epochs:
    - around `404 t`
    - `unsafe=0`
  - but `dry-run` remained very high

### `sac_cc_curriculum_scratch_v6_guard15`

- same idea as `v5`, plus:
  - `--low_buffer_fp_threshold 1.5`
- result:
  - first genuinely stable scratch training line
  - later epochs achieved:
    - `unsafe=0`
    - `dry-run=0`
    - `avg C_uf ≈ 0.724 ~ 0.727`
  - but still **under target**:
    - final evals around `350 ~ 365 t`

This is currently the most promising scratch curriculum recipe.

### `sac_cc_curriculum_scratch_v7_longtarget`

- longer `stage2`:
  - `stage1=4`
  - `stage2=10`
  - `stage3=2`
- result:
  - unstable again
  - repeatedly fell back into the old `C_uf≈0.8086` failure mode

Conclusion:

- simply extending `stage2` is not enough
- the `v6` recipe is more robust than `v7`

## Current conclusion

Best scratch-curriculum setting found in this round:

- run:
  - `runs/sac_cc_curriculum_scratch_v8_push400`
- best stage-2 checkpoint:
  - `runs/sac_cc_curriculum_scratch_v8_push400/checkpoints/best_stage2.pth`
- best observed eval:
  - `epoch 10`
  - `stage = S2-400T`
  - `mass = 402.14 t`
  - `avg C_uf = 0.7137`
  - `energy = 439.09`
  - `unsafe = 0`
  - `dry-run = 0`

This is the first scratch SAC curriculum run in this repair round that
simultaneously achieved:

- target completion
- zero unsafe episodes
- zero dry-run

### `sac_cc_curriculum_scratch_v8_push400`

- based on the `v6` recipe
- plus stronger target-shaping:
  - `--sac_target_cross_bonus 120`
  - `--sac_terminal_under_penalty_weight 20`
  - `--sac_terminal_target_band_bonus 500`
- curriculum:
  - `stage1=6`
  - `stage2=5`
  - `stage3=3`
- low-buffer FP guard:
  - `--low_buffer_fp_threshold 1.5`

Result:

- stage 2 became the sweet spot
- stage 3 still tends to trade away target completion for EEI
- therefore the **best usable checkpoint is `best_stage2.pth`**, not the final
  stage-3 snapshot

### Stage-1 upper soft-limit scan

Additional experiments after `v8`:

- `v9`:
  - `runs/sac_cc_curriculum_scratch_v9_soft80`
  - stage-1 upper soft limit raised to `0.80`
  - best usable checkpoint:
    - `best_stage2.pth`
    - `402.35 t / 414.90 energy / avg C_uf 0.7014 / unsafe=0 / dry-run=0`
  - observation:
    - target completion and safety remained good
    - average concentration dropped versus `v8`
    - energy improved

- `v10`:
  - `runs/sac_cc_curriculum_scratch_v10_soft76`
  - stage-1 upper soft limit set to `0.76`
  - observation:
    - unstable
    - repeatedly fell back into the `C_uf≈0.8086` bad mode

- `v11`:
  - `runs/sac_cc_curriculum_scratch_v11_soft77`
  - stage-1 upper soft limit set to `0.77`
  - observation:
    - also unstable
    - worse than both `v8` and `v9`

Conclusion of this scan:

- raising the stage-1 upper soft limit does **not** improve average concentration
- `0.80` can reduce energy, but at the cost of lower average concentration
- `0.76` / `0.77` are worse and should not become defaults
- default code path was therefore restored to the more conservative `0.735`
  while keeping CLI overrides for future sweeps

Recommended next step from here:

1. Keep the `v8` startup recipe fixed.
2. Treat `stage2` as the main deployment candidate for scratch SAC.
3. If needed, redesign stage 3 so it cannot pull the already-good stage-2
   policy back below the target band.
