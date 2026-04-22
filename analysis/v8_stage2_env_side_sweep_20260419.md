## v8 stage2 environment-side concentration sweep

Date: 2026-04-19

Fixed checkpoint:
- `runs/sac_cc_curriculum_scratch_v8_push400/checkpoints/best_stage2.pth`

Goal:
- keep the policy weights fixed
- improve average underflow concentration from the environment/post-processing side
- do not violate higher-priority constraints:
  - `unsafe = 0`
  - `dry-run = 0`
  - final mass remains safely above `400 t`

Why this route:
- previous stage-2 finetune did not beat `v8 best_stage2`
- stage 3 still tends to degrade a good stage-2 policy
- so the more promising next step is to tune the environment-side governors

Files changed:
- `env/gym_env.py`
- `evaluate.py`

What was added to `evaluate.py`:
- richer metrics:
  - `avg C_uf`
  - `unsafe ep / unsafe step`
  - `dry-run step`
  - `low-conc step`
  - `mixer idle minutes`
  - `midcourse governor / late concentration / late target` activation rates
- governor CLI parameters for direct fixed-checkpoint sweeps

Sweep records saved to:
- `__agent_debug__/env_sweep_v8_stage2_20260419/`

### Reference baseline

Old effective baseline (before this sweep):
- mass `402.655`
- energy `439.136`
- avg C_uf `0.713848`
- unsafe `0`
- dry-run `0`
- mixer idle minutes `596.4`
- midcourse governor step rate `1.0%`

### Tested configs

1. config_A
- earlier and stronger midcourse quality governor
- result:
  - mass `402.493`
  - energy `438.160`
  - avg C_uf `0.714297`
  - unsafe `0`
  - dry-run `0`

2. config_B
- milder midcourse governor with slightly stronger target recovery
- result:
  - mass `402.750`
  - energy `437.693`
  - avg C_uf `0.714060`
  - unsafe `0`
  - dry-run `0`

3. config_C
- stronger late-target help plus tighter midcourse Q_uf cap
- result:
  - mass `402.570`
  - energy `437.492`
  - avg C_uf `0.714362`
  - unsafe `0`
  - dry-run `0`

4. config_D
- slightly more aggressive midcourse concentration target
- result:
  - mass `402.701`
  - energy `437.668`
  - avg C_uf `0.714413`
  - unsafe `0`
  - dry-run `0`

### Main findings

1. Environment-side concentration improvement is real, but modest.
- all good configs increased average C_uf above the old baseline
- the gain is about `+0.0002 ~ +0.0006`

2. The improvement comes almost entirely from the midcourse quality governor.
- `late_concentration_keeper` activation stayed at `0.0%`
- `midcourse_quality_governor` activation rose from about `1.0%` to `2.5% ~ 3.2%`

3. The chosen config should respect the actual priority order.
- safety unchanged: `unsafe = 0`
- dry-run unchanged: `0`
- mass still clearly above `400 t`
- energy did not worsen; it actually improved slightly

### Second sweep around config_D

After config_D proved safe and slightly better than the original baseline, a
second sweep was run to push the midcourse governor closer to the real limit.

Key additional configs:

5. config_E
- mass `402.601`
- energy `436.763`
- avg C_uf `0.714426`
- unsafe `0`
- dry-run `0`

6. config_G
- mass `402.412`
- energy `435.710`
- avg C_uf `0.714746`
- unsafe `0`
- dry-run `0`
- first config where `late_concentration_keeper` started to appear (`0.3%`)

7. config_I
- mass `402.346`
- energy `428.190`
- avg C_uf `0.716304`
- unsafe `0`
- dry-run `0`
- in-band `100%`
- mixer idle minutes `597.2`
- midcourse governor step rate `9.7%`
- late concentration step rate `0.6%`
- late target step rate `3.6%`

Boundary checks:

8. config_H
- mass `395.774`
- energy `417.118`
- avg C_uf `0.717898`
- unsafe `0`
- dry-run `0`
- fails target completion

9. config_J
- mass `392.757`
- energy `413.948`
- avg C_uf `0.718627`
- unsafe `0`
- dry-run `0`
- fails target completion

Interpretation:
- there is still room to push concentration upward
- but the tradeoff cliff is now very clear:
  - around config_I: still safely above `400 t`
  - around config_H/J: concentration rises further, but throughput drops below target

### Final chosen default

Promoted config_I to the new default environment-side setup.

New defaults in `env/gym_env.py`:
- `midcourse_quality_start_mass = 115.0`
- `midcourse_quality_c_uf_target = 0.745`
- `midcourse_quality_q_uf_cap = 9.5`
- `midcourse_quality_q_fp_floor = 21.5`
- `midcourse_quality_buffer_min = 1.6`
- `midcourse_quality_buffer_max = 18.0`
- `late_target_window_minutes = 175`
- `late_target_mass_gap_limit = 21.0`
- `late_target_c_uf_limit = 0.746`
- `late_target_q_uf_bias_max = 6.2`
- `late_target_q_fp_bias_max = 12.8`

`evaluate.py` defaults were synchronized to the same values.

### New final default evaluation result

Using the fixed checkpoint:
- `runs/sac_cc_curriculum_scratch_v8_push400/checkpoints/best_stage2.pth`

with the new default environment-side setup:
- mass `402.3 +/- 0.3 t`
- energy `428.19 +/- 1.98`
- avg C_uf `0.7163 +/- 0.0003`
- unsafe episode rate `0.0%`
- unsafe step rate `0.0%`
- dry-run step rate `0.0%`
- in-band rate `100.0%`

Compared with the original fixed-checkpoint baseline:
- avg C_uf: `0.713848 -> 0.716304`
- energy: `439.136 -> 428.190`
- safety: unchanged (`0 unsafe`, `0 dry-run`)
- throughput: still safely above `400 t`

### Final decision

Keep:
- fixed policy checkpoint = `v8 best_stage2`
- environment/post-processing defaults = promoted `config_I`

This is the current best CC scratch deployment combination.
