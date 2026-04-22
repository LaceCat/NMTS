## v8 stage2 concentration-focused finetune attempt

Date: 2026-04-19

Base checkpoint:
- `runs/sac_cc_curriculum_scratch_v8_push400/checkpoints/best_stage2.pth`

New finetune run:
- `runs/sac_cc_v8_stage2_concft_v1`

Training intent:
- keep `>=400 t`
- keep `unsafe=0`
- keep `dry-run=0`
- add only mild stage-2 concentration guidance

Code changes:
- added stage-2-specific SAC guidance CLI knobs in `train.py`
- added mild default stage-2 concentration guidance for finetune curriculum:
  - target `0.72`
  - band `0.015`
  - start ratio `0.20`
  - mass gate ratio `0.55`
  - upper soft limit `0.735`
  - below weight `5.0`
  - above weight `90.0`
  - band bonus `0.20`
  - terminal average C_uf threshold `0.712`
  - terminal average C_uf bonus `1000.0`

Finetune command highlights:
- checkpoint mode: `finetune`
- epochs: `8`
- stage split: `1 / 6 / 1`
- low buffer guard threshold: `1.5`
- stage-2 override set used in run:
  - target `0.722`
  - band `0.014`
  - start ratio `0.15`
  - mass gate ratio `0.50`
  - upper soft limit `0.736`
  - below weight `7.0`
  - above weight `110.0`
  - band bonus `0.25`

Observed training behavior:
- epoch 2 (`best_stage2`) looked almost identical to the base policy
- stronger concentration drift appeared later, but it immediately traded away throughput
- stage 3 again pulled the policy away from the best stage-2 operating point

Single-run eval snapshots from training log:
- base-like stage-2 best:
  - mass `402.31 t`
  - avg C_uf `0.7138`
  - energy `439.77`
  - unsafe `0`
  - dry-run `0`
- later higher-concentration but underproducing stage-2 snapshots:
  - epoch 3: `380.76 t / 0.7215 / 401.42`
  - epoch 4: `365.49 t / 0.7271 / 460.77`
- stage-3 best-model snapshot:
  - `400.25 t / 0.7159 / 511.42`
  - but only `60%` meet rate on the 5-episode eval used during training

Five-seed reevaluation (`91~95`):

1. Base `v8` stage-2 checkpoint
- checkpoint: `runs/sac_cc_curriculum_scratch_v8_push400/checkpoints/best_stage2.pth`
- final mass: `402.7 +/- 0.2 t`
- energy cost: `439.14 +/- 1.08`
- reward: `400.94 +/- 1.35`
- in-band rate: `100%`

2. New finetuned `best_stage2`
- checkpoint: `runs/sac_cc_v8_stage2_concft_v1/checkpoints/best_stage2.pth`
- final mass: `402.7 +/- 0.2 t`
- energy cost: `439.14 +/- 1.08`
- reward: `400.94 +/- 1.35`
- in-band rate: `100%`

3. New finetuned `best_model` (stage 3)
- checkpoint: `runs/sac_cc_v8_stage2_concft_v1/checkpoints/best_model.pth`
- final mass: `399.9 +/- 0.8 t`
- energy cost: `510.77 +/- 1.57`
- reward: `-1164.61 +/- 1231.33`
- in-band rate: `40%`

Checkpoint hashes:
- base `best_stage2`: `0B6B44E80F9F5446CB76C3310D6EF49727F8425558BB80672F4BD767E9FCC40C`
- finetuned `best_stage2`: `9B3D8DAE6B4D083A641EDF8D8536BD57C08B61B16139C24C1A5992CA643B0F0D`

Interpretation:
- the finetune did change the weights
- but the deployable stage-2 policy did not improve in any meaningful way
- the best safe operating point remains the original `v8 best_stage2`
- stage 3 still tends to degrade a good stage-2 policy

Decision:
- keep `runs/sac_cc_curriculum_scratch_v8_push400/checkpoints/best_stage2.pth` as the main scratch CC baseline
- do not promote `runs/sac_cc_v8_stage2_concft_v1/checkpoints/best_model.pth`
