# CC-SAC Low-Energy Repaired

This archive records the restored continuous single-layer SAC baseline after adapting the environment dry-run semantics to the improved buffer/mixer logic.

## Model

- Name: `CC-SAC low-energy repaired (stage4 bc12 energy-first)`
- Source run: `runs/sac_cc_stage4_bc12_energyfirst_v1`
- Checkpoint: `checkpoint/best_model.pth`
- Control mode: `CC`
- `Q_uf`: delta control, `uf_delta_max=3.0`
- `Q_fp`: continuous control, `q_fp_delta_max=12.0`
- Post-target FP governor: off in evaluation
- Optional midcourse/late soft governors: off in the archived strict evaluation
- Dry-run threshold: `0.0`, matching the improved environment semantics where filter-press withdrawal is clipped by physical inventory rather than treated as dry-run at a positive low-buffer threshold.

## 5-Seed Evaluation

Source: `eval_result_5seeds.json`

- Seeds: `91-95`
- Mean final mass: `400.113 +/- 0.081 t`
- Mean energy cost: `95.988 +/- 1.722`
- Mean average `C_uf`: `0.717510`
- Unsafe episode rate: `0.0`
- Unsafe step rate: `0.0`
- Dry-run step rate: `0.0`
- Low-concentration step rate: `1.04%`
- Pass rate: `100%`
- In-band rate: `100%`

## Representative Rollout

- PNG: `artifacts/seed91_overview.png`
- CSV: `artifacts/seed91_overview.csv`
- Summary: `artifacts/seed91_overview_summary.json`
