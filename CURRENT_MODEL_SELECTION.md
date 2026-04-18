# Current Model Selection

This file records the currently adopted models for the thesis workflow so we do not mix old checkpoints with the latest agreed baselines.

## Continuous Model

- Official choice: `CC-TD3 with terminal governor`
- Checkpoint: [selected_epoch9_balanced.pth](/F:/毕设/claude-code/runs/cc_batch_dryrun_lowconc_probe_v2scan/checkpoints/selected_epoch9_balanced.pth)
- Environment behavior: terminal governor enabled by default in [gym_env.py](/F:/毕设/claude-code/env/gym_env.py)

### Why this one

- It satisfies the agreed priority:
  1. no `unsafe`
  2. lower energy
  3. `400 t` is a task constraint rather than the top optimization target
- It is currently the selected continuous-control model for all CC demonstrations, plots, and later thesis comparisons unless explicitly overridden.

### 5-seed evaluation summary

Source: [cc_governor_compare.json](/F:/毕设/claude-code/__agent_debug__/cc_governor_compare/cc_governor_compare.json)

- Seeds: `91-95`
- Mean final mass: `400.54 +- 0.27 t`
- Mean energy: `672.77 +- 7.17`
- `zero_unsafe_rate = 100%`
- `inband_rate = 100%`

### Representative process playback

- PNG: [cc_selected_epoch9_balanced_seed91_timeseries.png](/F:/毕设/claude-code/__agent_debug__/cc_selected_epoch9_governed_timeseries/cc_selected_epoch9_balanced_seed91_timeseries.png)
- CSV: [cc_selected_epoch9_balanced_seed91_timeseries.csv](/F:/毕设/claude-code/__agent_debug__/cc_selected_epoch9_governed_timeseries/cc_selected_epoch9_balanced_seed91_timeseries.csv)
- Summary: [cc_selected_epoch9_balanced_seed91_summary.json](/F:/毕设/claude-code/__agent_debug__/cc_selected_epoch9_governed_timeseries/cc_selected_epoch9_balanced_seed91_summary.json)

### Convenience

- [plot_model_rollout_timeseries.py](/F:/毕设/claude-code/plot_model_rollout_timeseries.py) now defaults to this checkpoint, so later CC rollout plots will use the selected continuous model unless a different checkpoint is passed manually.
