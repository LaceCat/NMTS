"""
Small automated tuning loop for the current CC-SAC baseline.

This script runs a handful of short training jobs, scores each run using the
saved epoch metrics, and optionally launches a longer follow-up job from the
best candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
DEBUG_ROOT = REPO_ROOT / "__agent_debug__"


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-tune the CC SAC baseline")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per short trial")
    parser.add_argument("--episodes_per_epoch", type=int, default=2)
    parser.add_argument("--followup_epochs", type=int, default=16, help="Extra epochs for the best candidate")
    parser.add_argument("--followup_episodes_per_epoch", type=int, default=3)
    parser.add_argument("--skip_followup", action="store_true")
    return parser.parse_args()


def _score_epoch(epoch: dict):
    prefix = "eval_" if epoch.get("eval_avg_reward") is not None else ""
    constraint_rate = float(epoch.get(f"{prefix}dry_run_step_rate", epoch.get("dry_run_step_rate", 0.0))) + float(
        epoch.get(f"{prefix}low_conc_step_rate", epoch.get("low_conc_step_rate", 0.0))
    )
    return (
        float(epoch.get(f"{prefix}unsafe_episode_rate", epoch.get("unsafe_episode_rate", 0.0))),
        float(epoch.get(f"{prefix}unsafe_step_rate", epoch.get("unsafe_step_rate", 0.0))),
        constraint_rate,
        float(epoch.get(f"{prefix}target_band_distance", epoch.get("avg_target_band_distance", 0.0))),
        float(epoch.get(f"{prefix}avg_energy", epoch.get("avg_energy", 0.0))),
        -float(epoch.get(f"{prefix}avg_reward", epoch.get("avg_reward", 0.0))),
    )


def _best_epoch_from_run(run_dir: Path):
    log_path = run_dir / "training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    epochs = payload.get("epochs", [])
    if not epochs:
        raise ValueError(f"No epoch metrics found in {log_path}")
    best_epoch = min(epochs, key=_score_epoch)
    return best_epoch


def _run_trial(run_name: str, device: str, epochs: int, episodes_per_epoch: int, overrides: dict):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--algo", "sac",
        "--mode", "CC",
        "--device", device,
        "--epochs", str(epochs),
        "--episodes_per_epoch", str(episodes_per_epoch),
        "--disable_post_target_fp_governor",
        "--eval_episodes", "1",
        "--run_name", run_name,
    ]
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])

    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Trial failed: {run_name}")
    return RUNS_DIR / run_name


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = DEBUG_ROOT / f"sac_autotune_{stamp}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "name": "a_fixed_a001",
            "alpha": 0.01,
            "lr_alpha": 0.0,
            "warmup_steps": 300,
            "uf_delta_max": 3.0,
            "sac_dry_run_penalty": 140.0,
            "sac_low_conc_penalty": 60.0,
            "sac_dry_run_flow_penalty": 70.0,
            "sac_uf_low_conc_flow_penalty": 120.0,
        },
        {
            "name": "b_fixed_a002",
            "alpha": 0.02,
            "lr_alpha": 0.0,
            "warmup_steps": 300,
            "uf_delta_max": 3.0,
            "sac_dry_run_penalty": 140.0,
            "sac_low_conc_penalty": 60.0,
            "sac_dry_run_flow_penalty": 70.0,
            "sac_uf_low_conc_flow_penalty": 120.0,
        },
        {
            "name": "c_fixed_a005",
            "alpha": 0.05,
            "lr_alpha": 0.0,
            "warmup_steps": 300,
            "uf_delta_max": 3.0,
            "sac_dry_run_penalty": 140.0,
            "sac_low_conc_penalty": 60.0,
            "sac_dry_run_flow_penalty": 70.0,
            "sac_uf_low_conc_flow_penalty": 120.0,
        },
        {
            "name": "d_fixed_a002_delta25",
            "alpha": 0.02,
            "lr_alpha": 0.0,
            "warmup_steps": 400,
            "uf_delta_max": 2.5,
            "sac_dry_run_penalty": 160.0,
            "sac_low_conc_penalty": 70.0,
            "sac_dry_run_flow_penalty": 90.0,
            "sac_uf_low_conc_flow_penalty": 140.0,
        },
        {
            "name": "e_auto_a005_ref",
            "alpha": 0.05,
            "lr_alpha": 3e-4,
            "warmup_steps": 500,
            "uf_delta_max": 3.0,
            "sac_dry_run_penalty": 140.0,
            "sac_low_conc_penalty": 60.0,
            "sac_dry_run_flow_penalty": 70.0,
            "sac_uf_low_conc_flow_penalty": 120.0,
        },
    ]

    results = []
    for candidate in candidates:
        run_name = f"sac_cc_autotune_{stamp}_{candidate['name']}"
        run_dir = _run_trial(
            run_name=run_name,
            device=args.device,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            overrides={
                key: value
                for key, value in candidate.items()
                if key != "name"
            },
        )
        best_epoch = _best_epoch_from_run(run_dir)
        record = {
            "candidate": candidate["name"],
            "run_name": run_name,
            "run_dir": str(run_dir),
            "params": candidate,
            "best_epoch": best_epoch,
            "score": _score_epoch(best_epoch),
        }
        results.append(record)

    results.sort(key=lambda item: item["score"])
    best = results[0]

    followup = None
    if not args.skip_followup:
        followup_name = f"sac_cc_autotune_{stamp}_followup_{best['candidate']}"
        followup_dir = _run_trial(
            run_name=followup_name,
            device=args.device,
            epochs=args.followup_epochs,
            episodes_per_epoch=args.followup_episodes_per_epoch,
            overrides={
                key: value
                for key, value in best["params"].items()
                if key != "name"
            },
        )
        followup = {
            "run_name": followup_name,
            "run_dir": str(followup_dir),
            "best_epoch": _best_epoch_from_run(followup_dir),
        }

    summary = {
        "timestamp": stamp,
        "device": args.device,
        "short_trials": results,
        "best_short_trial": best,
        "followup": followup,
    }
    out_path = debug_dir / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
