"""
Training entry for the thickener dewatering RL experiments.

Supported algorithms:
- ESAC
- TD3
"""

import argparse
import atexit
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.gym_env import ThickenerDewateringEnv
from env.constants import DEFAULT_CONTROL_STEPS, DEFAULT_DECISION_INTERVAL
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from utils.metrics import compute_inband_rate, evaluate_episode
from utils.visualization import save_training_plot


def parse_args():
    parser = argparse.ArgumentParser(description="Thickener dewatering RL training")

    parser.add_argument("--algo", type=str, default="esac", choices=["esac", "sac", "td3"], help="Training algorithm")

    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass in tons")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONTROL_STEPS, help="Decision steps per episode")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL, help="Physical minutes per decision step")
    parser.add_argument("--mode", type=str, default="CC", choices=["DD", "CD", "CC"], help="Physical action mode")
    parser.add_argument(
        "--uf_control_mode",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="Underflow pump control semantics for CD/CC. DD stays discrete on/off.",
    )
    parser.add_argument(
        "--uf_delta_max",
        type=float,
        default=5.0,
        help="Maximum absolute delta for Q_uf when --uf_control_mode delta is enabled.",
    )
    parser.add_argument(
        "--disable_post_target_fp_governor",
        action="store_true",
        help="Disable the post-target filter-press governor so the policy runs without terminal post-processing.",
    )
    parser.add_argument(
        "--q_fp_delta_max",
        type=float,
        default=-1.0,
        help="Per-decision maximum change of Q_fp in CC mode. <0 disables the rate limit.",
    )
    parser.add_argument(
        "--disable_low_buffer_fp_guard",
        action="store_true",
        help="Disable the hard guard that limits or shuts Q_fp when the buffer volume is too low.",
    )
    parser.add_argument(
        "--low_buffer_fp_threshold",
        type=float,
        default=0.8,
        help="Buffer-volume threshold below which the hard low-buffer Q_fp guard is active.",
    )
    parser.add_argument(
        "--low_buffer_fp_max",
        type=float,
        default=0.0,
        help="Maximum allowed Q_fp under the hard low-buffer guard.",
    )

    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--episodes_per_epoch", type=int, default=5, help="Episodes per epoch")
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=0,
        help="Replay buffer capacity. Use algorithm default when set to 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Batch size. Use algorithm default when set to 0.",
    )
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Steps before network updates")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Epoch interval for periodic checkpoints")
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=0,
        help="Deterministic evaluation episodes after each training epoch. 0 disables epoch-end evaluation.",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="safety_mass",
        choices=["none", "safety_mass"],
        help="Training curriculum. safety_mass = stage1 safety, stage2 safe 400t, stage3 safe 400t with stronger economic pressure.",
    )
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=0,
        help="Stage 1 duration. 0 = auto (about 15%% of total epochs).",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=0,
        help="Stage 2 duration. 0 = auto (about 30%% of total epochs).",
    )

    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--lr_actor", type=float, default=0.0, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=0.0, help="Critic learning rate")
    parser.add_argument(
        "--exploration_noise",
        type=float,
        default=0.0,
        help="Exploration noise scale. <=0 uses algorithm default.",
    )
    parser.add_argument(
        "--policy_noise",
        type=float,
        default=0.0,
        help="Target policy smoothing noise scale. <=0 uses algorithm default.",
    )
    parser.add_argument(
        "--noise_clip",
        type=float,
        default=0.0,
        help="Policy smoothing noise clip. <=0 uses algorithm default.",
    )
    parser.add_argument(
        "--discrete_exploration_prob",
        type=float,
        default=-1.0,
        help="Discrete branch exploration probability for CD mode. <0 uses algorithm default.",
    )
    parser.add_argument(
        "--use_gru_encoder",
        action="store_true",
        help="Use a GRU encoder over the built-in short state history for DD/CD/CC agents.",
    )
    parser.add_argument(
        "--gru_hidden_dim",
        type=int,
        default=96,
        help="GRU hidden size when --use_gru_encoder is enabled.",
    )
    parser.add_argument("--num_actors", type=int, default=5, help="Number of actors for ESAC")
    parser.add_argument("--alpha", type=float, default=0.99, help="ESAC alpha coefficient / SAC initial alpha")
    parser.add_argument("--lr_alpha", type=float, default=-1.0, help="SAC alpha learning rate. Set to 0 to freeze alpha.")
    parser.add_argument(
        "--sac_mean_q_weight",
        type=float,
        default=0.35,
        help="Auxiliary weight that pushes SAC mean actions to have high Q values.",
    )
    parser.add_argument(
        "--sac_std_reg_weight",
        type=float,
        default=0.02,
        help="Regularization weight that discourages overly large SAC action std.",
    )
    parser.add_argument(
        "--sac_deterministic_mix_prob",
        type=float,
        default=-1.0,
        help="During SAC training, probability of executing the deterministic mean action after warmup. <0 uses the SAC default.",
    )
    parser.add_argument(
        "--sac_n_step",
        type=int,
        default=-1,
        help="n-step return horizon for SAC. <0 uses the SAC default.",
    )
    parser.add_argument(
        "--sac_per_alpha",
        type=float,
        default=-1.0,
        help="Prioritized replay alpha for SAC. <=0 disables PER and uses uniform replay.",
    )
    parser.add_argument(
        "--sac_reward_scale",
        type=float,
        default=-1.0,
        help="Reward scaling factor applied before SAC stores transitions. <0 uses the SAC default.",
    )
    parser.add_argument(
        "--enable_sac_actor_prior",
        action="store_true",
        help="Enable the handcrafted SAC actor bias prior. Disabled by default for the clean SAC baseline.",
    )
    parser.add_argument(
        "--sac_dry_run_penalty",
        type=float,
        default=-1.0,
        help="Override SAC dry-run penalty when >= 0.",
    )
    parser.add_argument(
        "--sac_low_conc_penalty",
        type=float,
        default=-1.0,
        help="Override SAC low-concentration penalty when >= 0.",
    )
    parser.add_argument(
        "--sac_dry_run_flow_penalty",
        type=float,
        default=-1.0,
        help="Override SAC dense dry-run flow penalty when >= 0.",
    )
    parser.add_argument(
        "--sac_uf_low_conc_flow_penalty",
        type=float,
        default=-1.0,
        help="Override SAC low-concentration underflow-flow penalty when >= 0.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for resuming")
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="resume",
        choices=["resume", "finetune"],
        help="resume = continue an existing run in-place; finetune = load checkpoint weights into a new run.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="runs",
        help="Root directory that stores all independent training runs",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name. If omitted, a timestamped name is generated automatically.",
    )
    parser.add_argument(
        "--force_unlock",
        action="store_true",
        help="Remove an existing run lock in the target directory before starting",
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip saving the final training plot")

    args = parser.parse_args()
    # For SAC, default to single-stage training unless the user explicitly
    # requests a curriculum. This keeps the baseline simple and matches the
    # current experimental direction.
    if args.algo.lower() == "sac" and "--curriculum" not in sys.argv:
        args.curriculum = "none"
    if args.algo.lower() == "sac" and "--alpha" not in sys.argv:
        args.alpha = 0.2
    if args.algo.lower() == "sac" and "--lr_alpha" not in sys.argv:
        args.lr_alpha = 3e-4
    if args.algo.lower() == "sac" and "--uf_control_mode" not in sys.argv:
        args.uf_control_mode = "delta"
    if args.algo.lower() == "sac" and "--q_fp_delta_max" not in sys.argv:
        args.q_fp_delta_max = 12.0
    if args.algo.lower() == "sac" and "--sac_deterministic_mix_prob" not in sys.argv:
        args.sac_deterministic_mix_prob = 0.25
    if args.algo.lower() == "sac" and "--sac_n_step" not in sys.argv:
        args.sac_n_step = 1
    if args.algo.lower() == "sac" and "--eval_episodes" not in sys.argv:
        args.eval_episodes = 1
    return args


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _sanitize_name(text: str) -> str:
    safe = []
    for char in text.strip():
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    result = "".join(safe).strip("._")
    return result or "run"


def _format_target(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _build_default_run_name(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{args.algo}_{args.mode.lower()}_target{_format_target(args.target)}"
        f"_steps{args.steps}_int{args.interval}_seed{args.seed}_{timestamp}"
    )


def _build_finetune_run_name(args, checkpoint_path: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_name(checkpoint_path.stem)
    return (
        f"{args.algo}_{args.mode.lower()}_ft_{base}"
        f"_target{_format_target(args.target)}_seed{args.seed}_{timestamp}"
    )


def _fmt_progress_bar(current: int, total: int, width: int = 30) -> str:
    current = max(0, min(current, total))
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def _fmt_timedelta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_run_paths(args):
    save_root = Path(args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    resumed = args.checkpoint is not None
    if resumed:
        checkpoint_path = Path(args.checkpoint).resolve()
        if args.checkpoint_mode == "resume" and checkpoint_path.parent.name == "checkpoints":
            run_dir = checkpoint_path.parent.parent
        elif args.checkpoint_mode == "resume":
            run_dir = checkpoint_path.parent
        else:
            run_name = _sanitize_name(args.run_name) if args.run_name else _build_finetune_run_name(args, checkpoint_path)
            run_dir = save_root / run_name
            suffix = 1
            while run_dir.exists():
                run_dir = save_root / f"{run_name}_{suffix:02d}"
                suffix += 1
    else:
        run_name = _sanitize_name(args.run_name) if args.run_name else _build_default_run_name(args)
        run_dir = save_root / run_name
        suffix = 1
        while run_dir.exists():
            run_dir = save_root / f"{run_name}_{suffix:02d}"
            suffix += 1

    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return {
        "save_root": save_root,
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "log_path": run_dir / "training_log.json",
        "config_path": run_dir / "run_config.json",
        "lock_path": run_dir / "run.lock",
        "console_log_path": logs_dir / "training.log",
        "latest_model_path": checkpoints_dir / "latest.pth",
        "best_model_path": checkpoints_dir / "best_model.pth",
        "final_model_path": checkpoints_dir / "final_model.pth",
        "metrics_final_path": logs_dir / "metrics_final.json",
        "csv_path": plots_dir / "training_metrics.csv",
        "epoch_csv_path": plots_dir / "epoch_metrics.csv",
        "resumed": resumed,
    }


def _setup_logger(console_log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"thickener_train_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [Train] %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(console_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _write_run_config(path: Path, args, paths: dict):
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "command": sys.argv,
        "algo": args.algo,
        "target": args.target,
        "steps": args.steps,
        "interval": args.interval,
        "mode": args.mode,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "seed": args.seed,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "save_root": str(paths["save_root"]),
        "run_dir": str(paths["run_dir"]),
        "resumed": paths["resumed"],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _acquire_run_lock(lock_path: Path, force_unlock: bool):
    if lock_path.exists():
        if force_unlock:
            lock_path.unlink()
        else:
            raise RuntimeError(
                f"Run directory is already locked by another process: {lock_path.parent}\n"
                f"Use a different --run_name/--save_dir, or pass --force_unlock if you are sure "
                f"the old process is gone."
            )

    payload = {
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
    }
    with lock_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    def _cleanup():
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            pass

    atexit.register(_cleanup)
    return _cleanup


def _append_log(log_path: Path, entry: dict, agent, epoch_csv_path: Path | None = None):
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"epochs": []}

    data["epochs"].append(entry)
    data["all_rewards"] = agent.all_rewards[-50000:]
    data["all_masses"] = agent.all_final_masses[-5000:]
    data["all_energy"] = agent.all_energy_costs[-5000:]

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if epoch_csv_path is not None:
        _save_epoch_history_csv(epoch_csv_path, data.get("epochs", []))


def _load_existing_log(log_path: Path, agent):
    if not log_path.exists():
        return 1, -float("inf")

    with log_path.open("r", encoding="utf-8") as f:
        log_data = json.load(f)

    epochs = log_data.get("epochs", [])
    epoch_numbers = [int(item.get("epoch", 0)) for item in epochs]
    start_epoch = max(epoch_numbers, default=0) + 1

    agent.all_rewards = log_data.get("all_rewards", [])
    agent.all_final_masses = log_data.get("all_masses", [])
    agent.all_energy_costs = log_data.get("all_energy", [])

    best_reward = -float("inf")
    best_model_key = None
    if epochs:
        best_reward = max(
            float(item.get("best_reward", item.get("avg_reward", -float("inf"))))
            for item in epochs
        )

    return start_epoch, best_reward


def _save_metrics_snapshot(path: Path, agent, summary: dict):
    payload = {
        "summary": summary,
        "episode_rewards": agent.all_rewards,
        "final_masses": agent.all_final_masses,
        "energy_costs": agent.all_energy_costs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_training_csv(path: Path, agent):
    rows = zip(
        range(1, len(agent.all_rewards) + 1),
        agent.all_rewards,
        agent.all_final_masses,
        agent.all_energy_costs,
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "final_mass", "energy_cost"])
        writer.writerows(rows)


def _save_epoch_history_csv(path: Path, epochs: list[dict]):
    fieldnames = [
        "epoch",
        "stage",
        "stage_label",
        "avg_reward",
        "avg_mass",
        "avg_energy",
        "avg_safety_violations",
        "unsafe_episode_rate",
        "unsafe_step_rate",
        "completion_rate",
        "inband_rate",
        "avg_target_band_distance",
        "final_completion_rate",
        "final_inband_rate",
        "final_target_band_distance",
        "best_reward",
        "elapsed",
        "eta_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in epochs:
            row = {name: item.get(name) for name in fieldnames}
            writer.writerow(row)


def _target_band_distance(mass: float, low: float, high: float) -> float:
    if mass < low:
        return low - mass
    if mass > high:
        return mass - high
    return 0.0


def _completion_rate(final_masses: list[float], target_mass: float) -> float:
    if not final_masses:
        return 0.0
    passed = sum(1 for mass in final_masses if mass >= target_mass)
    return float(passed / len(final_masses))


def _summarize_epoch_metrics(
    *,
    rewards: list[float],
    masses: list[float],
    energy: list[float],
    safety_counts: list[int],
    unsafe_episode_flags: list[int],
    total_steps: int,
    unsafe_steps: int,
    dry_run_steps: int,
    low_conc_steps: int,
    reward_config: RewardConfig,
    stage: dict,
):
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_mass = float(np.mean(masses)) if masses else 0.0
    avg_energy = float(np.mean(energy)) if energy else 0.0
    avg_safety_violations = float(np.mean(safety_counts)) if safety_counts else 0.0
    unsafe_episode_rate = float(np.mean(unsafe_episode_flags)) if unsafe_episode_flags else 0.0
    unsafe_step_rate = float(unsafe_steps / max(total_steps, 1))
    dry_run_step_rate = float(dry_run_steps / max(total_steps, 1))
    low_conc_step_rate = float(low_conc_steps / max(total_steps, 1))

    if stage["target_enabled"]:
        inband_rate = float(
            compute_inband_rate(masses, reward_config.target_mass_low, reward_config.target_mass_high)
        )
        completion_rate = _completion_rate(masses, reward_config.target_mass)
        avg_target_band_distance = float(
            np.mean(
                [
                    _target_band_distance(mass, reward_config.target_mass_low, reward_config.target_mass_high)
                    for mass in masses
                ]
            )
        ) if masses else 0.0
    else:
        inband_rate = 0.0
        completion_rate = 0.0
        avg_target_band_distance = 0.0

    return {
        "avg_reward": avg_reward,
        "avg_mass": avg_mass,
        "avg_energy": avg_energy,
        "avg_safety_violations": avg_safety_violations,
        "unsafe_episode_rate": unsafe_episode_rate,
        "unsafe_step_rate": unsafe_step_rate,
        "dry_run_step_rate": dry_run_step_rate,
        "low_conc_step_rate": low_conc_step_rate,
        "completion_rate": completion_rate,
        "inband_rate": inband_rate,
        "avg_target_band_distance": avg_target_band_distance,
    }


def _evaluate_agent_policy(args, agent, stage: dict, episodes: int, seed_base: int):
    if episodes <= 0:
        return None

    eval_reward_config = _build_reward_config_for_stage(args, stage)
    eval_env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        mode=args.mode,
        uf_control_mode=args.uf_control_mode,
        uf_delta_max=args.uf_delta_max,
        q_fp_delta_max=(None if args.q_fp_delta_max < 0 else args.q_fp_delta_max),
        pricing=PricingPresets.daily_24h(),
        reward_config=eval_reward_config,
        enable_post_target_fp_governor=not args.disable_post_target_fp_governor,
        enable_low_buffer_fp_guard=not args.disable_low_buffer_fp_guard,
        low_buffer_fp_threshold=args.low_buffer_fp_threshold,
        low_buffer_fp_max=args.low_buffer_fp_max,
    )

    rewards = []
    masses = []
    energy = []
    safety_counts = []
    unsafe_episode_flags = []
    total_steps = 0
    unsafe_steps = 0
    dry_run_steps = 0
    low_conc_steps = 0

    for episode_idx in range(int(episodes)):
        state, _ = eval_env.reset(seed=int(seed_base + episode_idx))
        if args.algo == "sac":
            state = _transform_obs_for_sac(state, args)

        done = False
        episode_reward = 0.0
        episode_info_list = []
        episode_safety_violations = 0
        episode_unsafe_steps = 0
        episode_dry_run_steps = 0
        episode_low_conc_steps = 0
        episode_steps = 0

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            if args.algo == "sac":
                next_state = _transform_obs_for_sac(next_state, args)
            done = terminated or truncated

            episode_reward += float(reward)
            episode_steps += 1
            step_safety_violations = int(info.get("safety_violations", 0))
            episode_safety_violations += step_safety_violations
            if bool(info.get("safety_violation", False)) or step_safety_violations > 0:
                episode_unsafe_steps += 1
            if bool(info.get("dry_run_violation", False)):
                episode_dry_run_steps += 1
            if bool(info.get("low_conc_violation", False)):
                episode_low_conc_steps += 1

            episode_info_list.append(info)
            state = next_state

        episode_metrics = evaluate_episode(
            episode_info_list,
            eval_reward_config.target_mass_low,
            eval_reward_config.target_mass_high,
        )
        rewards.append(float(episode_reward))
        masses.append(float(episode_metrics.get("final_mass", 0.0)))
        energy.append(float(episode_metrics.get("energy_cost", 0.0)))
        safety_counts.append(int(episode_metrics.get("safety_violations", 0)))
        unsafe_episode_flags.append(1 if int(episode_metrics.get("safety_violations", 0)) > 0 else 0)
        total_steps += max(episode_steps, 0)
        unsafe_steps += episode_unsafe_steps
        dry_run_steps += episode_dry_run_steps
        low_conc_steps += episode_low_conc_steps

    return _summarize_epoch_metrics(
        rewards=rewards,
        masses=masses,
        energy=energy,
        safety_counts=safety_counts,
        unsafe_episode_flags=unsafe_episode_flags,
        total_steps=total_steps,
        unsafe_steps=unsafe_steps,
        dry_run_steps=dry_run_steps,
        low_conc_steps=low_conc_steps,
        reward_config=eval_reward_config,
        stage=stage,
    )


def _best_model_key(
    *,
    unsafe_step_rate: float,
    unsafe_episode_rate: float,
    inband_rate: float,
    avg_target_band_distance: float,
    avg_reward: float,
):
    return (
        float(unsafe_step_rate),
        float(unsafe_episode_rate),
        -float(inband_rate),
        float(avg_target_band_distance),
        -float(avg_reward),
    )


def _resolve_curriculum_lengths(args):
    if args.curriculum == "none":
        return 0, 0

    stage1_epochs = int(args.stage1_epochs)
    stage2_epochs = int(args.stage2_epochs)

    if stage1_epochs <= 0:
        stage1_epochs = max(2, int(round(args.epochs * 0.15)))
    if stage2_epochs <= 0:
        stage2_epochs = max(2, int(round(args.epochs * 0.30)))

    if stage1_epochs + stage2_epochs >= args.epochs:
        overflow = stage1_epochs + stage2_epochs - max(args.epochs - 1, 1)
        stage2_epochs = max(1, stage2_epochs - overflow)

    return stage1_epochs, stage2_epochs


def _select_curriculum_stage(epoch: int, args):
    stage1_epochs, stage2_epochs = _resolve_curriculum_lengths(args)
    if args.curriculum == "none":
        return {
            "name": "final",
            "label": "FINAL-EEI",
            "index": 3,
            "target_enabled": True,
            "target_mass": float(args.target),
            "band_tolerance": 20.0,
        }

    if epoch <= stage1_epochs:
        return {
            "name": "stage1",
            "label": "S1-SAFE",
            "index": 1,
            "target_enabled": False,
            "target_mass": float(args.target),
            "band_tolerance": 15.0,
        }

    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "name": "stage2",
            "label": "S2-400T",
            "index": 2,
            "target_enabled": True,
            "target_mass": float(args.target),
            "band_tolerance": 20.0,
        }

    return {
        "name": "stage3",
        "label": "S3-EEI",
        "index": 3,
        "target_enabled": True,
        "target_mass": float(args.target),
        "band_tolerance": 20.0,
    }


def _transform_obs_for_sac(obs: np.ndarray, args) -> np.ndarray:
    arr = np.array(obs, dtype=np.float32, copy=True)
    if arr.shape[0] < 42:
        return arr

    raw_c_uf = float(arr[0])
    raw_m_fp = float(arr[3])

    def _transform_state_block(offset: int):
        # [C_uf, V_buf, C_aver, M_FP, Mass_buf, price, remaining_steps, Qf, Cf]
        arr[offset + 1] = arr[offset + 1] / 30.0
        arr[offset + 3] = np.clip((float(args.target) - arr[offset + 3]) / max(float(args.target), 1e-6), -2.0, 2.0)
        arr[offset + 4] = np.clip(arr[offset + 4] / max(float(args.target), 1e-6), 0.0, 2.0)
        arr[offset + 6] = np.clip(arr[offset + 6] / max(float(args.steps), 1.0), 0.0, 1.0)

    _transform_state_block(0)

    # action history: 3 x (Q_uf, Q_fp)
    for idx in range(9, 15, 2):
        arr[idx] = arr[idx] / 50.0
        arr[idx + 1] = arr[idx + 1] / 70.0

    for block_start in (15, 24, 33):
        _transform_state_block(block_start)

    derived = np.array(
        [
            np.clip((float(args.target) - raw_m_fp) / max(float(args.target), 1e-6), -2.0, 2.0),
            1.0 if raw_m_fp >= float(args.target) else 0.0,
            raw_c_uf - 0.66,
            0.75 - raw_c_uf,
        ],
        dtype=np.float32,
    )
    return np.concatenate([arr, derived], dtype=np.float32)


def _build_reward_config_for_stage(args, stage: dict) -> RewardConfig:
    config = RewardConfig(target_mass=float(stage["target_mass"]), max_steps=args.steps)
    config.enable_target_objective = bool(stage["target_enabled"])
    config.target_band_tolerance = float(stage["band_tolerance"])
    is_sac = getattr(args, "algo", "").lower() == "sac"

    if stage["name"] == "stage1":
        config.enable_target_objective = False
        config.throughput_reward_weight = 0.00 if is_sac else 0.10
        config.post_target_delta_penalty_weight = 0.0
        config.pre_target_glide_margin = 180.0
        config.pre_target_glide_scale = 0.05
        config.energy_cost_weight = 0.00 if is_sac else 0.20
        if is_sac:
            # SAC simplified reward: progress + constraints.
            # Stage 1 progress is defined as reducing the underflow
            # concentration deficit relative to the soft lower bound.
            config.reward_mode = "progress_constraints"
            config.concentration_progress_weight = 200.0
            config.uf_conc_soft_low_limit = 0.66
            config.uf_conc_low_penalty = 40.0
            config.dry_run_buffer_threshold = 0.5
            config.dry_run_penalty = 80.0
            config.dry_run_flow_penalty_weight = 40.0
            config.uf_low_conc_flow_penalty_weight = 60.0
            config.constraint_step_reward_block = True
        config.uf_conc_penalty = 400.0
        config.buffer_vol_penalty = 400.0
        config.safety_violation_penalty = 1000.0
        config.terminal_safety_block_penalty = 1200.0
        config.reward_clip_min = -5000.0
        config.reward_clip_max = 2500.0
    elif stage["name"] == "stage2":
        config.throughput_reward_weight = 0.10 if is_sac else 0.70
        config.post_target_delta_penalty_weight = 4.0 if is_sac else 7.0
        config.pre_target_glide_margin = 220.0
        config.pre_target_glide_scale = 0.03
        config.energy_cost_weight = 0.20 if is_sac else 1.00
        if is_sac:
            config.reward_mode = "progress_constraints"
            # Stage 2 progress is mass-gap reduction, with all other items
            # treated as constraints rather than parallel optimization terms.
            # Also keep a concentration-progress term so the one-stage SAC does
            # not regard chronic low C_uf as acceptable as long as mass grows.
            config.throughput_reward_weight = 12.0
            config.concentration_progress_weight = 120.0
            config.post_target_delta_penalty_weight = 8.0
            config.energy_cost_weight = 0.10
            config.uf_conc_soft_low_limit = 0.66
            config.uf_conc_low_penalty = 45.0
            config.dry_run_buffer_threshold = 0.5
            config.dry_run_penalty = 100.0
            config.dry_run_flow_penalty_weight = 50.0
            config.uf_low_conc_flow_penalty_weight = 80.0
            config.constraint_step_reward_block = True
        config.uf_conc_penalty = 350.0
        config.buffer_vol_penalty = 350.0
        config.safety_violation_penalty = 1000.0
        config.terminal_safety_block_penalty = 1200.0
        config.terminal_target_band_bonus = 350.0 if is_sac else 2400.0
        config.terminal_inband_over_penalty_weight = 1.0 if is_sac else config.terminal_inband_over_penalty_weight
        config.terminal_under_penalty_weight = 8.0 if is_sac else 14.0
        config.terminal_under_penalty_quadratic = 0.0
        config.terminal_over_penalty_weight = 3.0 if is_sac else 10.0
        config.terminal_over_penalty_quadratic = 0.0
        config.reward_clip_min = -5000.0
        config.reward_clip_max = 2500.0
    else:
        # Stage 3 / final:
        # safety first, then make sure we finish the day at/above 400 t,
        # then minimize energy. Exact 400-t landing is only a weak preference.
        config.throughput_reward_weight = 0.01 if is_sac else 0.02
        config.post_target_delta_penalty_weight = 8.0 if is_sac else 7.0
        config.pre_target_glide_margin = 0.0
        config.pre_target_glide_scale = 0.0
        config.energy_cost_weight = 1.00 if is_sac else 2.10
        config.fp_usage_weight = 0.35 if is_sac else 0.80
        config.post_target_fp_usage_weight = 2.50 if is_sac else 8.00
        config.post_target_buffer_guard_level = 18.0
        config.post_target_buffer_hold_weight = 8.0 if is_sac else 25.0
        config.schedule_tolerance_ratio = 0.08
        config.schedule_behind_weight = 0.0
        config.schedule_ahead_weight = 0.60 if is_sac else 3.50
        config.uf_conc_soft_low_limit = 0.66
        config.uf_conc_low_penalty = 30.0 if is_sac else 220.0
        if is_sac:
            config.reward_mode = "progress_constraints"
            config.throughput_reward_weight = 10.0
            config.concentration_progress_weight = 100.0
            config.post_target_delta_penalty_weight = 12.0
            config.energy_cost_weight = 0.30
            config.uf_conc_low_penalty = 60.0
            config.dry_run_flow_penalty_weight = 70.0
            config.uf_low_conc_flow_penalty_weight = 120.0
            config.constraint_step_reward_block = True
        config.dry_run_buffer_threshold = 0.5
        config.dry_run_penalty = 140.0 if is_sac else 200.0
        config.uf_conc_penalty = 450.0
        config.buffer_vol_penalty = 450.0
        config.safety_violation_penalty = 1000.0
        config.terminal_safety_block_penalty = 1800.0
        config.terminal_target_band_bonus = 200.0 if is_sac else 2200.0
        config.terminal_inband_over_penalty_weight = 1.5 if is_sac else 4.0
        config.terminal_under_penalty_weight = 15.0 if is_sac else 30.0
        config.terminal_under_penalty_quadratic = 0.0
        config.terminal_over_penalty_weight = 12.0 if is_sac else 35.0
        config.terminal_over_penalty_quadratic = 0.0
        config.reward_clip_min = -5000.0
        config.reward_clip_max = 2800.0

    if is_sac:
        if args.sac_dry_run_penalty >= 0.0:
            config.dry_run_penalty = float(args.sac_dry_run_penalty)
        if args.sac_low_conc_penalty >= 0.0:
            config.uf_conc_low_penalty = float(args.sac_low_conc_penalty)
        if args.sac_dry_run_flow_penalty >= 0.0:
            config.dry_run_flow_penalty_weight = float(args.sac_dry_run_flow_penalty)
        if args.sac_uf_low_conc_flow_penalty >= 0.0:
            config.uf_low_conc_flow_penalty_weight = float(args.sac_uf_low_conc_flow_penalty)

    return config


def _best_model_key_for_stage(
    *,
    stage: dict,
    unsafe_step_rate: float,
    unsafe_episode_rate: float,
    dry_run_step_rate: float,
    low_conc_step_rate: float,
    avg_safety_violations: float,
    avg_energy: float,
    completion_rate: float,
    inband_rate: float,
    avg_target_band_distance: float,
    avg_reward: float,
):
    medium_constraint_rate = float(dry_run_step_rate + low_conc_step_rate)

    if not stage["target_enabled"]:
        return (
            float(unsafe_step_rate),
            float(unsafe_episode_rate),
            medium_constraint_rate,
            float(avg_safety_violations),
            float(avg_energy),
            -float(avg_reward),
        )

    if int(stage.get("index", 0)) >= 3:
        return (
            float(unsafe_step_rate),
            float(unsafe_episode_rate),
            medium_constraint_rate,
            -float(completion_rate),
            float(avg_energy),
            float(avg_target_band_distance),
            -float(avg_reward),
        )

    return (
        float(unsafe_step_rate),
        float(unsafe_episode_rate),
        medium_constraint_rate,
        -float(completion_rate),
        float(avg_target_band_distance),
        float(avg_energy),
        -float(avg_reward),
    )


def _scale_optimizer_lr(optimizer, scale: float, min_lr: float = 1e-6):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * float(scale), min_lr)


def _reset_optimizer_state(optimizer):
    if optimizer is None:
        return
    optimizer.state.clear()


def _apply_finetune_stabilization(agent):
    updates = {}

    if hasattr(agent, "actor_optimizer"):
        _reset_optimizer_state(agent.actor_optimizer)
        _scale_optimizer_lr(agent.actor_optimizer, 0.05)
        updates["actor_lr_scale"] = 0.05
        updates["actor_optimizer_reset"] = True
    elif hasattr(agent, "actor_optimizers"):
        for optimizer in agent.actor_optimizers:
            _reset_optimizer_state(optimizer)
            _scale_optimizer_lr(optimizer, 0.05)
        updates["actor_lr_scale"] = 0.05
        updates["actor_optimizer_reset"] = True

    if hasattr(agent, "critic1_optimizer"):
        _reset_optimizer_state(agent.critic1_optimizer)
        _scale_optimizer_lr(agent.critic1_optimizer, 0.05)
        updates["critic_lr_scale"] = 0.05
        updates["critic_optimizer_reset"] = True
    if hasattr(agent, "critic2_optimizer"):
        _reset_optimizer_state(agent.critic2_optimizer)
        _scale_optimizer_lr(agent.critic2_optimizer, 0.05)
        updates["critic_lr_scale"] = 0.05
        updates["critic_optimizer_reset"] = True
    if hasattr(agent, "alpha_optimizer"):
        _reset_optimizer_state(agent.alpha_optimizer)
        _scale_optimizer_lr(agent.alpha_optimizer, 0.05)
        updates["alpha_lr_scale"] = 0.05
        updates["alpha_optimizer_reset"] = True

    if hasattr(agent, "action_range_np"):
        action_range = np.asarray(agent.action_range_np, dtype=np.float32)
    elif hasattr(agent, "q_uf_high") and hasattr(agent, "q_uf_low"):
        action_range = np.asarray([float(agent.q_uf_high) - float(agent.q_uf_low), 70.0], dtype=np.float32)
    else:
        action_range = None

    if action_range is not None and hasattr(agent, "exploration_noise_np"):
        quiet_exploration = np.maximum(action_range * 0.005, np.asarray([0.10, 0.14], dtype=np.float32)[: len(action_range)])
        agent.exploration_noise_np = np.minimum(np.asarray(agent.exploration_noise_np, dtype=np.float32), quiet_exploration)
        updates["exploration_noise"] = agent.exploration_noise_np.tolist()

    if action_range is not None and hasattr(agent, "policy_noise_np") and hasattr(agent, "policy_noise"):
        quiet_policy = np.maximum(action_range * 0.0025, np.asarray([0.05, 0.07], dtype=np.float32)[: len(action_range)])
        agent.policy_noise_np = np.minimum(np.asarray(agent.policy_noise_np, dtype=np.float32), quiet_policy)
        agent.policy_noise = torch.tensor(agent.policy_noise_np, dtype=torch.float32, device=agent.device)
        updates["policy_noise"] = agent.policy_noise_np.tolist()

    if action_range is not None and hasattr(agent, "noise_clip_np") and hasattr(agent, "noise_clip"):
        quiet_clip = np.maximum(action_range * 0.005, np.asarray([0.10, 0.14], dtype=np.float32)[: len(action_range)])
        agent.noise_clip_np = np.minimum(np.asarray(agent.noise_clip_np, dtype=np.float32), quiet_clip)
        agent.noise_clip = torch.tensor(agent.noise_clip_np, dtype=torch.float32, device=agent.device)
        updates["noise_clip"] = agent.noise_clip_np.tolist()

    if hasattr(agent, "discrete_exploration_prob"):
        agent.discrete_exploration_prob = min(float(agent.discrete_exploration_prob), 0.02)
        updates["discrete_exploration_prob"] = float(agent.discrete_exploration_prob)

    if hasattr(agent, "policy_freq"):
        agent.policy_freq = max(int(agent.policy_freq), 20)
        updates["policy_freq"] = int(agent.policy_freq)

    return updates


def _apply_stage_transition_tuning(agent, stage: dict):
    if stage["name"] != "stage3":
        return {}

    updates = {}

    if hasattr(agent, "actor_optimizer"):
        _scale_optimizer_lr(agent.actor_optimizer, 0.35)
        updates["actor_lr_scale"] = 0.35
    elif hasattr(agent, "actor_optimizers"):
        for optimizer in agent.actor_optimizers:
            _scale_optimizer_lr(optimizer, 0.35)
        updates["actor_lr_scale"] = 0.35

    if hasattr(agent, "critic1_optimizer"):
        _scale_optimizer_lr(agent.critic1_optimizer, 0.5)
        updates["critic_lr_scale"] = 0.5
    if hasattr(agent, "critic2_optimizer"):
        _scale_optimizer_lr(agent.critic2_optimizer, 0.5)
        updates["critic_lr_scale"] = 0.5

    if hasattr(agent, "exploration_noise_np"):
        agent.exploration_noise_np = agent.exploration_noise_np * 0.35
        updates["exploration_noise_scale"] = 0.35

    if hasattr(agent, "policy_noise_np") and hasattr(agent, "policy_noise"):
        agent.policy_noise_np = agent.policy_noise_np * 0.35
        agent.policy_noise = torch.tensor(agent.policy_noise_np, dtype=torch.float32, device=agent.device)
        updates["policy_noise_scale"] = 0.35

    if hasattr(agent, "noise_clip_np") and hasattr(agent, "noise_clip"):
        agent.noise_clip_np = agent.noise_clip_np * 0.35
        agent.noise_clip = torch.tensor(agent.noise_clip_np, dtype=torch.float32, device=agent.device)
        updates["noise_clip_scale"] = 0.35

    return updates


def _decay_agent_exploration(agent, factor: float = 0.93):
    updates = {}

    if hasattr(agent, "exploration_noise_np"):
        if hasattr(agent, "action_range_np"):
            base = np.asarray(agent.exploration_noise_np, dtype=np.float32)
            floor = np.minimum(base * 0.25, np.asarray(agent.action_range_np, dtype=np.float32) * 0.02)
        elif hasattr(agent, "q_uf_high") and hasattr(agent, "q_uf_low"):
            base = np.asarray(agent.exploration_noise_np, dtype=np.float32)
            floor = np.minimum(
                base * 0.25,
                np.asarray([float(agent.q_uf_high) - float(agent.q_uf_low)], dtype=np.float32) * 0.02,
            )
        else:
            floor = np.asarray(agent.exploration_noise_np, dtype=np.float32) * 0.25

        new_noise = np.maximum(np.asarray(agent.exploration_noise_np, dtype=np.float32) * factor, floor)
        agent.exploration_noise_np = new_noise.astype(np.float32)
        updates["exploration_noise"] = agent.exploration_noise_np.tolist()

    if hasattr(agent, "policy_noise_np") and hasattr(agent, "policy_noise"):
        base = np.asarray(agent.policy_noise_np, dtype=np.float32)
        floor = np.maximum(base * 0.25, 1e-3)
        agent.policy_noise_np = np.maximum(base * factor, floor).astype(np.float32)
        agent.policy_noise = torch.tensor(agent.policy_noise_np, dtype=torch.float32, device=agent.device)
        updates["policy_noise"] = agent.policy_noise_np.tolist()

    if hasattr(agent, "noise_clip_np") and hasattr(agent, "noise_clip"):
        base = np.asarray(agent.noise_clip_np, dtype=np.float32)
        floor = np.maximum(base * 0.25, 1e-3)
        agent.noise_clip_np = np.maximum(base * factor, floor).astype(np.float32)
        agent.noise_clip = torch.tensor(agent.noise_clip_np, dtype=torch.float32, device=agent.device)
        updates["noise_clip"] = agent.noise_clip_np.tolist()

    if hasattr(agent, "discrete_exploration_prob"):
        agent.discrete_exploration_prob = max(float(agent.discrete_exploration_prob) * factor, 0.02)
        updates["discrete_exploration_prob"] = float(agent.discrete_exploration_prob)

    return updates


def _build_agent(args, state_dim: int, action_dim: int, action_low=None, action_high=None):
    if args.algo == "esac":
        if args.mode != "CC":
            raise ValueError("ESAC currently only supports true CC mode. Use TD3-family baselines for DD/CD.")
        from algorithms.esac import ESACAgent

        buffer_cap = args.buffer_capacity if args.buffer_capacity > 0 else 4000
        batch = args.batch_size if args.batch_size > 0 else 128
        lr_a = args.lr_actor if args.lr_actor > 0 else 3e-4
        lr_c = args.lr_critic if args.lr_critic > 0 else 3e-3

        agent = ESACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_capacity=buffer_cap,
            batch_size=batch,
            hidden_dim=args.hidden_dim,
            lr_actor=lr_a,
            lr_critic=lr_c,
            alpha=args.alpha,
            num_actors=args.num_actors,
            device=args.device,
        )
        algo_name = f"ESAC (N={args.num_actors})"
    elif args.algo == "sac":
        if args.mode != "CC":
            raise ValueError("SAC currently only supports true CC mode.")
        from algorithms.sac import SACAgent

        if state_dim >= 42:
            state_dim = state_dim + 4

        buffer_cap = args.buffer_capacity if args.buffer_capacity > 0 else 1_000_000
        batch = args.batch_size if args.batch_size > 0 else 256
        lr_a = args.lr_actor if args.lr_actor > 0 else 3e-4
        lr_c = args.lr_critic if args.lr_critic > 0 else 3e-4
        lr_alpha = args.lr_alpha if args.lr_alpha >= 0 else 3e-4

        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_capacity=buffer_cap,
            batch_size=batch,
            hidden_dim=args.hidden_dim,
            lr_actor=lr_a,
            lr_critic=lr_c,
            lr_alpha=lr_alpha,
            init_alpha=args.alpha,
            action_low=np.asarray(action_low, dtype=np.float32) if action_low is not None else None,
            action_high=np.asarray(action_high, dtype=np.float32) if action_high is not None else None,
            use_gru_encoder=args.use_gru_encoder,
            gru_hidden_dim=args.gru_hidden_dim,
            mean_action_q_weight=args.sac_mean_q_weight,
            std_reg_weight=args.sac_std_reg_weight,
            n_step=(args.sac_n_step if args.sac_n_step > 0 else 1),
            per_alpha=(args.sac_per_alpha if args.sac_per_alpha >= 0 else 0.0),
            reward_scale=(args.sac_reward_scale if args.sac_reward_scale >= 0 else 0.01),
            use_actor_prior=args.enable_sac_actor_prior,
            device=args.device,
        )
        algo_name = "GRU-SAC (CC)" if args.use_gru_encoder else "SAC (CC)"
    else:
        buffer_cap = args.buffer_capacity if args.buffer_capacity > 0 else 1_000_000
        batch = args.batch_size if args.batch_size > 0 else 256
        lr_a = args.lr_actor if args.lr_actor > 0 else 1e-4
        lr_c = args.lr_critic if args.lr_critic > 0 else 1e-3
        exploration_noise = args.exploration_noise if args.exploration_noise > 0 else None
        policy_noise = args.policy_noise if args.policy_noise > 0 else None
        noise_clip = args.noise_clip if args.noise_clip > 0 else None

        if args.mode == "DD":
            from algorithms.discrete_ddqn import DiscreteDDQNAgent

            agent = DiscreteDDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                buffer_capacity=buffer_cap,
                batch_size=batch,
                hidden_dim=args.hidden_dim,
                lr_actor=lr_c,
                lr_critic=lr_c,
                use_gru_encoder=args.use_gru_encoder,
                gru_hidden_dim=args.gru_hidden_dim,
                device=args.device,
            )
            algo_name = "GRU-Discrete DDQN (DD)" if args.use_gru_encoder else "Discrete DDQN (DD)"
        elif args.mode == "CD":
            from algorithms.hybrid_td3 import HybridTD3Agent

            hybrid_kwargs = {}
            if exploration_noise is not None:
                hybrid_kwargs["exploration_noise"] = exploration_noise
            if policy_noise is not None:
                hybrid_kwargs["policy_noise"] = policy_noise
            if noise_clip is not None:
                hybrid_kwargs["noise_clip"] = noise_clip
            if args.discrete_exploration_prob >= 0:
                hybrid_kwargs["discrete_exploration_prob"] = args.discrete_exploration_prob
            if action_low is not None:
                hybrid_kwargs["q_uf_low"] = float(np.asarray(action_low, dtype=np.float32)[0])
            if action_high is not None:
                hybrid_kwargs["q_uf_high"] = float(np.asarray(action_high, dtype=np.float32)[0])

            agent = HybridTD3Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                buffer_capacity=buffer_cap,
                batch_size=batch,
                hidden_dim=args.hidden_dim,
                lr_actor=lr_a,
                lr_critic=lr_c,
                use_gru_encoder=args.use_gru_encoder,
                gru_hidden_dim=args.gru_hidden_dim,
                device=args.device,
                **hybrid_kwargs,
            )
            algo_name = "GRU-Hybrid TD3 (CD)" if args.use_gru_encoder else "Hybrid TD3 (CD)"
        else:
            from algorithms.td3 import TD3Agent

            td3_kwargs = {}
            if exploration_noise is not None:
                td3_kwargs["exploration_noise"] = exploration_noise
            if policy_noise is not None:
                td3_kwargs["policy_noise"] = policy_noise
            if noise_clip is not None:
                td3_kwargs["noise_clip"] = noise_clip
            if action_low is not None:
                td3_kwargs["action_low"] = np.asarray(action_low, dtype=np.float32)
            if action_high is not None:
                td3_kwargs["action_high"] = np.asarray(action_high, dtype=np.float32)

            agent = TD3Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                buffer_capacity=buffer_cap,
                batch_size=batch,
                hidden_dim=args.hidden_dim,
                lr_actor=lr_a,
                lr_critic=lr_c,
                use_gru_encoder=args.use_gru_encoder,
                gru_hidden_dim=args.gru_hidden_dim,
                device=args.device,
                **td3_kwargs,
            )
            algo_name = "GRU-TD3 (CC)" if args.use_gru_encoder else "TD3 (CC)"

    return agent, algo_name


def train():
    args = parse_args()
    set_seed(args.seed)

    paths = _resolve_run_paths(args)
    logger = _setup_logger(paths["console_log_path"])
    release_lock = _acquire_run_lock(paths["lock_path"], args.force_unlock)
    _write_run_config(paths["config_path"], args, paths)

    final_reward_config = RewardConfig(target_mass=args.target, max_steps=args.steps)
    current_stage = _select_curriculum_stage(1, args)
    reward_config = _build_reward_config_for_stage(args, current_stage)
    env = ThickenerDewateringEnv(
        max_steps=args.steps,
        decision_interval=args.interval,
        target_mass=args.target,
        mode=args.mode,
        uf_control_mode=args.uf_control_mode,
        uf_delta_max=args.uf_delta_max,
        q_fp_delta_max=(None if args.q_fp_delta_max < 0 else args.q_fp_delta_max),
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
        enable_post_target_fp_governor=not args.disable_post_target_fp_governor,
        enable_low_buffer_fp_guard=not args.disable_low_buffer_fp_guard,
        low_buffer_fp_threshold=args.low_buffer_fp_threshold,
        low_buffer_fp_max=args.low_buffer_fp_max,
    )

    agent, algo_name = _build_agent(
        args,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    stage1_epochs, stage2_epochs = _resolve_curriculum_lengths(args)

    start_epoch = 1
    best_reward = -float("inf")
    best_model_key = None
    finetune_updates = {}
    if args.checkpoint:
        agent.load(args.checkpoint)
        if args.checkpoint_mode == "resume":
            start_epoch, best_reward = _load_existing_log(paths["log_path"], agent)
        else:
            finetune_updates = _apply_finetune_stabilization(agent)

    logger.info("=" * 72)
    logger.info(f"Start {algo_name} training")
    logger.info(f"Target dry mass: {args.target} t")
    logger.info(f"Decision steps: {args.steps}")
    logger.info(f"Decision interval: {args.interval} min")
    logger.info(f"Action mode: {args.mode}")
    logger.info(
        f"Post-target FP governor: {'off' if args.disable_post_target_fp_governor else 'on'}"
    )
    logger.info(
        f"Low-buffer FP guard: {'off' if args.disable_low_buffer_fp_guard else 'on'}"
    )
    if args.mode != "DD":
        logger.info(f"Q_uf control: {args.uf_control_mode}")
        if args.uf_control_mode == "delta":
            logger.info(f"Q_uf delta max: ±{args.uf_delta_max}")
        if args.q_fp_delta_max >= 0:
            logger.info(f"Q_fp delta max: ±{args.q_fp_delta_max}")
        if not args.disable_low_buffer_fp_guard:
            logger.info(
                f"Low-buffer guard threshold/max: {args.low_buffer_fp_threshold} m3 / {args.low_buffer_fp_max} m3/h"
            )
    logger.info(f"Target band (descriptive only): {reward_config.target_mass_low:.1f} ~ {reward_config.target_mass_high:.1f} t")
    if args.curriculum == "none":
        logger.info("Curriculum: none")
    else:
        logger.info(
            f"Curriculum: safety_mass | stage1={stage1_epochs} ep (safe only) | "
            f"stage2={stage2_epochs} ep (safe 400t) | stage3={max(args.epochs - stage1_epochs - stage2_epochs, 0)} ep (safe 400t + low EEI)"
        )
    logger.info(f"Total physical time: {args.steps * args.interval} min = {args.steps * args.interval / 60:.1f} h")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Checkpoint mode: {args.checkpoint_mode}")
    if args.algo in ("td3", "sac"):
        logger.info(f"GRU encoder: {'on' if args.use_gru_encoder else 'off'}")
        if args.use_gru_encoder:
            logger.info(f"GRU hidden dim: {args.gru_hidden_dim}")
    if finetune_updates:
        logger.info(
            "Finetune stabilization | "
            + ", ".join(f"{k}={v}" for k, v in finetune_updates.items())
        )
    logger.info(f"Run root: {paths['save_root']}")
    logger.info(f"Run dir: {paths['run_dir']}")
    if args.algo == "esac":
        logger.info(f"ESAC alpha: {args.alpha}")
        logger.info(f"ESAC actors: {args.num_actors}")
    elif args.algo == "sac":
        logger.info(f"SAC init alpha: {args.alpha}")
        logger.info(f"SAC alpha lr: {args.lr_alpha if args.lr_alpha >= 0 else 3e-4}")
        logger.info(f"SAC mean-Q weight: {args.sac_mean_q_weight}")
        logger.info(f"SAC std-reg weight: {args.sac_std_reg_weight}")
        logger.info(f"SAC deterministic-mix prob: {args.sac_deterministic_mix_prob}")
        logger.info(f"SAC n-step: {args.sac_n_step if args.sac_n_step > 0 else 1}")
        logger.info(f"SAC PER alpha: {args.sac_per_alpha if args.sac_per_alpha >= 0 else 0.0}")
        logger.info(f"SAC reward scale: {args.sac_reward_scale if args.sac_reward_scale >= 0 else 0.01}")
        logger.info(f"SAC actor prior: {'on' if args.enable_sac_actor_prior else 'off'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    if start_epoch > 1:
        logger.info(f"Resume from epoch {start_epoch}")
    logger.info("=" * 72)

    total_steps = 0
    effective_warmup_steps = int(args.warmup_steps)
    if args.checkpoint and args.checkpoint_mode == "finetune":
        loaded_update_step = int(getattr(agent, "update_step", 0))
        if loaded_update_step > 0:
            # A pretrained checkpoint already carries mature network weights and
            # optimizer history statistics in the model parameters. For these
            # cases, a very long replay-only warmup wastes early finetune
            # epochs, so we respect the user-configured warmup directly.
            effective_warmup_steps = max(effective_warmup_steps, 0)
            logger.info(
                f"Effective warmup steps: {effective_warmup_steps} "
                f"(finetune from trained checkpoint, update_step={loaded_update_step})"
            )
        else:
            effective_warmup_steps = max(effective_warmup_steps, 3000)
            logger.info(f"Effective warmup steps: {effective_warmup_steps} (finetune)")
    start_time = time.time()
    epoch_times = []
    last_completed_epoch = start_epoch - 1
    interrupted = False
    active_stage_name = current_stage["name"]

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            stage = _select_curriculum_stage(epoch, args)
            if stage["name"] != active_stage_name:
                active_stage_name = stage["name"]
                best_reward = -float("inf")
                best_model_key = None
                tuning_updates = _apply_stage_transition_tuning(agent, stage)
                tuning_text = ""
                if tuning_updates:
                    tuning_text = " | " + ", ".join(f"{k}={v}" for k, v in tuning_updates.items())
                logger.info(
                    f"Curriculum switch -> {stage['label']} | "
                    f"target={'no-target' if not stage['target_enabled'] else f'{stage['target_mass']:.0f}t'}"
                    f"{tuning_text}"
                )

            if (
                reward_config.target_mass != stage["target_mass"]
                or reward_config.enable_target_objective != stage["target_enabled"]
                or reward_config.target_band_tolerance != stage["band_tolerance"]
            ):
                reward_config = _build_reward_config_for_stage(args, stage)
                env.reward_config = reward_config
                env.reward_scheme = env.reward_scheme.__class__(reward_config)

            epoch_start = time.time()
            epoch_rewards = []
            epoch_masses = []
            epoch_energy = []
            epoch_safety_counts = []
            epoch_unsafe_episode_flags = []
            epoch_total_steps = 0
            epoch_unsafe_steps = 0
            epoch_dry_run_steps = 0
            epoch_low_conc_steps = 0

            for _ in range(args.episodes_per_epoch):
                state, _ = env.reset()
                if args.algo == "sac":
                    state = _transform_obs_for_sac(state, args)
                done = False
                episode_reward = 0.0
                info = {}
                episode_safety_violations = 0
                episode_unsafe_steps = 0
                episode_dry_run_steps = 0
                episode_low_conc_steps = 0
                episode_steps = 0

                while not done:
                    if args.algo == "sac" and total_steps < effective_warmup_steps:
                        action = env.action_space.sample().astype(np.float32)
                    else:
                        deterministic_action = False
                        if args.algo == "sac" and args.sac_deterministic_mix_prob > 0.0:
                            deterministic_action = bool(np.random.rand() < args.sac_deterministic_mix_prob)
                        action = agent.select_action(state, deterministic=deterministic_action)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    if args.algo == "sac":
                        next_state = _transform_obs_for_sac(next_state, args)
                    done = terminated or truncated

                    agent.store_transition(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    total_steps += 1
                    episode_steps += 1

                    step_safety_violations = int(info.get("safety_violations", 0))
                    episode_safety_violations += step_safety_violations
                    if bool(info.get("safety_violation", False)) or step_safety_violations > 0:
                        episode_unsafe_steps += 1
                    if bool(info.get("dry_run_violation", False)):
                        episode_dry_run_steps += 1
                    if bool(info.get("low_conc_violation", False)):
                        episode_low_conc_steps += 1

                    if total_steps > effective_warmup_steps:
                        agent.update()

                final_mass = float(info.get("current_mass", info.get("final_mass", 0.0)))
                energy = float(info.get("total_energy_cost", 0.0))

                epoch_rewards.append(float(episode_reward))
                epoch_masses.append(final_mass)
                epoch_energy.append(energy)
                epoch_safety_counts.append(int(episode_safety_violations))
                epoch_unsafe_episode_flags.append(1 if episode_safety_violations > 0 else 0)
                epoch_total_steps += max(episode_steps, 0)
                epoch_unsafe_steps += episode_unsafe_steps
                epoch_dry_run_steps += episode_dry_run_steps
                epoch_low_conc_steps += episode_low_conc_steps

                agent.all_rewards.append(float(episode_reward))
                agent.all_final_masses.append(final_mass)
                agent.all_energy_costs.append(energy)

            last_completed_epoch = epoch
            epoch_times.append(time.time() - epoch_start)
            train_metrics = _summarize_epoch_metrics(
                rewards=epoch_rewards,
                masses=epoch_masses,
                energy=epoch_energy,
                safety_counts=epoch_safety_counts,
                unsafe_episode_flags=epoch_unsafe_episode_flags,
                total_steps=epoch_total_steps,
                unsafe_steps=epoch_unsafe_steps,
                dry_run_steps=epoch_dry_run_steps,
                low_conc_steps=epoch_low_conc_steps,
                reward_config=reward_config,
                stage=stage,
            )
            eval_metrics = _evaluate_agent_policy(
                args,
                agent,
                stage,
                episodes=args.eval_episodes,
                seed_base=args.seed + epoch * 1000,
            )
            selection_metrics = eval_metrics if eval_metrics is not None else train_metrics
            selection_source = "eval" if eval_metrics is not None else "train"

            avg_reward = float(selection_metrics["avg_reward"])
            avg_mass = float(selection_metrics["avg_mass"])
            avg_energy = float(selection_metrics["avg_energy"])
            avg_safety_violations = float(selection_metrics["avg_safety_violations"])
            unsafe_episode_rate = float(selection_metrics["unsafe_episode_rate"])
            unsafe_step_rate = float(selection_metrics["unsafe_step_rate"])
            dry_run_step_rate = float(selection_metrics["dry_run_step_rate"])
            low_conc_step_rate = float(selection_metrics["low_conc_step_rate"])
            completion_rate = float(selection_metrics["completion_rate"])
            inband_rate = float(selection_metrics["inband_rate"])
            avg_target_band_distance = float(selection_metrics["avg_target_band_distance"])

            if stage["target_enabled"] and selection_source == "eval":
                final_inband_rate = inband_rate
                final_target_band_distance = avg_target_band_distance
                final_completion_rate = completion_rate
            else:
                final_inband_rate = float(
                    compute_inband_rate(
                        epoch_masses,
                        final_reward_config.target_mass_low,
                        final_reward_config.target_mass_high,
                    )
                )
                final_target_band_distance = float(
                    np.mean(
                        [
                            _target_band_distance(
                                mass,
                                final_reward_config.target_mass_low,
                                final_reward_config.target_mass_high,
                            )
                            for mass in epoch_masses
                        ]
                    )
                ) if epoch_masses else 0.0
                final_completion_rate = _completion_rate(epoch_masses, final_reward_config.target_mass)

            recent_epoch_times = epoch_times[-10:]
            avg_epoch_time = float(np.mean(recent_epoch_times)) if recent_epoch_times else 0.0
            remaining_epochs = max(args.epochs - epoch, 0)
            eta_seconds = avg_epoch_time * remaining_epochs
            progress_bar = _fmt_progress_bar(epoch, args.epochs, width=30)
            progress_pct = 100.0 * epoch / max(args.epochs, 1)

            candidate_key = _best_model_key_for_stage(
                stage=stage,
                unsafe_step_rate=unsafe_step_rate,
                unsafe_episode_rate=unsafe_episode_rate,
                dry_run_step_rate=dry_run_step_rate,
                low_conc_step_rate=low_conc_step_rate,
                avg_safety_violations=avg_safety_violations,
                avg_energy=avg_energy,
                completion_rate=completion_rate,
                inband_rate=inband_rate,
                avg_target_band_distance=avg_target_band_distance,
                avg_reward=avg_reward,
            )
            is_best = best_model_key is None or candidate_key < best_model_key
            if is_best:
                best_model_key = candidate_key
                best_reward = avg_reward
                stage_best_path = paths["checkpoints_dir"] / f"best_{stage['name']}.pth"
                agent.save(str(stage_best_path))
                if stage["index"] >= 3 or args.curriculum == "none":
                    agent.save(str(paths["best_model_path"]))

            agent.save(str(paths["latest_model_path"]))

            summary = {
                "epoch": epoch,
                "avg_reward": avg_reward,
                "avg_mass": avg_mass,
                "avg_energy": avg_energy,
                "avg_safety_violations": avg_safety_violations,
                "unsafe_episode_rate": unsafe_episode_rate,
                "unsafe_step_rate": unsafe_step_rate,
                "dry_run_step_rate": dry_run_step_rate,
                "low_conc_step_rate": low_conc_step_rate,
                "stage": stage["name"],
                "stage_label": stage["label"],
                "stage_target_enabled": bool(stage["target_enabled"]),
                "stage_target_mass": float(stage["target_mass"]),
                "completion_rate": completion_rate,
                "inband_rate": inband_rate,
                "pass_rate": inband_rate,
                "avg_target_band_distance": avg_target_band_distance,
                "final_completion_rate": final_completion_rate,
                "final_inband_rate": final_inband_rate,
                "final_pass_rate": final_inband_rate,
                "final_target_band_distance": final_target_band_distance,
                "best_reward": float(best_reward),
                "algo": args.algo,
                "selection_source": selection_source,
                "train_avg_reward": float(train_metrics["avg_reward"]),
                "train_avg_mass": float(train_metrics["avg_mass"]),
                "train_avg_energy": float(train_metrics["avg_energy"]),
                "train_avg_safety_violations": float(train_metrics["avg_safety_violations"]),
                "train_unsafe_episode_rate": float(train_metrics["unsafe_episode_rate"]),
                "train_unsafe_step_rate": float(train_metrics["unsafe_step_rate"]),
                "train_dry_run_step_rate": float(train_metrics["dry_run_step_rate"]),
                "train_low_conc_step_rate": float(train_metrics["low_conc_step_rate"]),
                "train_completion_rate": float(train_metrics["completion_rate"]),
                "train_inband_rate": float(train_metrics["inband_rate"]),
                "train_target_band_distance": float(train_metrics["avg_target_band_distance"]),
                "eval_avg_reward": None if eval_metrics is None else float(eval_metrics["avg_reward"]),
                "eval_avg_mass": None if eval_metrics is None else float(eval_metrics["avg_mass"]),
                "eval_avg_energy": None if eval_metrics is None else float(eval_metrics["avg_energy"]),
                "eval_avg_safety_violations": None if eval_metrics is None else float(eval_metrics["avg_safety_violations"]),
                "eval_unsafe_episode_rate": None if eval_metrics is None else float(eval_metrics["unsafe_episode_rate"]),
                "eval_unsafe_step_rate": None if eval_metrics is None else float(eval_metrics["unsafe_step_rate"]),
                "eval_dry_run_step_rate": None if eval_metrics is None else float(eval_metrics["dry_run_step_rate"]),
                "eval_low_conc_step_rate": None if eval_metrics is None else float(eval_metrics["low_conc_step_rate"]),
                "eval_completion_rate": None if eval_metrics is None else float(eval_metrics["completion_rate"]),
                "eval_inband_rate": None if eval_metrics is None else float(eval_metrics["inband_rate"]),
                "eval_target_band_distance": None if eval_metrics is None else float(eval_metrics["avg_target_band_distance"]),
                "elapsed": float(time.time() - start_time),
                "eta_seconds": float(eta_seconds),
            }

            if epoch % args.checkpoint_interval == 0:
                checkpoint_path = paths["checkpoints_dir"] / f"epoch_{epoch}.pth"
                metrics_path = paths["logs_dir"] / f"metrics_epoch_{epoch}.json"
                agent.save(str(checkpoint_path))
                _save_metrics_snapshot(metrics_path, agent, summary)

            _append_log(paths["log_path"], summary, agent, paths["epoch_csv_path"])

            logger.info(
                f"{progress_bar} {progress_pct:5.1f}% | "
                f"{stage['label']} | "
                f"E {epoch:4d}/{args.epochs} | "
                f"{selection_source.upper()} | "
                f"Rwd {avg_reward:+8.2f} | "
                f"Mass {avg_mass:7.2f}/"
                f"{'safe' if not stage['target_enabled'] else f'{reward_config.target_mass_low:.0f}-{reward_config.target_mass_high:.0f}'} | "
                f"Energy {avg_energy:8.2f} | "
                f"UnsafeEp {unsafe_episode_rate:5.1%} | "
                f"UnsafeSt {unsafe_step_rate:5.1%} | "
                f"Dry {dry_run_step_rate:5.1%} | "
                f"LowC {low_conc_step_rate:5.1%} | "
                f"Meet {completion_rate:5.1%} | "
                f"Band {inband_rate:5.1%} | "
                f"{'*BEST*' if is_best else '      '} | "
                f"ETA {_fmt_timedelta(eta_seconds)}"
            )
            if eval_metrics is not None:
                logger.info(
                    "TrainStats | "
                    f"Rwd {train_metrics['avg_reward']:+8.2f} | "
                    f"Mass {train_metrics['avg_mass']:7.2f} | "
                    f"Energy {train_metrics['avg_energy']:8.2f} | "
                    f"UnsafeEp {train_metrics['unsafe_episode_rate']:5.1%} | "
                    f"UnsafeSt {train_metrics['unsafe_step_rate']:5.1%} | "
                    f"Dry {train_metrics['dry_run_step_rate']:5.1%} | "
                    f"LowC {train_metrics['low_conc_step_rate']:5.1%}"
                )

            if args.curriculum == "none":
                noise_updates = _decay_agent_exploration(agent)
                if noise_updates and epoch < args.epochs:
                    logger.info(
                        "Adaptive noise decay | "
                        + ", ".join(f"{k}={v}" for k, v in noise_updates.items())
                    )

    except KeyboardInterrupt:
        interrupted = True
        logger.info("Received Ctrl+C. Saving latest checkpoint and metrics before exit...")
        agent.save(str(paths["latest_model_path"]))
        partial_summary = {
            "epoch": last_completed_epoch,
            "avg_reward": None,
            "avg_mass": None,
            "avg_energy": None,
            "inband_rate": None,
            "pass_rate": None,
            "best_reward": float(best_reward),
            "algo": args.algo,
            "elapsed": float(time.time() - start_time),
            "interrupted": True,
        }
        _save_metrics_snapshot(paths["logs_dir"] / "metrics_interrupted.json", agent, partial_summary)
    finally:
        try:
            agent.save(str(paths["final_model_path"]))
            final_summary = {
                "epoch": last_completed_epoch,
                "best_reward": float(best_reward),
                "algo": args.algo,
                "elapsed": float(time.time() - start_time),
                "interrupted": interrupted,
            }
            _save_metrics_snapshot(paths["metrics_final_path"], agent, final_summary)
            _save_training_csv(paths["csv_path"], agent)

            if not args.no_plot:
                save_training_plot(
                    agent.all_rewards,
                    agent.all_final_masses,
                    agent.all_energy_costs,
                    save_dir=str(paths["plots_dir"]),
                )
        finally:
            release_lock()

    total_time = time.time() - start_time
    logger.info(f"Training finished. Total time: {_fmt_timedelta(total_time)}")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Artifacts saved in: {paths['run_dir']}")


if __name__ == "__main__":
    train()
