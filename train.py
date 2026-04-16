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
from env.reward.config import RewardConfig
from env.reward.pricing import PricingPresets
from utils.metrics import compute_pass_rate
from utils.visualization import save_training_plot


def parse_args():
    parser = argparse.ArgumentParser(description="Thickener dewatering RL training")

    parser.add_argument("--algo", type=str, default="esac", choices=["esac", "td3"], help="Training algorithm")

    parser.add_argument("--target", type=float, default=400.0, help="Target dry mass in tons")
    parser.add_argument("--steps", type=int, default=288, help="Decision steps per episode")
    parser.add_argument("--interval", type=int, default=5, help="Physical minutes per decision step")

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
        "--curriculum",
        type=str,
        default="safety_mass",
        choices=["none", "safety_mass"],
        help="Training curriculum. safety_mass = stage1 safety, stage2 safe 600t, stage3 safe 400t.",
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
    parser.add_argument("--num_actors", type=int, default=5, help="Number of actors for ESAC")
    parser.add_argument("--alpha", type=float, default=0.99, help="ESAC alpha coefficient")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for resuming")
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

    return parser.parse_args()


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
        f"{args.algo}_target{_format_target(args.target)}"
        f"_steps{args.steps}_int{args.interval}_seed{args.seed}_{timestamp}"
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
        if checkpoint_path.parent.name == "checkpoints":
            run_dir = checkpoint_path.parent.parent
        else:
            run_dir = checkpoint_path.parent
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


def _append_log(log_path: Path, entry: dict, agent):
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


def _target_band_distance(mass: float, low: float, high: float) -> float:
    if mass < low:
        return low - mass
    if mass > high:
        return mass - high
    return 0.0


def _best_model_key(
    *,
    unsafe_step_rate: float,
    unsafe_episode_rate: float,
    pass_rate: float,
    avg_target_band_distance: float,
    avg_reward: float,
):
    return (
        float(unsafe_step_rate),
        float(unsafe_episode_rate),
        -float(pass_rate),
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
            "label": "FINAL",
            "index": 3,
            "target_enabled": True,
            "target_mass": float(args.target),
            "band_tolerance": 15.0,
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
            "label": "S2-600T",
            "index": 2,
            "target_enabled": True,
            "target_mass": max(600.0, float(args.target)),
            "band_tolerance": 30.0,
        }

    return {
        "name": "stage3",
        "label": "S3-400T",
        "index": 3,
        "target_enabled": True,
        "target_mass": float(args.target),
        "band_tolerance": 15.0,
    }


def _build_reward_config_for_stage(args, stage: dict) -> RewardConfig:
    config = RewardConfig(target_mass=float(stage["target_mass"]), max_steps=args.steps)
    config.enable_target_objective = bool(stage["target_enabled"])
    config.target_band_tolerance = float(stage["band_tolerance"])

    if stage["name"] == "stage1":
        config.enable_target_objective = False
        config.throughput_reward_weight = 0.05
        config.energy_cost_weight = 0.10
        config.smoothness_weight = 0.20
        config.uf_conc_penalty = 140.0
        config.buffer_vol_penalty = 140.0
        config.safety_violation_penalty = 40.0
        config.terminal_safety_block_penalty = 400.0
        config.reward_clip_min = -1500.0
        config.reward_clip_max = 150.0
    elif stage["name"] == "stage2":
        config.target_gap_improvement_weight = 1.2
        config.overshoot_delta_penalty_weight = 10.0
        config.overshoot_inventory_penalty_weight = 20.0
        config.target_cross_bonus = 15.0
        config.in_band_step_bonus = 2.0
        config.schedule_behind_weight = 0.4
        config.schedule_ahead_weight = 8.0
        config.energy_cost_weight = 0.25
        config.smoothness_weight = 0.25
        config.terminal_target_band_bonus = 260.0
        config.terminal_under_penalty_weight = 4.0
        config.terminal_under_penalty_quadratic = 0.02
        config.terminal_over_penalty_weight = 16.0
        config.terminal_over_penalty_quadratic = 0.10
    else:
        config.target_gap_improvement_weight = 1.0
        config.overshoot_delta_penalty_weight = 20.0
        config.overshoot_inventory_penalty_weight = 40.0
        config.target_cross_bonus = 5.0
        config.in_band_step_bonus = 4.0
        config.schedule_behind_weight = 0.6
        config.schedule_ahead_weight = 14.0
        config.energy_cost_weight = 0.30
        config.smoothness_weight = 0.35
        config.uf_conc_penalty = 120.0
        config.buffer_vol_penalty = 120.0
        config.safety_violation_penalty = 40.0
        config.terminal_target_band_bonus = 360.0
        config.terminal_under_penalty_weight = 5.0
        config.terminal_under_penalty_quadratic = 0.03
        config.terminal_over_penalty_weight = 28.0
        config.terminal_over_penalty_quadratic = 0.25

    return config


def _best_model_key_for_stage(
    *,
    stage: dict,
    unsafe_step_rate: float,
    unsafe_episode_rate: float,
    avg_safety_violations: float,
    avg_energy: float,
    pass_rate: float,
    avg_target_band_distance: float,
    avg_reward: float,
):
    if not stage["target_enabled"]:
        return (
            float(unsafe_step_rate),
            float(unsafe_episode_rate),
            float(avg_safety_violations),
            float(avg_energy),
            -float(avg_reward),
        )

    return _best_model_key(
        unsafe_step_rate=unsafe_step_rate,
        unsafe_episode_rate=unsafe_episode_rate,
        pass_rate=pass_rate,
        avg_target_band_distance=avg_target_band_distance,
        avg_reward=avg_reward,
    )


def _scale_optimizer_lr(optimizer, scale: float, min_lr: float = 1e-6):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * float(scale), min_lr)


def _apply_stage_transition_tuning(agent, stage: dict):
    if stage["name"] != "stage3":
        return {}

    updates = {}

    if hasattr(agent, "actor_optimizer"):
        _scale_optimizer_lr(agent.actor_optimizer, 0.5)
        updates["actor_lr_scale"] = 0.5
    elif hasattr(agent, "actor_optimizers"):
        for optimizer in agent.actor_optimizers:
            _scale_optimizer_lr(optimizer, 0.5)
        updates["actor_lr_scale"] = 0.5

    if hasattr(agent, "critic1_optimizer"):
        _scale_optimizer_lr(agent.critic1_optimizer, 0.5)
        updates["critic_lr_scale"] = 0.5
    if hasattr(agent, "critic2_optimizer"):
        _scale_optimizer_lr(agent.critic2_optimizer, 0.5)
        updates["critic_lr_scale"] = 0.5

    if hasattr(agent, "exploration_noise_np"):
        agent.exploration_noise_np = agent.exploration_noise_np * 0.5
        updates["exploration_noise_scale"] = 0.5

    if hasattr(agent, "policy_noise_np") and hasattr(agent, "policy_noise"):
        agent.policy_noise_np = agent.policy_noise_np * 0.5
        agent.policy_noise = torch.tensor(agent.policy_noise_np, dtype=torch.float32, device=agent.device)
        updates["policy_noise_scale"] = 0.5

    if hasattr(agent, "noise_clip_np") and hasattr(agent, "noise_clip"):
        agent.noise_clip_np = agent.noise_clip_np * 0.5
        agent.noise_clip = torch.tensor(agent.noise_clip_np, dtype=torch.float32, device=agent.device)
        updates["noise_clip_scale"] = 0.5

    return updates


def _build_agent(args, state_dim: int, action_dim: int):
    if args.algo == "esac":
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
    else:
        from algorithms.td3 import TD3Agent

        buffer_cap = args.buffer_capacity if args.buffer_capacity > 0 else 1_000_000
        batch = args.batch_size if args.batch_size > 0 else 256
        lr_a = args.lr_actor if args.lr_actor > 0 else 1e-4
        lr_c = args.lr_critic if args.lr_critic > 0 else 1e-3

        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_capacity=buffer_cap,
            batch_size=batch,
            hidden_dim=args.hidden_dim,
            lr_actor=lr_a,
            lr_critic=lr_c,
            device=args.device,
        )
        algo_name = "TD3"

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
        pricing=PricingPresets.daily_24h(),
        reward_config=reward_config,
    )

    agent, algo_name = _build_agent(
        args,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    stage1_epochs, stage2_epochs = _resolve_curriculum_lengths(args)

    start_epoch = 1
    best_reward = -float("inf")
    best_model_key = None
    if args.checkpoint:
        agent.load(args.checkpoint)
        start_epoch, best_reward = _load_existing_log(paths["log_path"], agent)

    logger.info("=" * 72)
    logger.info(f"Start {algo_name} training")
    logger.info(f"Target dry mass: {args.target} t")
    logger.info(f"Decision steps: {args.steps}")
    logger.info(f"Decision interval: {args.interval} min")
    logger.info(f"Pass band: {reward_config.target_mass_low:.1f} ~ {reward_config.target_mass_high:.1f} t")
    if args.curriculum == "none":
        logger.info("Curriculum: none")
    else:
        logger.info(
            f"Curriculum: safety_mass | stage1={stage1_epochs} ep (safe only) | "
            f"stage2={stage2_epochs} ep (safe 600t) | stage3={max(args.epochs - stage1_epochs - stage2_epochs, 0)} ep (safe 400t)"
        )
    logger.info(f"Total physical time: {args.steps * args.interval} min = {args.steps * args.interval / 60:.1f} h")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Run root: {paths['save_root']}")
    logger.info(f"Run dir: {paths['run_dir']}")
    if args.algo == "esac":
        logger.info(f"ESAC alpha: {args.alpha}")
        logger.info(f"ESAC actors: {args.num_actors}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    if start_epoch > 1:
        logger.info(f"Resume from epoch {start_epoch}")
    logger.info("=" * 72)

    total_steps = 0
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

            for _ in range(args.episodes_per_epoch):
                state, _ = env.reset()
                done = False
                episode_reward = 0.0
                info = {}
                episode_safety_violations = 0
                episode_unsafe_steps = 0
                episode_steps = 0

                while not done:
                    action = agent.select_action(state, deterministic=False)
                    next_state, reward, terminated, truncated, info = env.step(action)
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

                    if total_steps > args.warmup_steps:
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

                agent.all_rewards.append(float(episode_reward))
                agent.all_final_masses.append(final_mass)
                agent.all_energy_costs.append(energy)

            last_completed_epoch = epoch
            epoch_times.append(time.time() - epoch_start)
            avg_reward = float(np.mean(epoch_rewards))
            avg_mass = float(np.mean(epoch_masses))
            avg_energy = float(np.mean(epoch_energy))
            avg_safety_violations = float(np.mean(epoch_safety_counts)) if epoch_safety_counts else 0.0
            unsafe_episode_rate = float(np.mean(epoch_unsafe_episode_flags)) if epoch_unsafe_episode_flags else 0.0
            unsafe_step_rate = float(epoch_unsafe_steps / max(epoch_total_steps, 1))
            if stage["target_enabled"]:
                pass_rate = float(
                    compute_pass_rate(epoch_masses, reward_config.target_mass_low, reward_config.target_mass_high)
                )
                avg_target_band_distance = float(
                    np.mean(
                        [
                            _target_band_distance(mass, reward_config.target_mass_low, reward_config.target_mass_high)
                            for mass in epoch_masses
                        ]
                    )
                )
            else:
                pass_rate = 0.0
                avg_target_band_distance = 0.0

            final_pass_rate = float(
                compute_pass_rate(
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
            )

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
                avg_safety_violations=avg_safety_violations,
                avg_energy=avg_energy,
                pass_rate=pass_rate,
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
                "stage": stage["name"],
                "stage_label": stage["label"],
                "stage_target_enabled": bool(stage["target_enabled"]),
                "stage_target_mass": float(stage["target_mass"]),
                "pass_rate": pass_rate,
                "avg_target_band_distance": avg_target_band_distance,
                "final_pass_rate": final_pass_rate,
                "final_target_band_distance": final_target_band_distance,
                "best_reward": float(best_reward),
                "algo": args.algo,
                "elapsed": float(time.time() - start_time),
                "eta_seconds": float(eta_seconds),
            }

            if epoch % args.checkpoint_interval == 0:
                checkpoint_path = paths["checkpoints_dir"] / f"epoch_{epoch}.pth"
                metrics_path = paths["logs_dir"] / f"metrics_epoch_{epoch}.json"
                agent.save(str(checkpoint_path))
                _save_metrics_snapshot(metrics_path, agent, summary)

            _append_log(paths["log_path"], summary, agent)

            logger.info(
                f"{progress_bar} {progress_pct:5.1f}% | "
                f"{stage['label']} | "
                f"E {epoch:4d}/{args.epochs} | "
                f"Rwd {avg_reward:+8.2f} | "
                f"Mass {avg_mass:7.2f}/"
                f"{'safe' if not stage['target_enabled'] else f'{reward_config.target_mass_low:.0f}-{reward_config.target_mass_high:.0f}'} | "
                f"Energy {avg_energy:8.2f} | "
                f"UnsafeEp {unsafe_episode_rate:5.1%} | "
                f"UnsafeSt {unsafe_step_rate:5.1%} | "
                f"Pass {final_pass_rate:5.1%} | "
                f"{'*BEST*' if is_best else '      '} | "
                f"ETA {_fmt_timedelta(eta_seconds)}"
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
