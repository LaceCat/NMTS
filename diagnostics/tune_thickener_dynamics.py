from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from env.physics.thickener import (
    DEFAULT_INITIAL_CONCENTRATION_PROFILE,
    DEFAULT_THICKENER_STATE,
    ThickenerModel,
)


TARGET_RESPONSE = {
    "initial": 0.6592648463090376,
    "steady": 0.6940544935389042,
    "delay_5pct": 9,
    "tau_63pct": 70,
    "t95": 237,
}

BASE_PARAMS = list(ThickenerModel()._params)
TARGET_WARMUP_PROFILE = np.array(DEFAULT_INITIAL_CONCENTRATION_PROFILE, dtype=np.float64)
TARGET_WARMUP_QF = 41.87270059423681
TARGET_WARMUP_CF = 0.3975357153204958
TARGET_WARMUP_QUF = 20.0
TARGET_WARMUP_BOTTOM = 0.66


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune thickener PDE parameters toward target step-response dynamics."
    )
    parser.add_argument("--qf", type=float, default=42.5, help="Feed flow rate used for identification.")
    parser.add_argument("--cf", type=float, default=0.375, help="Feed concentration used for identification.")
    parser.add_argument("--step_q_uf", type=float, default=25.0, help="Constant underflow action for the step-response test.")
    parser.add_argument("--horizon", type=int, default=400, help="Simulation horizon in minutes.")
    parser.add_argument("--random_trials", type=int, default=300, help="Number of random search candidates.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the search process.")
    parser.add_argument("--warmup_qf", type=float, default=TARGET_WARMUP_QF, help="Warm-up feed flow rate for profile matching.")
    parser.add_argument("--warmup_cf", type=float, default=TARGET_WARMUP_CF, help="Warm-up feed concentration for profile matching.")
    parser.add_argument("--warmup_q_uf", type=float, default=TARGET_WARMUP_QUF, help="Warm-up underflow rate for profile matching.")
    parser.add_argument("--warmup_target", type=float, default=TARGET_WARMUP_BOTTOM, help="Target bottom concentration used to stop warm-up.")
    parser.add_argument("--warmup_max_minutes", type=int, default=1000, help="Maximum warm-up minutes for profile matching.")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/dynamics_tuning/best_result.json",
        help="Path for the saved tuning summary JSON.",
    )
    return parser.parse_args()


def make_scaled_params(scales: dict[str, float]) -> list[float]:
    params = list(BASE_PARAMS)
    params[0] *= scales["D_top"]
    params[1] *= scales["D_top"]
    params[2] *= scales["D_low"]
    params[3] *= scales["D_low"]

    params[4] *= scales["v_top"]
    params[5] *= scales["v_mid"]
    params[6] *= scales["v_low"]
    params[7] *= scales["v_low"]

    params[8] *= scales["rv_top"]
    params[9] *= scales["rv_mid"]
    params[10] *= scales["rv_low"]
    params[11] *= scales["rv_low"]
    return params


def measure_step_response(
    *,
    scales: dict[str, float],
    qf: float,
    cf: float,
    step_q_uf: float,
    horizon: int,
) -> dict[str, float | int | None]:
    model = ThickenerModel()
    model._params = make_scaled_params(scales)

    state = DEFAULT_THICKENER_STATE.copy()
    series = []
    for _ in range(horizon):
        c_uf_profile, state = model.step(step_q_uf, qf, cf, state)
        series.append(float(c_uf_profile[-1]))

    response = np.asarray(series, dtype=np.float64)
    initial = float(DEFAULT_INITIAL_CONCENTRATION_PROFILE[-1])
    steady = float(np.mean(response[-30:]))
    delta = steady - initial

    delay_5pct = None
    t63_absolute = None
    t95 = None

    if abs(delta) > 1e-12:
        delay_idx = np.where(np.abs(response - initial) >= 0.05 * abs(delta))[0]
        if len(delay_idx) > 0:
            delay_5pct = int(delay_idx[0] + 1)

        t63_idx = np.where(np.abs(response - initial) >= 0.632 * abs(delta))[0]
        if len(t63_idx) > 0:
            t63_absolute = int(t63_idx[0] + 1)

        t95_idx = np.where(np.abs(response - initial) >= 0.95 * abs(delta))[0]
        if len(t95_idx) > 0:
            t95 = int(t95_idx[0] + 1)

    tau_63pct = None
    if delay_5pct is not None and t63_absolute is not None:
        tau_63pct = int(t63_absolute - delay_5pct)

    return {
        "initial": initial,
        "steady": steady,
        "delta": delta,
        "delay_5pct": delay_5pct,
        "tau_63pct": tau_63pct,
        "t63_absolute": t63_absolute,
        "t95": t95,
    }


def measure_warmup_profile(
    *,
    scales: dict[str, float],
    qf: float,
    cf: float,
    q_uf: float,
    target_bottom: float,
    max_minutes: int,
) -> dict[str, float | int | list[float] | None]:
    model = ThickenerModel()
    model._params = make_scaled_params(scales)

    state = np.ones(model.N_LAYERS, dtype=np.float64) * 1e6

    reached_profile = None
    reached_minute = None
    for minute in range(max_minutes + 1):
        profile = np.array([max(0.0, model.d2c(value / 1e6)) for value in state], dtype=np.float64)
        if float(profile[-1]) >= target_bottom:
            reached_profile = profile
            reached_minute = minute
            break
        _, state = model.step(q_uf, qf, cf, state)

    if reached_profile is None:
        return {
            "warmup_minutes": None,
            "profile_rmse": None,
            "top6_near_zero": None,
            "profile": None,
        }

    rmse = float(np.sqrt(np.mean((reached_profile - TARGET_WARMUP_PROFILE) ** 2)))
    top6_near_zero = int(np.sum(reached_profile[:6] < 1e-4))
    return {
        "warmup_minutes": int(reached_minute),
        "profile_rmse": rmse,
        "top6_near_zero": top6_near_zero,
        "profile": reached_profile.tolist(),
    }


def score_metrics(
    metrics: dict[str, float | int | None],
    warmup: dict[str, float | int | list[float] | None],
) -> float:
    if (
        metrics["delay_5pct"] is None
        or metrics["tau_63pct"] is None
        or metrics["t95"] is None
        or warmup["profile_rmse"] is None
        or warmup["top6_near_zero"] is None
        or warmup["warmup_minutes"] is None
    ):
        return 1e12

    score = 0.0
    score += ((float(metrics["initial"]) - TARGET_RESPONSE["initial"]) / 0.01) ** 2
    score += ((float(metrics["steady"]) - TARGET_RESPONSE["steady"]) / 0.03) ** 2
    score += ((int(metrics["delay_5pct"]) - TARGET_RESPONSE["delay_5pct"]) / 4.0) ** 2
    score += ((int(metrics["tau_63pct"]) - TARGET_RESPONSE["tau_63pct"]) / 25.0) ** 2
    score += ((int(metrics["t95"]) - TARGET_RESPONSE["t95"]) / 50.0) ** 2
    score += (float(warmup["profile_rmse"]) / 0.02) ** 2 * 4.0
    score += float(warmup["top6_near_zero"]) * 4.0
    score += ((int(warmup["warmup_minutes"]) - 191) / 50.0) ** 2
    return float(score)


def seed_candidates() -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for d_top in (0.5, 1.0, 2.0):
        for d_low in (0.5, 1.0, 2.0):
            for v_top in (0.25, 0.5, 1.0, 2.0, 4.0):
                for v_mid in (0.25, 0.5, 1.0, 2.0, 4.0):
                    for v_low in (0.25, 0.5, 1.0, 2.0, 4.0):
                        for rv_top in (0.5, 1.0, 2.0):
                            for rv_mid in (0.5, 1.0, 2.0):
                                for rv_low in (0.5, 1.0, 2.0):
                                    if len(candidates) >= 200:
                                        return candidates
                                    candidates.append(
                                        {
                                            "D_top": d_top,
                                            "D_low": d_low,
                                            "v_top": v_top,
                                            "v_mid": v_mid,
                                            "v_low": v_low,
                                            "rv_top": rv_top,
                                            "rv_mid": rv_mid,
                                            "rv_low": rv_low,
                                        }
                                    )
    return candidates


def random_candidates(count: int, rng: random.Random) -> list[dict[str, float]]:
    items = []
    for _ in range(count):
        items.append(
            {
                "D_top": 10 ** rng.uniform(-0.7, 0.7),
                "D_low": 10 ** rng.uniform(-0.7, 0.7),
                "v_top": 10 ** rng.uniform(-0.7, 0.7),
                "v_mid": 10 ** rng.uniform(-0.7, 0.7),
                "v_low": 10 ** rng.uniform(-0.7, 0.7),
                "rv_top": 10 ** rng.uniform(-0.7, 0.7),
                "rv_mid": 10 ** rng.uniform(-0.7, 0.7),
                "rv_low": 10 ** rng.uniform(-0.7, 0.7),
            }
        )
    return items


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    candidates = seed_candidates() + random_candidates(args.random_trials, rng)

    best_score = float("inf")
    best_scales: dict[str, float] | None = None
    best_metrics: dict[str, float | int | None] | None = None
    best_warmup: dict[str, float | int | list[float] | None] | None = None

    for index, scales in enumerate(candidates, start=1):
        metrics = measure_step_response(
            scales=scales,
            qf=args.qf,
            cf=args.cf,
            step_q_uf=args.step_q_uf,
            horizon=args.horizon,
        )
        warmup = measure_warmup_profile(
            scales=scales,
            qf=args.warmup_qf,
            cf=args.warmup_cf,
            q_uf=args.warmup_q_uf,
            target_bottom=args.warmup_target,
            max_minutes=args.warmup_max_minutes,
        )
        score = score_metrics(metrics, warmup)
        if score < best_score:
            best_score = score
            best_scales = scales
            best_metrics = metrics
            best_warmup = warmup
            print(
                f"best@{index}: score={best_score:.4f} | "
                f"delay={metrics['delay_5pct']} tau={metrics['tau_63pct']} "
                f"steady={metrics['steady']:.6f} "
                f"profile_rmse={warmup['profile_rmse']}"
            )

    result = {
        "target_response": TARGET_RESPONSE,
        "search_config": {
            "qf": args.qf,
            "cf": args.cf,
            "step_q_uf": args.step_q_uf,
            "horizon": args.horizon,
            "random_trials": args.random_trials,
            "seed": args.seed,
            "num_candidates": len(candidates),
        },
        "best_score": best_score,
        "best_scales": best_scales,
        "best_metrics": best_metrics,
        "best_warmup": best_warmup,
        "best_params": make_scaled_params(best_scales) if best_scales is not None else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
