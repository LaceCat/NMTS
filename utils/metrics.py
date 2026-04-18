"""
Evaluation metrics for thickener-dewatering experiments.

Primary priority:
1. Safety
2. Energy / EEI
3. Task completion (>= target mass)

Target-band statistics are kept only as descriptive references.
"""

from typing import Dict, List

import numpy as np


def compute_eei(energy_cost_sum: float, safety_penalty_sum: float = 0.0) -> float:
    """
    Compute EEI (Energy Efficiency Index).

    Smaller is better.
    """
    return float(energy_cost_sum + abs(safety_penalty_sum))


def compute_inband_rate(final_masses: List[float], target_low: float, target_high: float) -> float:
    """
    Descriptive statistic only:
    ratio of episodes whose final mass lies inside [target_low, target_high].
    """
    if not final_masses:
        return 0.0
    matched = sum(1 for mass in final_masses if target_low <= mass <= target_high)
    return float(matched / len(final_masses))


def compute_pass_rate(final_masses: List[float], target_low: float, target_high: float) -> float:
    """
    Backward-compatible alias.
    Older code paths may still call this name, but it should be interpreted
    as an in-band descriptive statistic, not as the main pass/fail criterion.
    """
    return compute_inband_rate(final_masses, target_low, target_high)


def compute_action_variation(actions: List[np.ndarray]) -> float:
    """Average step-to-step action variation."""
    if len(actions) < 2:
        return 0.0
    variations = []
    for i in range(1, len(actions)):
        diff = np.abs(np.asarray(actions[i]) - np.asarray(actions[i - 1]))
        variations.append(np.mean(diff))
    return float(np.mean(variations))


def evaluate_episode(info_list: List[Dict], target_low: float, target_high: float) -> Dict[str, float]:
    """
    Evaluate a single episode summary.
    """
    if not info_list:
        return {}

    last_info = info_list[-1]
    energy_cost = float(last_info.get("total_energy_cost", 0.0))
    final_mass = float(last_info.get("final_mass", last_info.get("current_mass", 0.0)))
    switch_count = int(last_info.get("switch_count", 0))
    safety_violations = int(sum(1 for info in info_list if info.get("safety_violation", False)))
    target_reached = bool(last_info.get("target_reached", False))
    inband = bool(target_low <= final_mass <= target_high)

    return {
        "final_mass": final_mass,
        "energy_cost": energy_cost,
        "eei": compute_eei(energy_cost),
        "switch_count": switch_count,
        "safety_violations": safety_violations,
        "target_reached": target_reached,
        "inband": inband,
    }
