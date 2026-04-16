"""
评价指标计算

EEI (Energy Efficiency Index): 累计能耗成本 + 安全越限惩罚
"""

import numpy as np
from typing import List, Dict


def compute_eei(energy_cost_sum: float, safety_penalty_sum: float = 0.0) -> float:
    """
    计算能源效率指数 (EEI)

    EEI = 累计能耗成本 + 安全越限惩罚
    越小表示综合运行表现越优
    """
    return energy_cost_sum + abs(safety_penalty_sum)


def compute_pass_rate(final_masses: List[float], target_low: float,
                      target_high: float) -> float:
    """
    计算达标率

    达标条件: target_low <= final_mass <= target_high
    """
    if not final_masses:
        return 0.0
    passed = sum(1 for m in final_masses if target_low <= m <= target_high)
    return passed / len(final_masses)


def compute_action_variation(actions: List[np.ndarray]) -> float:
    """
    计算动作变化幅度均值

    衡量控制平滑性
    """
    if len(actions) < 2:
        return 0.0
    variations = []
    for i in range(1, len(actions)):
        diff = np.abs(np.array(actions[i]) - np.array(actions[i - 1]))
        variations.append(np.mean(diff))
    return float(np.mean(variations))


def evaluate_episode(info_list: List[Dict], target_low: float,
                     target_high: float) -> Dict[str, float]:
    """
    评估单个 episode 的指标

    返回:
        metrics: {
            'final_mass': 最终干矿量,
            'energy_cost': 累计能耗成本,
            'eei': 能源效率指数,
            'switch_count': 开关次数,
            'safety_violations': 安全越限次数,
            'target_reached': 是否达标,
        }
    """
    if not info_list:
        return {}

    last_info = info_list[-1]
    energy_cost = last_info.get('total_energy_cost', 0.0)
    final_mass = last_info.get('final_mass', last_info.get('current_mass', 0.0))
    switch_count = last_info.get('switch_count', 0)
    safety_violations = sum(1 for info in info_list if info.get('safety_violation', False))
    target_reached = last_info.get('target_reached', False)

    eei = compute_eei(energy_cost)

    return {
        'final_mass': final_mass,
        'energy_cost': energy_cost,
        'eei': eei,
        'switch_count': switch_count,
        'safety_violations': safety_violations,
        'target_reached': target_reached,
    }
