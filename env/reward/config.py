"""
Reward configuration for the thickener dewatering environment.

Design priority:
1. Safety first.
2. On the safe set, reach 400 t by the end of the day.
3. On the safe and feasible set, stay close to 400 t instead of aggressively overproducing.
4. Reduce energy cost only after safety and target feasibility are satisfied.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Reward and task configuration for RL training."""

    # Basic task settings
    target_mass: float = 400.0
    max_steps: int = 288
    enable_target_objective: bool = True

    @property
    def target_mass_low(self) -> float:
        return self.target_mass

    @property
    def target_mass_high(self) -> float:
        return self.target_mass + self.target_band_tolerance

    # Dense mass shaping:
    # reward reducing |M - target|, and strongly penalize any extra ton after crossing target.
    target_gap_improvement_weight: float = 2.5
    overshoot_delta_penalty_weight: float = 6.0
    overshoot_inventory_penalty_weight: float = 12.0
    target_cross_bonus: float = 80.0
    in_band_step_bonus: float = 0.0
    throughput_reward_weight: float = 0.0

    # Time pacing:
    # being ahead of schedule is more dangerous than being slightly behind,
    # because the current environment tends to learn overproduction.
    schedule_tolerance_ratio: float = 0.05
    schedule_behind_weight: float = 0.8
    schedule_ahead_weight: float = 4.0

    # Operating cost
    energy_cost_weight: float = 0.30

    # Safety penalties: should dominate any productivity benefit on unsafe trajectories.
    uf_conc_hard_limit: float = 0.75
    uf_conc_penalty: float = 80.0

    buffer_vol_hard_limit: float = 30.0
    buffer_vol_penalty: float = 80.0
    safety_violation_penalty: float = 30.0
    unsafe_step_reward_block: bool = True
    terminal_safety_block_penalty: float = 300.0

    # Terminal objective in ton units:
    # under-target is bad, but large overshoot is punished even harder.
    target_band_tolerance: float = 15.0
    terminal_target_band_bonus: float = 260.0
    terminal_under_penalty_weight: float = 6.0
    terminal_under_penalty_quadratic: float = 0.03
    terminal_over_penalty_weight: float = 8.0
    terminal_over_penalty_quadratic: float = 0.04

    # Smooth control regularization
    smoothness_weight: float = 0.35

    # Reward clipping
    reward_clip_min: float = -1200.0
    reward_clip_max: float = 200.0

    # Keep feed running; let reward shape the stopping strategy.
    target_completion_short_circuit: bool = False
