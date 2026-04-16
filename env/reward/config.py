"""
Reward configuration for the thickener dewatering environment.

Design priority:
1. Safety first.
2. Finish the day inside the target band.
3. Allow the process to glide through inertia instead of overreacting near 400 t.
4. Use energy only as a secondary cost signal.
"""

from dataclasses import dataclass

from env.constants import DEFAULT_CONTROL_STEPS


@dataclass
class RewardConfig:
    """Reward and task configuration for RL training."""

    # Basic task settings
    target_mass: float = 400.0
    max_steps: int = DEFAULT_CONTROL_STEPS
    enable_target_objective: bool = True

    @property
    def target_mass_low(self) -> float:
        return self.target_mass

    @property
    def target_mass_high(self) -> float:
        return self.target_mass + self.target_band_tolerance

    # Simplified V4-style shaping:
    # before target -> reward production; between [target, high] -> let inertia glide;
    # above target band -> only mild per-step penalty and let terminal reward dominate.
    throughput_reward_weight: float = 0.8
    post_target_delta_penalty_weight: float = 8.0
    pre_target_glide_margin: float = 180.0
    pre_target_glide_scale: float = 0.05

    # Legacy knobs kept for backward compatibility with older configs / logs.
    target_gap_improvement_weight: float = 0.0
    overshoot_delta_penalty_weight: float = 0.0
    overshoot_inventory_penalty_weight: float = 0.0
    target_cross_bonus: float = 0.0
    in_band_step_bonus: float = 0.0
    schedule_tolerance_ratio: float = 0.0
    schedule_behind_weight: float = 0.0
    schedule_ahead_weight: float = 0.0

    # Operating cost
    energy_cost_weight: float = 0.40

    # Safety penalties: should dominate any productivity benefit on unsafe trajectories.
    uf_conc_hard_limit: float = 0.75
    uf_conc_penalty: float = 400.0

    buffer_vol_hard_limit: float = 30.0
    buffer_vol_penalty: float = 400.0
    safety_violation_penalty: float = 1000.0
    unsafe_step_reward_block: bool = True
    terminal_safety_block_penalty: float = 1500.0

    # Terminal objective in ton units:
    # under-target is heavily punished, while mild overshoot is tolerated more than safety violations.
    target_band_tolerance: float = 20.0
    terminal_target_band_bonus: float = 2600.0
    terminal_under_penalty_weight: float = 15.0
    terminal_under_penalty_quadratic: float = 0.0
    terminal_over_penalty_weight: float = 16.0
    terminal_over_penalty_quadratic: float = 0.0

    # Smoothness is intentionally disabled in the simplified reward.
    smoothness_weight: float = 0.0

    # Reward clipping
    reward_clip_min: float = -5000.0
    reward_clip_max: float = 2500.0

    # Keep feed running; let reward shape the stopping strategy.
    target_completion_short_circuit: bool = False
