"""
Reward configuration for the thickener dewatering environment.

Design priority:
1. Safety first.
2. Finish the day at or above the required production target.
3. Minimize price-weighted energy cost once safety and completion are satisfied.
4. Staying close to 400 t is only a secondary preference, not the main objective.
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
    # Reward mode:
    # - default: legacy multi-term shaping
    # - progress_constraints: simplified "progress reward + constraint penalties"
    # - minimal_constraints: aggressively simplified "progress + energy + hard/soft constraints + terminal"
    reward_mode: str = "default"

    @property
    def target_mass_low(self) -> float:
        return self.target_mass

    @property
    def target_mass_high(self) -> float:
        return self.target_mass + self.target_band_tolerance

    # Simplified shaping:
    # before target -> small production incentive;
    # after target -> let energy and terminal criteria dominate;
    # exact target-band landing is only a weak preference.
    throughput_reward_weight: float = 1.0
    post_target_delta_penalty_weight: float = 2.5
    pre_target_glide_margin: float = 160.0
    pre_target_glide_scale: float = 0.08
    concentration_progress_weight: float = 0.0

    # Legacy knobs kept for backward compatibility with older configs / logs.
    target_gap_improvement_weight: float = 0.0
    overshoot_delta_penalty_weight: float = 0.0
    overshoot_inventory_penalty_weight: float = 0.0
    target_cross_bonus: float = 0.0
    in_band_step_bonus: float = 0.0
    schedule_tolerance_ratio: float = 0.0
    schedule_behind_weight: float = 0.0
    schedule_ahead_weight: float = 0.0

    # Operating cost. In the final stage this should dominate once safety and
    # target completion are both satisfied.
    energy_cost_weight: float = 0.90
    fp_usage_weight: float = 0.0
    post_target_fp_usage_weight: float = 0.0
    post_target_q_uf_hold_weight: float = 0.0
    post_target_q_uf_guard_level: float = 3.0
    # Post-target operating logic:
    # once 400 t is already completed, the filter press should normally stay off
    # unless the buffer is getting close to its upper limit. We therefore add an
    # extra "hold" penalty on Q_fp whenever the target is met but the buffer is
    # still comfortably below a guard level.
    post_target_buffer_guard_level: float = 24.0
    post_target_buffer_hold_weight: float = 0.0

    # Safety penalties: should dominate any productivity benefit on unsafe trajectories.
    uf_conc_hard_limit: float = 0.75
    uf_conc_penalty: float = 400.0

    buffer_vol_hard_limit: float = 30.0
    buffer_vol_penalty: float = 400.0
    safety_violation_penalty: float = 1000.0
    unsafe_step_reward_block: bool = True
    terminal_safety_block_penalty: float = 1500.0

    # Medium-severity operational constraints:
    # filter-press dry-run and underflow concentration below 0.66 are both
    # undesirable and should be penalized explicitly, but they are intentionally
    # lighter than hard unsafe events such as overflow or excessive C_uf.
    # The improved physical environment clips filter-press withdrawal at the
    # available inventory. A dry-run should therefore mean attempting to pull
    # below zero inventory, not merely operating with a small positive buffer.
    dry_run_buffer_threshold: float = 0.0
    dry_run_penalty: float = 200.0
    # Dense dry-run guidance:
    # when the buffer is near empty, keeping Q_fp high is already a bad control
    # direction even before a discrete dry-run event is counted.
    dry_run_flow_penalty_weight: float = 0.0

    # Engineering power-off zone for the buffer mixer.
    # Once the buffer inventory is already very low, standby stirring power is
    # considered avoidable and the policy can be rewarded for fully shutting
    # down the tail section after the production target is reached.
    mixer_idle_volume_threshold: float = 1.5
    idle_shutdown_q_fp_threshold: float = 3.0
    idle_shutdown_bonus: float = 0.0

    # Direct-Q_fp training support:
    # once environment-side soft Q_fp post-processing is removed, we want the
    # actor to stop relying on infeasible high Q_fp commands that later get
    # reduced by physical execution limits. These penalties are applied on the
    # gap between:
    # - scheduled Q_fp and minute-level executed/applied Q_fp
    # - applied Q_fp and actual slurry-limited Q_fp
    q_fp_schedule_gap_penalty_weight: float = 0.0
    q_fp_actual_gap_penalty_weight: float = 0.0
    q_fp_gap_tolerance: float = 0.0
    # Discourage relying on environment-side Q_fp correction during the final
    # governor-withdrawal stage. Small residual correction inside the tolerance
    # band is allowed, but persistent dependence should be pushed back into the
    # policy itself.
    guard_intervention_penalty_weight: float = 0.0
    q_fp_correction_excess_penalty_weight: float = 0.0
    q_fp_correction_tolerance: float = 0.0

    # Product-quality soft constraint:
    # keep C_uf above 0.66 when possible, but do not treat short violations as
    # catastrophes on the same level as hard safety-limit breaches.
    uf_conc_soft_low_limit: float = 0.66
    uf_conc_low_penalty: float = 150.0
    # Continuous concentration-quality shaping:
    # add a smoother signal around a desired operating region so the policy can
    # distinguish "slightly low" from "far too low", instead of learning only
    # from a binary threshold-style penalty.
    uf_conc_guidance_target: float = 0.68
    uf_conc_guidance_band: float = 0.02
    uf_conc_guidance_start_ratio: float = 0.0
    uf_conc_guidance_mass_gate_ratio: float = 0.0
    # Use a separate upper soft limit so the controller can operate above the
    # nominal target concentration without being punished immediately, while
    # still receiving a clear warning before the hard unsafe limit at 0.75.
    uf_conc_guidance_upper_soft_limit: float = 0.68
    uf_conc_guidance_below_weight: float = 0.0
    uf_conc_guidance_above_weight: float = 0.0
    uf_conc_guidance_band_bonus: float = 0.0
    # When the underflow concentration is too low, a large Q_uf usually pushes
    # the policy deeper into a dilute operating regime. This term gives the
    # learner a more direct hint: if C_uf is low, avoid keeping Q_uf high.
    uf_low_conc_flow_penalty_weight: float = 0.0
    # For the simplified reward, medium-severity constraint violations can also
    # block positive progress reward on the current step so the agent does not
    # learn to "buy" production using bad operating practice.
    constraint_step_reward_block: bool = False

    # Terminal objective in ton units:
    # under-target is heavily punished, while mild overshoot is tolerated
    # more than safety violations and is primarily handled through energy cost.
    target_band_tolerance: float = 20.0
    terminal_target_band_bonus: float = 2600.0
    terminal_inband_over_penalty_weight: float = 0.0
    terminal_under_penalty_weight: float = 18.0
    terminal_under_penalty_quadratic: float = 0.0
    terminal_over_penalty_weight: float = 2.0
    terminal_over_penalty_quadratic: float = 0.0
    # Episode-level average concentration objective:
    # only matters at the episode end, and should be secondary to target
    # completion. Use it to prefer policies that achieve the target while
    # staying in a higher average underflow-concentration regime.
    terminal_avg_cuf_threshold: float = 0.71
    terminal_avg_cuf_bonus_weight: float = 0.0
    terminal_avg_cuf_penalty_weight: float = 0.0

    # Smoothness is intentionally disabled in the simplified reward.
    smoothness_weight: float = 0.0

    # Reward clipping
    reward_clip_min: float = -5000.0
    reward_clip_max: float = 2500.0

    # Keep feed running; let reward shape the stopping strategy.
    target_completion_short_circuit: bool = False
