"""
Reward computation for the thickener dewatering environment.

Key idea:
reward the controller for moving toward the 400 t target,
and penalize it for continuing to produce after the target has been crossed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class RewardScheme:
    """Compute shaped rewards for training."""

    def __init__(self, config):
        self.config = config

    def compute(
        self,
        *,
        delta_m_fp: float,
        energy_cost: float,
        current_price: float,
        min_price: float,
        max_price: float,
        m_fp: float,
        c_uf: float,
        v_buf: float,
        c_aver: float,
        q_uf: float,
        q_fp: float,
        prev_q_uf: float,
        prev_q_fp: float,
        prev_v_buf: float,
        prev_c_aver: float,
        prev_m_fp: float,
        timecnt: int,
        max_time_steps: int,
        is_safe: bool,
        target_reached: bool,
        episode_step: int,
        max_steps: int,
        safety_violations: int = 0,
    ) -> Tuple[float, Dict[str, float], bool]:
        del current_price, min_price, max_price, c_aver, prev_v_buf, prev_c_aver
        del timecnt, max_time_steps, is_safe

        rc = self.config
        done = False

        target_gap_reward = 0.0
        overshoot_penalty = 0.0
        schedule_penalty = 0.0
        target_cross_reward = 0.0
        throughput_reward = 0.0
        band_hold_reward = 0.0

        # 1) Curriculum-aware mass objective:
        # Stage 1 can disable hard production targets and focus on safe operation.
        if rc.enable_target_objective:
            prev_gap = abs(rc.target_mass - prev_m_fp)
            curr_gap = abs(rc.target_mass - m_fp)
            target_gap_reward = rc.target_gap_improvement_weight * (prev_gap - curr_gap)

            prev_overshoot = max(prev_m_fp - rc.target_mass, 0.0)
            curr_overshoot = max(m_fp - rc.target_mass, 0.0)
            overshoot_delta = max(curr_overshoot - prev_overshoot, 0.0)
            overshoot_penalty = (
                -rc.overshoot_delta_penalty_weight * overshoot_delta
                -rc.overshoot_inventory_penalty_weight * (curr_overshoot / rc.target_mass)
            )

            target_progress = rc.target_mass * min((episode_step + 1) / max_steps, 1.0)
            schedule_slack = rc.target_mass * rc.schedule_tolerance_ratio
            behind_gap = max(target_progress - schedule_slack - m_fp, 0.0)
            ahead_gap = max(m_fp - target_progress - schedule_slack, 0.0)
            schedule_penalty = (
                -rc.schedule_behind_weight * behind_gap / rc.target_mass
                -rc.schedule_ahead_weight * ahead_gap / rc.target_mass
            )

            if (not target_reached) and prev_m_fp < rc.target_mass <= m_fp:
                target_cross_reward = rc.target_cross_bonus

            if rc.target_mass_low <= m_fp <= rc.target_mass_high:
                band_hold_reward = rc.in_band_step_bonus
        else:
            throughput_reward = rc.throughput_reward_weight * max(delta_m_fp, 0.0)

        # 2) Energy is important, but secondary to safety and feasible production.
        energy_reward = -rc.energy_cost_weight * energy_cost

        # 3) Safety remains a hard preference in the reward.
        uf_unsafe = c_uf > rc.uf_conc_hard_limit
        buf_unsafe = v_buf > rc.buffer_vol_hard_limit
        unsafe_now = bool(safety_violations > 0 or uf_unsafe or buf_unsafe)

        safety_penalty = -rc.safety_violation_penalty * float(safety_violations)
        if uf_unsafe:
            exceed = max(c_uf - rc.uf_conc_hard_limit, 0.0)
            safety_penalty -= rc.uf_conc_penalty * (1.0 + 10.0 * exceed)
        if buf_unsafe:
            exceed = max(v_buf - rc.buffer_vol_hard_limit, 0.0)
            safety_penalty -= rc.buffer_vol_penalty * (1.0 + exceed / max(rc.buffer_vol_hard_limit, 1.0))

        # On unsafe steps, do not grant positive progress/target bonuses.
        if rc.unsafe_step_reward_block and unsafe_now:
            target_gap_reward = min(target_gap_reward, 0.0)
            target_cross_reward = 0.0
            throughput_reward = min(throughput_reward, 0.0)
            band_hold_reward = 0.0

        # 4) Smooth actions to suppress chattering.
        smooth_penalty = 0.0
        if episode_step > 0:
            du = abs(q_uf - prev_q_uf) / 50.0
            df = abs(q_fp - prev_q_fp) / 70.0
            smooth_penalty = -rc.smoothness_weight * (du + df)

        # 5) Terminal objective:
        # Stage 1 can skip mass targets, while later stages use progressively tighter bands.
        terminal_reward = 0.0
        is_final_step = episode_step >= max_steps - 1
        if is_final_step:
            done = True
            if rc.enable_target_objective:
                deficit = max(rc.target_mass - m_fp, 0.0)
                overshoot = max(m_fp - rc.target_mass, 0.0)

                if deficit > 0.0:
                    terminal_reward = (
                        -rc.terminal_under_penalty_weight * deficit
                        -rc.terminal_under_penalty_quadratic * deficit ** 2
                    )
                elif overshoot <= rc.target_band_tolerance:
                    terminal_reward = rc.terminal_target_band_bonus - 2.0 * overshoot
                else:
                    excess = overshoot - rc.target_band_tolerance
                    terminal_reward = (
                        rc.terminal_target_band_bonus
                        - 2.0 * rc.target_band_tolerance
                        - rc.terminal_over_penalty_weight * excess
                        - rc.terminal_over_penalty_quadratic * excess ** 2
                    )

            if unsafe_now:
                terminal_reward -= rc.terminal_safety_block_penalty

        total_reward = (
            target_gap_reward
            + overshoot_penalty
            + schedule_penalty
            + target_cross_reward
            + throughput_reward
            + band_hold_reward
            + energy_reward
            + safety_penalty
            + smooth_penalty
            + terminal_reward
        )
        total_reward = float(np.clip(total_reward, rc.reward_clip_min, rc.reward_clip_max))

        breakdown = {
            "target_gap": float(target_gap_reward),
            "overshoot": float(overshoot_penalty),
            "schedule": float(schedule_penalty),
            "target_cross": float(target_cross_reward),
            "throughput": float(throughput_reward),
            "band_hold": float(band_hold_reward),
            "energy": float(energy_reward),
            "safety": float(safety_penalty),
            "smooth": float(smooth_penalty),
            "terminal": float(terminal_reward),
            "total": total_reward,
        }
        return total_reward, breakdown, done
