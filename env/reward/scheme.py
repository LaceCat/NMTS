"""
Reward computation for the thickener dewatering environment.

Simplified four-part reward:
1. Safety is absolute.
2. Production is rewarded only before the target band.
3. Energy is a secondary running cost.
4. Final success is judged mainly at the episode end.
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

        production_reward = 0.0
        productive_delta = 0.0
        glide_delta = 0.0
        excess_delta = 0.0

        # 1) Production shaping:
        # before target -> reward each extra ton;
        # inside target band -> no extra push;
        # above target band -> only mild negative feedback.
        if rc.enable_target_objective:
            glide_start = max(rc.target_mass - rc.pre_target_glide_margin, 0.0)
            target_capped_prev = min(prev_m_fp, rc.target_mass)
            target_capped_curr = min(m_fp, rc.target_mass)

            below_glide_prev = min(target_capped_prev, glide_start)
            below_glide_curr = min(target_capped_curr, glide_start)
            productive_delta = max(below_glide_curr - below_glide_prev, 0.0)

            glide_prev = np.clip(target_capped_prev, glide_start, rc.target_mass)
            glide_curr = np.clip(target_capped_curr, glide_start, rc.target_mass)
            glide_delta = max(glide_curr - glide_prev, 0.0)

            above_target_prev = max(prev_m_fp - rc.target_mass, 0.0)
            above_target_curr = max(m_fp - rc.target_mass, 0.0)
            excess_delta = max(above_target_curr - above_target_prev, 0.0)
            production_reward = (
                rc.throughput_reward_weight * productive_delta
                + rc.throughput_reward_weight * rc.pre_target_glide_scale * glide_delta
                - rc.post_target_delta_penalty_weight * excess_delta
            )
        else:
            productive_delta = max(delta_m_fp, 0.0)
            production_reward = rc.throughput_reward_weight * productive_delta

        # 2) Energy is important, but still secondary to safety and end-of-day success.
        energy_reward = -rc.energy_cost_weight * energy_cost

        # 3) Safety remains a hard preference in the reward.
        uf_unsafe = c_uf > rc.uf_conc_hard_limit
        buf_unsafe = v_buf > rc.buffer_vol_hard_limit
        unsafe_now = bool(safety_violations > 0 or uf_unsafe or buf_unsafe)

        safety_penalty = 0.0
        if unsafe_now:
            safety_penalty -= rc.safety_violation_penalty * float(max(safety_violations, 1))
        if uf_unsafe:
            exceed = max(c_uf - rc.uf_conc_hard_limit, 0.0)
            safety_penalty -= rc.uf_conc_penalty * (1.0 + 10.0 * exceed)
        if buf_unsafe:
            exceed = max(v_buf - rc.buffer_vol_hard_limit, 0.0)
            safety_penalty -= rc.buffer_vol_penalty * (1.0 + exceed / max(rc.buffer_vol_hard_limit, 1.0))

        # On unsafe steps, do not grant positive production reward.
        if rc.unsafe_step_reward_block and unsafe_now:
            production_reward = min(production_reward, 0.0)

        # 4) Terminal objective:
        # final judgement is dominated by whether the episode lands inside the target band.
        terminal_reward = 0.0
        is_final_step = episode_step >= max_steps - 1
        if is_final_step:
            done = True
            if rc.enable_target_objective:
                deficit = max(rc.target_mass - m_fp, 0.0)
                overshoot = max(m_fp - rc.target_mass_high, 0.0)

                if deficit > 0.0:
                    terminal_reward = (
                        -rc.terminal_under_penalty_weight * deficit
                        -rc.terminal_under_penalty_quadratic * deficit ** 2
                    )
                elif overshoot <= 0.0:
                    terminal_reward = rc.terminal_target_band_bonus
                else:
                    terminal_reward = (
                        -rc.terminal_over_penalty_weight * overshoot
                        -rc.terminal_over_penalty_quadratic * overshoot ** 2
                    )

            if unsafe_now:
                terminal_reward -= rc.terminal_safety_block_penalty

        total_reward = (
            production_reward
            + energy_reward
            + safety_penalty
            + terminal_reward
        )
        total_reward = float(np.clip(total_reward, rc.reward_clip_min, rc.reward_clip_max))

        breakdown = {
            "production": float(production_reward),
            "productive_delta": float(productive_delta),
            "glide_delta": float(glide_delta),
            "excess_delta": float(excess_delta),
            "target_gap": 0.0,
            "overshoot": float(-rc.post_target_delta_penalty_weight * excess_delta),
            "schedule": 0.0,
            "target_cross": 0.0,
            "throughput": float(production_reward),
            "band_hold": 0.0,
            "energy": float(energy_reward),
            "safety": float(safety_penalty),
            "smooth": 0.0,
            "terminal": float(terminal_reward),
            "total": total_reward,
        }
        return total_reward, breakdown, done
