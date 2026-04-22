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


def _compute_concentration_guidance_reward(
    rc,
    c_uf: float,
    m_fp: float,
    episode_step: int,
    max_steps: int,
) -> float:
    if (
        rc.uf_conc_guidance_below_weight <= 0.0
        and rc.uf_conc_guidance_above_weight <= 0.0
        and rc.uf_conc_guidance_band_bonus == 0.0
    ):
        return 0.0

    start_ratio = float(np.clip(getattr(rc, "uf_conc_guidance_start_ratio", 0.0), 0.0, 1.0))
    if max_steps > 0 and episode_step < start_ratio * max_steps:
        return 0.0
    mass_gate_ratio = float(np.clip(getattr(rc, "uf_conc_guidance_mass_gate_ratio", 0.0), 0.0, 2.0))
    if rc.enable_target_objective and mass_gate_ratio > 0.0 and m_fp < mass_gate_ratio * rc.target_mass:
        return 0.0

    target = float(rc.uf_conc_guidance_target)
    band = max(float(rc.uf_conc_guidance_band), 0.0)
    upper_soft_limit = max(float(getattr(rc, "uf_conc_guidance_upper_soft_limit", target)), target)
    below_gap = max(target - c_uf, 0.0)
    above_gap = max(c_uf - upper_soft_limit, 0.0)

    reward = 0.0
    reward -= rc.uf_conc_guidance_below_weight * (below_gap ** 2)
    reward -= rc.uf_conc_guidance_above_weight * (above_gap ** 2)
    if band > 0.0:
        distance = abs(c_uf - target)
        if distance <= band:
            reward += rc.uf_conc_guidance_band_bonus * (1.0 - distance / band)
    elif c_uf == target:
        reward += rc.uf_conc_guidance_band_bonus
    return float(reward)


def _compute_terminal_average_cuf_reward(rc, episode_mean_c_uf: float) -> float:
    threshold = float(getattr(rc, "terminal_avg_cuf_threshold", 0.0))
    bonus_weight = float(getattr(rc, "terminal_avg_cuf_bonus_weight", 0.0))
    penalty_weight = float(getattr(rc, "terminal_avg_cuf_penalty_weight", 0.0))
    if bonus_weight <= 0.0 and penalty_weight <= 0.0:
        return 0.0

    reward = 0.0
    above_gap = max(float(episode_mean_c_uf) - threshold, 0.0)
    below_gap = max(threshold - float(episode_mean_c_uf), 0.0)
    reward += bonus_weight * above_gap
    reward -= penalty_weight * below_gap
    return float(reward)


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
        prev_c_uf: float,
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
        episode_mean_c_uf: float = 0.0,
        safety_violations: int = 0,
        dry_run_minutes: int = 0,
        low_conc_minutes: int = 0,
    ) -> Tuple[float, Dict[str, float], bool]:
        del min_price, c_aver, prev_v_buf, prev_c_aver
        del timecnt, max_time_steps, is_safe

        rc = self.config
        done = False

        reward_mode = getattr(rc, "reward_mode", "default")
        if reward_mode in {"progress_constraints", "minimal_constraints"}:
            minimal_mode = reward_mode == "minimal_constraints"
            progress_reward = 0.0
            productive_delta = 0.0
            excess_delta = 0.0
            target_cross_reward = 0.0

            if rc.enable_target_objective:
                prev_gap = max(rc.target_mass - prev_m_fp, 0.0)
                curr_gap = max(rc.target_mass - m_fp, 0.0)
                progress_reward = rc.throughput_reward_weight * (prev_gap - curr_gap)

                if not minimal_mode:
                    prev_c_gap = max(rc.uf_conc_soft_low_limit - prev_c_uf, 0.0)
                    curr_c_gap = max(rc.uf_conc_soft_low_limit - c_uf, 0.0)
                    progress_reward += rc.concentration_progress_weight * (prev_c_gap - curr_c_gap)

                above_target_prev = max(prev_m_fp - rc.target_mass, 0.0)
                above_target_curr = max(m_fp - rc.target_mass, 0.0)
                excess_delta = max(above_target_curr - above_target_prev, 0.0)
                progress_reward -= rc.post_target_delta_penalty_weight * excess_delta
                productive_delta = max(prev_gap - curr_gap, 0.0)
                if prev_m_fp < rc.target_mass <= m_fp:
                    target_cross_reward = float(rc.target_cross_bonus)
            else:
                prev_deficit = max(rc.uf_conc_soft_low_limit - prev_c_uf, 0.0)
                curr_deficit = max(rc.uf_conc_soft_low_limit - c_uf, 0.0)
                progress_reward = rc.concentration_progress_weight * (prev_deficit - curr_deficit)

            energy_reward = -rc.energy_cost_weight * energy_cost

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

            if rc.unsafe_step_reward_block and unsafe_now:
                progress_reward = min(progress_reward, 0.0)

            dry_run_penalty = 0.0
            if dry_run_minutes > 0:
                dry_run_penalty -= rc.dry_run_penalty * float(dry_run_minutes)

            low_conc_penalty = 0.0
            if low_conc_minutes > 0:
                low_conc_penalty -= rc.uf_conc_low_penalty * float(low_conc_minutes)

            conc_quality_reward = 0.0
            if not minimal_mode:
                conc_quality_reward = _compute_concentration_guidance_reward(
                    rc,
                    c_uf=float(c_uf),
                    m_fp=float(m_fp),
                    episode_step=int(episode_step),
                    max_steps=int(max_steps),
                )

            idle_shutdown_reward = 0.0
            if (
                not minimal_mode
                and
                rc.idle_shutdown_bonus > 0.0
                and rc.enable_target_objective
                and m_fp >= rc.target_mass
                and v_buf <= rc.mixer_idle_volume_threshold
                and q_fp <= rc.idle_shutdown_q_fp_threshold
            ):
                idle_shutdown_reward += rc.idle_shutdown_bonus

            dry_run_flow_penalty = 0.0
            if not minimal_mode and rc.dry_run_flow_penalty_weight > 0.0:
                q_fp_norm = float(np.clip(q_fp / 70.0, 0.0, 1.0))
                low_buffer_gap = max(float(rc.dry_run_buffer_threshold) - v_buf, 0.0)
                low_buffer_gap = low_buffer_gap / max(float(rc.dry_run_buffer_threshold), 1e-6)
                dry_run_flow_penalty -= rc.dry_run_flow_penalty_weight * low_buffer_gap * (q_fp_norm ** 2)

            uf_flow_quality_penalty = 0.0
            if not minimal_mode and rc.uf_low_conc_flow_penalty_weight > 0.0:
                low_gap = max(float(rc.uf_conc_guidance_target) - c_uf, 0.0)
                q_uf_norm = float(np.clip(q_uf / 50.0, 0.0, 1.0))
                uf_flow_quality_penalty -= rc.uf_low_conc_flow_penalty_weight * (low_gap ** 2) * q_uf_norm

            smoothness_penalty = 0.0
            if not minimal_mode and rc.smoothness_weight > 0.0:
                delta_q_uf = float((q_uf - prev_q_uf) / 50.0)
                delta_q_fp = float((q_fp - prev_q_fp) / 70.0)
                smoothness_penalty -= rc.smoothness_weight * (delta_q_uf ** 2 + delta_q_fp ** 2)

            medium_constraint_now = bool(dry_run_minutes > 0 or low_conc_minutes > 0)
            if rc.constraint_step_reward_block and medium_constraint_now:
                progress_reward = min(progress_reward, 0.0)

            terminal_reward = 0.0
            avg_cuf_terminal_reward = 0.0
            is_final_step = episode_step >= max_steps - 1
            if is_final_step:
                done = True
                if rc.enable_target_objective:
                    deficit = max(rc.target_mass - m_fp, 0.0)
                    overshoot = max(m_fp - rc.target_mass_high, 0.0)
                    if deficit > 0.0:
                        terminal_reward -= rc.terminal_under_penalty_weight * deficit
                    else:
                        terminal_reward += rc.terminal_target_band_bonus
                        terminal_reward -= rc.terminal_over_penalty_weight * overshoot
                        if not unsafe_now:
                            avg_cuf_terminal_reward = _compute_terminal_average_cuf_reward(
                                rc,
                                episode_mean_c_uf=float(episode_mean_c_uf),
                            )
                            terminal_reward += avg_cuf_terminal_reward
                else:
                    terminal_reward += rc.terminal_target_band_bonus if c_uf >= rc.uf_conc_soft_low_limit else 0.0

                if unsafe_now:
                    terminal_reward -= rc.terminal_safety_block_penalty

            total_reward = (
                progress_reward
                + target_cross_reward
                + energy_reward
                + safety_penalty
                + dry_run_penalty
                + low_conc_penalty
                + conc_quality_reward
                + idle_shutdown_reward
                + dry_run_flow_penalty
                + uf_flow_quality_penalty
                + smoothness_penalty
                + terminal_reward
            )
            total_reward = float(np.clip(total_reward, rc.reward_clip_min, rc.reward_clip_max))

            breakdown = {
                "production": float(progress_reward),
                "productive_delta": float(productive_delta),
                "glide_delta": 0.0,
                "excess_delta": float(excess_delta),
                "target_gap": 0.0,
                "overshoot": float(-rc.post_target_delta_penalty_weight * excess_delta),
                "schedule": 0.0,
                "target_cross": float(target_cross_reward),
                "throughput": float(progress_reward),
                "band_hold": 0.0,
                "energy": float(energy_reward),
                "fp_usage": 0.0,
                "post_target_hold": float(dry_run_flow_penalty),
                "safety": float(safety_penalty),
                "dry_run": float(dry_run_penalty),
                "low_conc": float(low_conc_penalty),
                "conc_quality": float(conc_quality_reward),
                "idle_shutdown": float(idle_shutdown_reward),
                "uf_flow_quality": float(uf_flow_quality_penalty),
                "smooth": float(smoothness_penalty),
                "avg_cuf_terminal": float(avg_cuf_terminal_reward),
                "terminal": float(terminal_reward),
                "total": total_reward,
            }
            return total_reward, breakdown, done

        production_reward = 0.0
        productive_delta = 0.0
        glide_delta = 0.0
        excess_delta = 0.0
        target_cross_reward = 0.0

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
            if prev_m_fp < rc.target_mass <= m_fp:
                target_cross_reward = float(rc.target_cross_bonus)
        else:
            productive_delta = max(delta_m_fp, 0.0)
            production_reward = rc.throughput_reward_weight * productive_delta

        # 2) Energy is important, but still secondary to safety and end-of-day success.
        energy_reward = -rc.energy_cost_weight * energy_cost
        price_scale = float(current_price / max(max_price, 1e-6))
        q_fp_norm = float(np.clip(q_fp / 70.0, 0.0, 1.0))
        base_fp_usage_penalty = -rc.fp_usage_weight * (q_fp_norm ** 2) * price_scale
        post_target_fp_penalty = 0.0
        post_target_hold_penalty = 0.0
        post_target_q_uf_hold_penalty = 0.0
        if rc.enable_target_objective and m_fp >= rc.target_mass:
            post_target_fp_penalty -= rc.post_target_fp_usage_weight * (q_fp_norm ** 2) * price_scale
            if rc.post_target_buffer_hold_weight > 0.0:
                guard_level = max(rc.post_target_buffer_guard_level, 1e-6)
                # If the target is already met and the buffer is still far from full,
                # keeping the filter press on has little value. Relax this pressure
                # only when the buffer rises close to its upper range.
                hold_scale = float(np.clip((guard_level - v_buf) / guard_level, 0.0, 1.0))
                post_target_hold_penalty -= rc.post_target_buffer_hold_weight * hold_scale * (q_fp_norm ** 2)
            if rc.post_target_q_uf_hold_weight > 0.0:
                q_uf_guard_level = max(rc.post_target_q_uf_guard_level, 1e-6)
                q_uf_hold_scale = float(np.clip((q_uf_guard_level - v_buf) / q_uf_guard_level, 0.0, 1.0))
                q_uf_norm = float(np.clip(q_uf / 50.0, 0.0, 1.0))
                post_target_q_uf_hold_penalty -= rc.post_target_q_uf_hold_weight * q_uf_hold_scale * (q_uf_norm ** 2)
        fp_usage_penalty = (
            base_fp_usage_penalty
            + post_target_fp_penalty
            + post_target_hold_penalty
            + post_target_q_uf_hold_penalty
        )

        schedule_reward = 0.0
        if rc.enable_target_objective and max_steps > 0:
            expected_mass = rc.target_mass * min((episode_step + 1) / max_steps, 1.0)
            tolerance = rc.schedule_tolerance_ratio * rc.target_mass
            ahead_gap = max(m_fp - expected_mass - tolerance, 0.0)
            behind_gap = max(expected_mass - m_fp - tolerance, 0.0)
            schedule_reward = (
                -rc.schedule_ahead_weight * ahead_gap
                -rc.schedule_behind_weight * behind_gap
            )

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

        # Medium-severity constraint penalties:
        # these terms discourage poor operation quality, but are deliberately
        # weaker than hard-safety penalties so the reward hierarchy remains:
        # hard safety > soft quality / dry-run constraints > economics.
        dry_run_penalty = 0.0
        if dry_run_minutes > 0:
            dry_run_penalty -= rc.dry_run_penalty * float(dry_run_minutes)

        low_conc_penalty = 0.0
        if low_conc_minutes > 0:
            low_conc_penalty -= rc.uf_conc_low_penalty * float(low_conc_minutes)

        conc_quality_reward = _compute_concentration_guidance_reward(
            rc,
            c_uf=float(c_uf),
            m_fp=float(m_fp),
            episode_step=int(episode_step),
            max_steps=int(max_steps),
        )

        idle_shutdown_reward = 0.0
        if (
            rc.idle_shutdown_bonus > 0.0
            and rc.enable_target_objective
            and m_fp >= rc.target_mass
            and v_buf <= rc.mixer_idle_volume_threshold
            and q_fp <= rc.idle_shutdown_q_fp_threshold
        ):
            idle_shutdown_reward += rc.idle_shutdown_bonus

        uf_flow_quality_penalty = 0.0
        if rc.uf_low_conc_flow_penalty_weight > 0.0:
            low_gap = max(float(rc.uf_conc_guidance_target) - c_uf, 0.0)
            q_uf_norm = float(np.clip(q_uf / 50.0, 0.0, 1.0))
            uf_flow_quality_penalty -= rc.uf_low_conc_flow_penalty_weight * (low_gap ** 2) * q_uf_norm

        smoothness_penalty = 0.0
        if rc.smoothness_weight > 0.0:
            delta_q_uf = float((q_uf - prev_q_uf) / 50.0)
            delta_q_fp = float((q_fp - prev_q_fp) / 70.0)
            smoothness_penalty -= rc.smoothness_weight * (delta_q_uf ** 2 + delta_q_fp ** 2)

        # 4) Terminal objective:
        # final judgement is dominated by whether the episode lands inside the target band.
        terminal_reward = 0.0
        avg_cuf_terminal_reward = 0.0
        is_final_step = episode_step >= max_steps - 1
        if is_final_step:
            done = True
            if rc.enable_target_objective:
                deficit = max(rc.target_mass - m_fp, 0.0)
                total_overshoot = max(m_fp - rc.target_mass, 0.0)
                inband_overshoot = min(total_overshoot, max(rc.target_mass_high - rc.target_mass, 0.0))
                overshoot = max(m_fp - rc.target_mass_high, 0.0)

                if deficit > 0.0:
                    terminal_reward = (
                        -rc.terminal_under_penalty_weight * deficit
                        -rc.terminal_under_penalty_quadratic * deficit ** 2
                    )
                else:
                    terminal_reward = (
                        rc.terminal_target_band_bonus
                        - rc.terminal_inband_over_penalty_weight * inband_overshoot
                    )
                    if not unsafe_now:
                        avg_cuf_terminal_reward = _compute_terminal_average_cuf_reward(
                            rc,
                            episode_mean_c_uf=float(episode_mean_c_uf),
                        )
                        terminal_reward += avg_cuf_terminal_reward
                    if overshoot > 0.0:
                        terminal_reward -= (
                            rc.terminal_over_penalty_weight * overshoot
                            + rc.terminal_over_penalty_quadratic * overshoot ** 2
                        )

            if unsafe_now:
                terminal_reward -= rc.terminal_safety_block_penalty

        total_reward = (
            production_reward
            + target_cross_reward
            + energy_reward
            + fp_usage_penalty
            + schedule_reward
            + safety_penalty
            + dry_run_penalty
            + low_conc_penalty
            + conc_quality_reward
            + idle_shutdown_reward
            + uf_flow_quality_penalty
            + smoothness_penalty
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
            "schedule": float(schedule_reward),
            "target_cross": float(target_cross_reward),
            "throughput": float(production_reward),
            "band_hold": 0.0,
            "energy": float(energy_reward),
            "fp_usage": float(fp_usage_penalty),
            "post_target_hold": float(post_target_hold_penalty),
            "safety": float(safety_penalty),
            "dry_run": float(dry_run_penalty),
            "low_conc": float(low_conc_penalty),
            "conc_quality": float(conc_quality_reward),
            "idle_shutdown": float(idle_shutdown_reward),
            "uf_flow_quality": float(uf_flow_quality_penalty),
            "smooth": float(smoothness_penalty),
            "avg_cuf_terminal": float(avg_cuf_terminal_reward),
            "terminal": float(terminal_reward),
            "total": total_reward,
        }
        return total_reward, breakdown, done
