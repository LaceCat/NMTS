"""
浓密脱水过程 Gymnasium 环境

整合:
- 物理模型 (浓密机 + 搅拌罐 + 压滤)
- 奖励函数
- 分时电价
- 策略步/物理步双时间尺度

动作空间: action[0] = Q_uf (底流泵, [0,50]), action[1] = Q_fp (压滤泵, [0,70])
观测空间 (9 维): [C_uf, V_buf, C_aver, M_FP, Mass_buf, price, remaining_steps, Qf, Cf]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple

from .constants import (
    AVAILABLE_CONTROL_MINUTES,
    DEFAULT_CONTROL_STEPS,
    FP_BATCH_MASS,
    FP_DOWNTIME_MINUTES,
    STARTUP_FILL_MINUTES,
    TOTAL_PROCESS_MINUTES,
)
from .physics.thickener import (
    ThickenerModel,
    DEFAULT_THICKENER_STATE,
)
from .physics.buffer_press import BufferPressModel
from .reward.config import RewardConfig
from .reward.scheme import RewardScheme
from .reward.pricing import ElectricityPricing, PricingPresets


class ThickenerDewateringEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    VALID_ACTION_MODES = ("DD", "CD", "CC")
    BASE_OBS_DIM = 9
    ACTION_HISTORY_STEPS = 3
    STATE_HISTORY_STEPS = 3
    ACTION_HISTORY_DIM = ACTION_HISTORY_STEPS * 2
    STATE_HISTORY_DIM = STATE_HISTORY_STEPS * BASE_OBS_DIM
    OBSERVATION_DIM = BASE_OBS_DIM + ACTION_HISTORY_DIM + STATE_HISTORY_DIM

    def __init__(
        self,
        max_steps: int = DEFAULT_CONTROL_STEPS,
        decision_interval: int = 5,
        target_mass: float = 400.0,
        mode: str = "CC",
        uf_control_mode: str = "absolute",
        uf_delta_max: float = 5.0,
        q_fp_delta_max: Optional[float] = None,
        pricing: Optional[ElectricityPricing] = None,
        reward_config: Optional[RewardConfig] = None,
        feed_volatility: str = "normal",
        verbose: bool = False,
        total_minutes: int = TOTAL_PROCESS_MINUTES,
        startup_minutes: int = STARTUP_FILL_MINUTES,
        enable_fp_batching: bool = True,
        fp_batch_mass: float = FP_BATCH_MASS,
        fp_downtime_minutes: int = FP_DOWNTIME_MINUTES,
        enable_fp_edge_smoothing: bool = False,
        fp_edge_smoothing_steps: int = 3,
        fp_prestop_ramp_mass: float = 2.0,
        fp_prestop_ramp_floor: float = 0.88,
        fp_restart_ramp_minutes: int = 3,
        enable_post_target_fp_governor: bool = True,
        post_target_fp_guard_level: float = 24.0,
        enable_post_target_idle_seeker: bool = True,
        post_target_idle_q_uf_cap: float = 0.5,
        post_target_idle_q_fp_floor: float = 20.0,
        enable_low_buffer_fp_guard: bool = True,
        low_buffer_fp_threshold: float = 0.0,
        low_buffer_fp_max: float = 0.0,
        low_buffer_fp_guard_max_correction: float = 2.0,
        governor_total_correction_limit: float = -1.0,
        enable_midcourse_quality_governor: bool = True,
        midcourse_quality_start_mass: float = 115.0,
        midcourse_quality_c_uf_target: float = 0.745,
        midcourse_quality_q_uf_cap: float = 9.5,
        midcourse_quality_q_fp_floor: float = 21.5,
        midcourse_quality_buffer_min: float = 1.6,
        midcourse_quality_buffer_max: float = 18.0,
        enable_late_concentration_keeper: bool = True,
        late_concentration_window_minutes: int = 120,
        late_concentration_mass_gap_limit: float = 24.0,
        late_concentration_c_uf_target: float = 0.72,
        late_concentration_q_uf_cap: float = 18.0,
        late_concentration_q_fp_floor: float = 10.0,
        late_concentration_buffer_min: float = 2.0,
        late_concentration_buffer_max: float = 18.0,
        enable_late_target_compensator: bool = True,
        late_target_window_minutes: int = 175,
        late_target_mass_gap_limit: float = 21.0,
        late_target_c_uf_limit: float = 0.746,
        late_target_v_buf_limit: float = 24.0,
        late_target_q_uf_bias_max: float = 6.2,
        late_target_q_fp_bias_max: float = 12.8,
        late_target_q_fp_min_buffer: float = 1.0,
        mixer_idle_volume_threshold: float = 0.05,
        mixer_power_off_volume_threshold: float = 1.5,
        enable_buffer_zero_finisher: bool = True,
        buffer_zero_finish_threshold: float = 2.0,
        direct_q_fp_physical_only: bool = False,
        enable_q_fp_actual_slew_limit: bool = True,
        q_fp_actual_slew_up_per_minute: float = 9.0,
        q_fp_actual_slew_down_per_minute: float = 9.0,
        q_fp_actual_slew_restart_vbuf_limit: float = 2.5,
        enable_q_fp_actual_prestop_taper: bool = False,
        q_fp_actual_prestop_mass: float = 1.5,
        q_fp_actual_prestop_vbuf_limit: float = 2.5,
        q_fp_actual_prestop_floor_ratio: float = 0.25,
        q_fp_deadzone: float = 1.0,
        q_uf_deadzone: float = 0.25,
        uf_delta_deadzone: float = 0.15,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.decision_interval = max(1, decision_interval)
        self.total_minutes = int(total_minutes)
        self.startup_minutes = int(startup_minutes)
        self.available_control_minutes = self.total_minutes
        self.enable_fp_batching = bool(enable_fp_batching)
        self.fp_batch_mass = float(max(fp_batch_mass, 1e-6))
        self.fp_downtime_minutes = int(max(fp_downtime_minutes, 1))
        self.enable_fp_edge_smoothing = bool(enable_fp_edge_smoothing)
        self.fp_edge_smoothing_steps = int(max(fp_edge_smoothing_steps, 0))
        self.fp_prestop_ramp_mass = float(max(fp_prestop_ramp_mass, 0.0))
        self.fp_prestop_ramp_floor = float(np.clip(fp_prestop_ramp_floor, 0.0, 1.0))
        self.fp_restart_ramp_minutes = int(max(fp_restart_ramp_minutes, 0))
        if self.fp_edge_smoothing_steps <= 0 and self.fp_restart_ramp_minutes > 0:
            self.fp_edge_smoothing_steps = int(self.fp_restart_ramp_minutes)
        self.enable_post_target_fp_governor = bool(enable_post_target_fp_governor)
        self.post_target_fp_guard_level = float(max(post_target_fp_guard_level, 1e-6))
        self.enable_post_target_idle_seeker = bool(enable_post_target_idle_seeker)
        self.post_target_idle_q_uf_cap = float(np.clip(post_target_idle_q_uf_cap, 0.0, 50.0))
        self.post_target_idle_q_fp_floor = float(np.clip(post_target_idle_q_fp_floor, 0.0, 70.0))
        self.enable_low_buffer_fp_guard = bool(enable_low_buffer_fp_guard)
        self.low_buffer_fp_threshold = float(max(low_buffer_fp_threshold, 0.0))
        self.low_buffer_fp_max = float(np.clip(low_buffer_fp_max, 0.0, 70.0))
        self.low_buffer_fp_guard_max_correction = float(max(low_buffer_fp_guard_max_correction, 0.0))
        self.governor_total_correction_limit = float(governor_total_correction_limit)
        self.enable_midcourse_quality_governor = bool(enable_midcourse_quality_governor)
        self.midcourse_quality_start_mass = float(max(midcourse_quality_start_mass, 0.0))
        self.midcourse_quality_c_uf_target = float(np.clip(midcourse_quality_c_uf_target, 0.0, 1.0))
        self.midcourse_quality_q_uf_cap = float(np.clip(midcourse_quality_q_uf_cap, 0.0, 50.0))
        self.midcourse_quality_q_fp_floor = float(np.clip(midcourse_quality_q_fp_floor, 0.0, 70.0))
        self.midcourse_quality_buffer_min = float(max(midcourse_quality_buffer_min, 0.0))
        self.midcourse_quality_buffer_max = float(max(midcourse_quality_buffer_max, 0.0))
        self.enable_late_concentration_keeper = bool(enable_late_concentration_keeper)
        self.late_concentration_window_minutes = int(max(late_concentration_window_minutes, 1))
        self.late_concentration_mass_gap_limit = float(max(late_concentration_mass_gap_limit, 1e-6))
        self.late_concentration_c_uf_target = float(np.clip(late_concentration_c_uf_target, 0.0, 1.0))
        self.late_concentration_q_uf_cap = float(np.clip(late_concentration_q_uf_cap, 0.0, 50.0))
        self.late_concentration_q_fp_floor = float(np.clip(late_concentration_q_fp_floor, 0.0, 70.0))
        self.late_concentration_buffer_min = float(max(late_concentration_buffer_min, 0.0))
        self.late_concentration_buffer_max = float(max(late_concentration_buffer_max, 0.0))
        self.enable_late_target_compensator = bool(enable_late_target_compensator)
        self.late_target_window_minutes = int(max(late_target_window_minutes, 1))
        self.late_target_mass_gap_limit = float(max(late_target_mass_gap_limit, 1e-6))
        self.late_target_c_uf_limit = float(np.clip(late_target_c_uf_limit, 0.0, 1.0))
        self.late_target_v_buf_limit = float(max(late_target_v_buf_limit, 0.0))
        self.late_target_q_uf_bias_max = float(max(late_target_q_uf_bias_max, 0.0))
        self.late_target_q_fp_bias_max = float(max(late_target_q_fp_bias_max, 0.0))
        self.late_target_q_fp_min_buffer = float(max(late_target_q_fp_min_buffer, 0.0))
        self.mixer_idle_volume_threshold = float(max(mixer_idle_volume_threshold, 0.0))
        self.mixer_power_off_volume_threshold = float(
            max(mixer_power_off_volume_threshold, self.mixer_idle_volume_threshold)
        )
        self.enable_buffer_zero_finisher = bool(enable_buffer_zero_finisher)
        self.buffer_zero_finish_threshold = float(max(buffer_zero_finish_threshold, 0.0))
        self.direct_q_fp_physical_only = bool(direct_q_fp_physical_only)
        self.enable_q_fp_actual_slew_limit = bool(enable_q_fp_actual_slew_limit)
        self.q_fp_actual_slew_up_per_minute = float(max(q_fp_actual_slew_up_per_minute, 0.0))
        self.q_fp_actual_slew_down_per_minute = float(max(q_fp_actual_slew_down_per_minute, 0.0))
        self.q_fp_actual_slew_restart_vbuf_limit = float(max(q_fp_actual_slew_restart_vbuf_limit, 0.0))
        self.enable_q_fp_actual_prestop_taper = bool(enable_q_fp_actual_prestop_taper)
        self.q_fp_actual_prestop_mass = float(max(q_fp_actual_prestop_mass, 0.0))
        self.q_fp_actual_prestop_vbuf_limit = float(max(q_fp_actual_prestop_vbuf_limit, 0.0))
        self.q_fp_actual_prestop_floor_ratio = float(np.clip(q_fp_actual_prestop_floor_ratio, 0.0, 1.0))
        self.q_fp_deadzone = float(max(q_fp_deadzone, 0.0))
        self.q_uf_deadzone = float(max(q_uf_deadzone, 0.0))
        self.uf_delta_deadzone = float(max(uf_delta_deadzone, 0.0))
        self.mode = str(mode).upper()
        if self.mode not in self.VALID_ACTION_MODES:
            raise ValueError(f"Unsupported action mode: {mode}. Expected one of {self.VALID_ACTION_MODES}.")
        self.uf_control_mode = str(uf_control_mode).lower()
        if self.uf_control_mode not in ("absolute", "delta"):
            raise ValueError("uf_control_mode must be 'absolute' or 'delta'.")
        self.uf_delta_max = float(max(uf_delta_max, 1e-6))
        self.q_fp_delta_max = None if q_fp_delta_max is None or q_fp_delta_max < 0 else float(q_fp_delta_max)

        self.thickener = ThickenerModel()
        self.buffer_press = BufferPressModel()

        self.pricing = pricing if pricing is not None else PricingPresets.daily_24h()

        if reward_config is not None:
            self.reward_config = reward_config
        else:
            self.reward_config = RewardConfig(target_mass=target_mass, max_steps=max_steps)
        self.reward_scheme = RewardScheme(self.reward_config)

        self.feed_volatility = feed_volatility

        action_low = np.array([0.0, 0.0], dtype=np.float32)
        action_high = np.array([50.0, 70.0], dtype=np.float32)
        if self.mode != "DD" and self.uf_control_mode == "delta":
            action_low[0] = -self.uf_delta_max
            action_high[0] = self.uf_delta_max
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(2,),
            dtype=np.float32,
        )
        self.physical_action_space = self._build_physical_action_space()

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.OBSERVATION_DIM,), dtype=np.float32
        )

        self.verbose = verbose
        self._init_state()

    def _init_state(self):
        self.Q_uf = 0.0
        self.Q_fp = 0.0
        self.Qf = 40.0
        self.Cf = 0.35
        self.thickener_state = DEFAULT_THICKENER_STATE.copy()
        self.v_buf = 0.0
        self.m_fp = 0.0
        self.c_aver = 0.0
        self.timecnt = 0
        self.policy_stepcnt = 0
        self.energy_cost_sum = 0.0
        self.episode_c_uf_sum = 0.0
        self.episode_c_uf_count = 0
        self.target_reached = False
        self.warmup_minutes_used = 0

        self.prev_q_uf = 0.0
        self.prev_q_fp = 0.0
        self.prev_v_buf = 0.0
        self.prev_c_aver = 0.0
        self.prev_m_fp = 0.0
        self.last_c_uf = float(self.thickener.d2c(DEFAULT_THICKENER_STATE[-1] / 1e6))
        self.action_history = [(0.0, 0.0) for _ in range(self.ACTION_HISTORY_STEPS)]
        self.state_history = [
            np.zeros(self.BASE_OBS_DIM, dtype=np.float32)
            for _ in range(self.STATE_HISTORY_STEPS)
        ]
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_applied_q_uf_delta = 0.0
        self.last_delta_m_fp_step = 0.0
        self.last_actual_q_fp_minute = 0.0
        self.fp_busy = False
        self.fp_downtime_remain = 0
        self.fp_cycle_mass = 0.0
        self.fp_total_cycles = 0
        self.fp_restart_ramp_remain = 0
        self.last_fp_batch_triggered = False
        self.last_fp_prestop_ramped = False
        self.last_fp_restart_ramped = False
        self.last_post_target_fp_governed = False
        self.last_post_target_idle_seeking = False
        self.last_low_buffer_fp_guarded = False
        self.last_midcourse_quality_governed = False
        self.last_late_concentration_kept = False
        self.last_late_target_compensated = False
        self.last_late_target_q_uf_bias = 0.0
        self.last_late_target_q_fp_bias = 0.0
        self.last_buffer_zero_finished = False
        self.last_buffer_zero_finish_q_fp = 0.0
        self.last_soft_teacher_q_uf = 0.0
        self.last_soft_teacher_q_fp = 0.0
        self.last_soft_teacher_active = False
        self.last_soft_teacher_flags = {}

    def _build_physical_action_space(self):
        if self.mode == "DD":
            return spaces.MultiDiscrete([2, 2])
        if self.mode == "CD":
            return spaces.Tuple(
                (
                    spaces.Box(
                        low=np.array([0.0], dtype=np.float32),
                        high=np.array([50.0], dtype=np.float32),
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    spaces.Discrete(2),
                )
            )
        return spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 70.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    @staticmethod
    def _discrete_on_off(value: float, on_value: float) -> float:
        if -0.5 <= value <= 1.5:
            return float(on_value if int(round(value)) > 0 else 0.0)
        return float(on_value if value >= (on_value / 2.0) else 0.0)

    def _resolve_physical_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"Expected 2 action values, got shape {np.asarray(action).shape}")

        if self.mode == "DD":
            q_uf = self._discrete_on_off(float(arr[0]), 50.0)
            q_fp = self._discrete_on_off(float(arr[1]), 70.0)
        elif self.mode == "CD":
            if self.uf_control_mode == "delta":
                requested_delta = float(np.clip(arr[0], -self.uf_delta_max, self.uf_delta_max))
                if abs(requested_delta) < self.uf_delta_deadzone:
                    requested_delta = 0.0
                q_uf = float(np.clip(self.Q_uf + requested_delta, 0.0, 50.0))
            else:
                q_uf = float(np.clip(arr[0], 0.0, 50.0))
                if q_uf < self.q_uf_deadzone:
                    q_uf = 0.0
            q_fp = self._discrete_on_off(float(arr[1]), 70.0)
        else:
            if self.uf_control_mode == "delta":
                requested_delta = float(np.clip(arr[0], -self.uf_delta_max, self.uf_delta_max))
                if abs(requested_delta) < self.uf_delta_deadzone:
                    requested_delta = 0.0
                q_uf = float(np.clip(self.Q_uf + requested_delta, 0.0, 50.0))
            else:
                q_uf = float(np.clip(arr[0], 0.0, 50.0))
                if q_uf < self.q_uf_deadzone:
                    q_uf = 0.0
            q_fp = float(np.clip(arr[1], 0.0, 70.0))
            if (not self._direct_q_fp_physical_only_enabled()) and self.q_fp_delta_max is not None:
                q_fp = float(
                    np.clip(
                        q_fp,
                        max(0.0, self.Q_fp - self.q_fp_delta_max),
                        min(70.0, self.Q_fp + self.q_fp_delta_max),
                    )
                )
            if (not self._direct_q_fp_physical_only_enabled()) and q_fp < self.q_fp_deadzone:
                q_fp = 0.0

        return np.array([q_uf, q_fp], dtype=np.float32)

    def _direct_q_fp_physical_only_enabled(self) -> bool:
        return bool(self.direct_q_fp_physical_only and self.mode == "CC")

    def _sample_feed_conditions(self):
        if self.feed_volatility == "high":
            self.Qf = float(np.random.uniform(35, 50))
            self.Cf = float(np.random.uniform(0.30, 0.45))
        else:
            self.Qf = float(np.random.uniform(40, 45))
            self.Cf = float(np.random.uniform(0.35, 0.40))

    def _compute_mass_buf(self) -> float:
        if self.v_buf <= self.mixer_idle_volume_threshold:
            return 0.0
        return float(self.c_aver * self.v_buf * self.thickener.c2d(self.c_aver))

    def _make_base_obs(self, c_uf: float, mass_buf: float) -> list[float]:
        price = self.pricing.get_price(self.timecnt)
        remaining = max(self.max_steps - self.policy_stepcnt, 0)
        return [
            c_uf, self.v_buf, self.c_aver, self.m_fp,
            mass_buf, price, remaining,
            self.Qf, self.Cf,
        ]

    def _push_state_history(self, base_obs: list[float]) -> None:
        self.state_history.pop(0)
        self.state_history.append(np.asarray(base_obs, dtype=np.float32).copy())

    def _make_obs(self, c_uf: float, mass_buf: float) -> np.ndarray:
        base_obs = self._make_base_obs(c_uf, mass_buf)
        hist_obs = []
        for q_uf_hist, q_fp_hist in self.action_history:
            hist_obs.extend([q_uf_hist, q_fp_hist])
        state_hist_obs = []
        for hist_state in self.state_history:
            state_hist_obs.extend(hist_state.tolist())
        return np.array(base_obs + hist_obs + state_hist_obs, dtype=np.float32)

    def _update_fp_batch_state(self, delta_m_fp: float) -> bool:
        if not self.enable_fp_batching:
            return False

        if self.fp_busy:
            self.fp_downtime_remain = max(self.fp_downtime_remain - 1, 0)
            if self.fp_downtime_remain == 0:
                self.fp_busy = False
                self.fp_restart_ramp_remain = self.fp_edge_smoothing_steps
            return False

        if delta_m_fp > 0.0:
            self.fp_cycle_mass += float(delta_m_fp)

        if self.fp_cycle_mass + 1e-9 >= self.fp_batch_mass:
            self.fp_cycle_mass = max(self.fp_cycle_mass - self.fp_batch_mass, 0.0)
            self.fp_busy = True
            self.fp_downtime_remain = self.fp_downtime_minutes
            self.fp_total_cycles += 1
            self.fp_restart_ramp_remain = 0
            return True

        return False

    def _apply_fp_batch_edge_ramps(self, q_fp: float) -> Tuple[float, bool, bool]:
        """
        Smooth the batch-state on/off edges to avoid a hard jump in the filter
        press flow when a 20 t cycle ends or when the downtime finishes.
        """
        ramped_q_fp = float(max(q_fp, 0.0))
        prestop_ramped = False
        restart_ramped = False

        if not self.enable_fp_batching or not self.enable_fp_edge_smoothing or self.fp_busy:
            return ramped_q_fp, prestop_ramped, restart_ramped

        remaining_cycle_mass = max(self.fp_batch_mass - self.fp_cycle_mass, 0.0)
        recent_step_mass = float(max(self.last_delta_m_fp_step, 1e-6))
        estimated_steps_to_batch = float(remaining_cycle_mass / recent_step_mass) if recent_step_mass > 1e-6 else 999.0
        smoothing_steps = int(max(self.fp_edge_smoothing_steps, 0))
        if 0.0 < remaining_cycle_mass < self.fp_batch_mass:
            prestop_scale = 1.0
            if smoothing_steps > 0 and estimated_steps_to_batch <= float(smoothing_steps):
                if smoothing_steps <= 1:
                    prestop_scale = float(self.fp_prestop_ramp_floor)
                else:
                    normalized_step = float(
                        np.clip((estimated_steps_to_batch - 1.0) / max(float(smoothing_steps - 1), 1.0), 0.0, 1.0)
                    )
                    prestop_scale = float(
                        self.fp_prestop_ramp_floor
                        + (1.0 - self.fp_prestop_ramp_floor) * normalized_step
                    )
            elif self.fp_prestop_ramp_mass > 1e-9 and remaining_cycle_mass < self.fp_prestop_ramp_mass:
                remaining_ratio = float(np.clip(remaining_cycle_mass / self.fp_prestop_ramp_mass, 0.0, 1.0))
                prestop_scale = float(
                    np.clip(
                        self.fp_prestop_ramp_floor
                        + (1.0 - self.fp_prestop_ramp_floor) * np.sqrt(remaining_ratio),
                        0.0,
                        1.0,
                    )
                )
            if prestop_scale < 1.0:
                prestop_target_q_fp = ramped_q_fp * prestop_scale
                if prestop_target_q_fp + 1e-6 < ramped_q_fp:
                    ramped_q_fp = float(prestop_target_q_fp)
                    prestop_ramped = True

        if self.fp_restart_ramp_remain > 0 and smoothing_steps > 0:
            restart_linear_progress = float(
                np.clip(
                    (smoothing_steps - self.fp_restart_ramp_remain + 1)
                    / max(smoothing_steps, 1),
                    0.0,
                    1.0,
                )
            )
            restart_progress = float(restart_linear_progress)
            restart_cap_q_fp = float(ramped_q_fp * restart_progress)
            if restart_cap_q_fp + 1e-6 < ramped_q_fp:
                ramped_q_fp = restart_cap_q_fp
                restart_ramped = True

        return ramped_q_fp, prestop_ramped, restart_ramped

    def _apply_q_fp_actual_slew_limit(self, target_actual_q_fp: float) -> Tuple[float, bool]:
        """
        Limit the minute-to-minute change of the physically executed filter
        press flow so actuator restart edges are smoother than the scheduled
        command without globally slowing the whole episode.
        """
        target_actual_q_fp = float(max(target_actual_q_fp, 0.0))

        if self.fp_busy:
            self.last_actual_q_fp_minute = 0.0
            return 0.0, False

        restart_edge_active = bool(
            self.fp_restart_ramp_remain > 0
            and self.v_buf <= self.q_fp_actual_slew_restart_vbuf_limit + 1e-6
        )
        if (not self.enable_q_fp_actual_slew_limit) or (not restart_edge_active):
            self.last_actual_q_fp_minute = target_actual_q_fp
            return target_actual_q_fp, False

        prev_actual_q_fp = float(max(self.last_actual_q_fp_minute, 0.0))
        lower_bound = max(0.0, prev_actual_q_fp - self.q_fp_actual_slew_down_per_minute)
        upper_bound = prev_actual_q_fp + self.q_fp_actual_slew_up_per_minute
        limited_actual_q_fp = float(np.clip(target_actual_q_fp, lower_bound, upper_bound))
        slew_limited = bool(abs(limited_actual_q_fp - target_actual_q_fp) > 1e-6)
        self.last_actual_q_fp_minute = limited_actual_q_fp
        return limited_actual_q_fp, slew_limited

    def _apply_q_fp_actual_prestop_taper(self, target_actual_q_fp: float) -> Tuple[float, bool]:
        """
        Taper the actual filter-press flow before a batch shutdown so the
        nonzero->0 edge is softened ahead of the mandatory busy downtime.
        """
        target_actual_q_fp = float(max(target_actual_q_fp, 0.0))
        if (
            not self.enable_q_fp_actual_prestop_taper
            or self.fp_busy
            or self.q_fp_actual_prestop_mass <= 1e-9
            or self.v_buf > self.q_fp_actual_prestop_vbuf_limit + 1e-6
        ):
            return target_actual_q_fp, False

        remaining_cycle_mass = max(self.fp_batch_mass - self.fp_cycle_mass, 0.0)
        if remaining_cycle_mass <= 0.0 or remaining_cycle_mass >= self.q_fp_actual_prestop_mass:
            return target_actual_q_fp, False

        remaining_ratio = float(np.clip(remaining_cycle_mass / self.q_fp_actual_prestop_mass, 0.0, 1.0))
        taper_ratio = float(
            np.clip(
                self.q_fp_actual_prestop_floor_ratio
                + (1.0 - self.q_fp_actual_prestop_floor_ratio) * np.sqrt(remaining_ratio),
                0.0,
                1.0,
            )
        )
        tapered_actual_q_fp = float(target_actual_q_fp * taper_ratio)
        tapered = bool(tapered_actual_q_fp + 1e-6 < target_actual_q_fp)
        return tapered_actual_q_fp, tapered

    def _apply_post_target_fp_governor(self, q_fp: float) -> Tuple[float, bool]:
        """
        Tail-only governor for the filter press.

        Design intent:
        - do not alter the policy before the production target is met;
        - once 400 t is already finished, keep Q_fp off while buffer inventory
          is still comfortably below a high guard level;
        - if the buffer rises near the guard level, release control to avoid
          turning an economic preference into a safety risk.
        """
        if not self.enable_post_target_fp_governor:
            return float(q_fp), False
        if self.enable_post_target_idle_seeker and self.m_fp >= self.reward_config.target_mass:
            return float(q_fp), False
        if self.m_fp < self.reward_config.target_mass:
            return float(q_fp), False
        if self.v_buf >= self.post_target_fp_guard_level:
            return float(q_fp), False
        return 0.0, bool(q_fp > 1e-6)

    def _is_post_target_idle_mode(self) -> bool:
        return bool(
            self.enable_post_target_idle_seeker
            and self.m_fp >= self.reward_config.target_mass
        )

    def _apply_post_target_idle_seeker(self, q_uf: float, q_fp: float) -> Tuple[float, float, bool]:
        """
        After the target mass is reached, preserve higher concentration by
        capping underflow and actively clear residual buffer inventory down to
        the true mixer-idle threshold.
        """
        if not self._is_post_target_idle_mode():
            return float(q_uf), float(q_fp), False

        if self.v_buf <= self.mixer_idle_volume_threshold + 1e-6:
            shutdown_q_uf = 0.0
            shutdown_q_fp = 0.0
            changed = bool(abs(shutdown_q_uf - q_uf) > 1e-6 or abs(shutdown_q_fp - q_fp) > 1e-6)
            return shutdown_q_uf, shutdown_q_fp, changed

        capped_q_uf = float(min(max(q_uf, 0.0), self.post_target_idle_q_uf_cap))
        drained_q_fp = float(max(q_fp, self.post_target_idle_q_fp_floor))
        if self.q_fp_delta_max is not None:
            drained_q_fp = float(
                np.clip(
                    drained_q_fp,
                    max(0.0, self.Q_fp - self.q_fp_delta_max),
                    min(70.0, self.Q_fp + self.q_fp_delta_max),
                )
            )

        changed = bool(abs(capped_q_uf - q_uf) > 1e-6 or abs(drained_q_fp - q_fp) > 1e-6)
        return capped_q_uf, drained_q_fp, changed

    def _active_low_buffer_threshold(self) -> float:
        if self._is_post_target_idle_mode():
            return float(self.mixer_idle_volume_threshold)
        return float(self.low_buffer_fp_threshold)

    def _active_dry_run_threshold(self) -> float:
        if self._is_post_target_idle_mode():
            return 0.0
        return float(self.reward_config.dry_run_buffer_threshold)

    def _apply_low_buffer_fp_guard(self, q_fp: float) -> Tuple[float, bool]:
        """
        Hard safety/equipment guard for near-empty buffer operation.

        When the buffer volume is already very low, allowing the policy to keep
        commanding the filter press produces repeated dry-run minutes that are
        easy for SAC to stumble into during exploration. This guard only clips
        Q_fp in that narrow region and leaves the rest of the trajectory to the
        policy.
        """
        if not self.enable_low_buffer_fp_guard:
            return float(q_fp), False
        if self.v_buf >= self._active_low_buffer_threshold():
            return float(q_fp), False

        guarded_floor = float(max(float(q_fp) - self.low_buffer_fp_guard_max_correction, 0.0))
        guarded_q_fp = float(min(max(float(q_fp), 0.0), max(self.low_buffer_fp_max, guarded_floor)))
        return guarded_q_fp, bool(guarded_q_fp + 1e-6 < float(q_fp))

    def _apply_buffer_zero_finisher(
        self,
        q_uf: float,
        q_fp: float,
        minutes_this_step: int,
    ) -> Tuple[float, bool]:
        """
        If the projected end-of-step residual buffer volume is already very
        small, proactively raise Q_fp just enough so the current control step
        can finish with an empty buffer.
        """
        if not self.enable_buffer_zero_finisher:
            return float(q_fp), False
        if minutes_this_step <= 0 or self.fp_busy:
            return float(q_fp), False

        projected_residual = float(
            self.v_buf + minutes_this_step * (float(q_uf) - float(q_fp)) / 60.0
        )
        if not (0.0 < projected_residual <= self.buffer_zero_finish_threshold):
            return float(q_fp), False

        required_q_fp = float(q_uf) + 60.0 * float(self.v_buf) / max(int(minutes_this_step), 1)
        finished_q_fp = float(np.clip(max(float(q_fp), required_q_fp), 0.0, 70.0))
        return finished_q_fp, bool(finished_q_fp > float(q_fp) + 1e-6)

    def _cap_governor_total_adjustment(
        self,
        base_q_uf: float,
        base_q_fp: float,
        target_q_uf: float,
        target_q_fp: float,
        budget: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """
        Cap the total governor-induced change in one step using an L1 budget:
            |ΔQ_uf| + |ΔQ_fp| <= budget
        """
        if budget is None:
            budget = self.governor_total_correction_limit
        budget = float(budget)

        base_q_uf = float(base_q_uf)
        base_q_fp = float(base_q_fp)
        target_q_uf = float(np.clip(target_q_uf, 0.0, 50.0))
        target_q_fp = float(np.clip(target_q_fp, 0.0, 70.0))

        if budget < 0.0:
            return target_q_uf, target_q_fp, False

        delta_q_uf = float(target_q_uf - base_q_uf)
        delta_q_fp = float(target_q_fp - base_q_fp)
        total_delta = abs(delta_q_uf) + abs(delta_q_fp)
        if total_delta <= budget + 1e-9:
            return target_q_uf, target_q_fp, False
        if total_delta <= 1e-9 or budget <= 1e-9:
            return base_q_uf, base_q_fp, total_delta > 1e-9

        scale = float(budget / total_delta)
        capped_q_uf = float(np.clip(base_q_uf + delta_q_uf * scale, 0.0, 50.0))
        capped_q_fp = float(np.clip(base_q_fp + delta_q_fp * scale, 0.0, 70.0))
        return capped_q_uf, capped_q_fp, True

    def _compute_soft_teacher_action(
        self,
        commanded_q_uf: float,
        commanded_q_fp: float,
        minutes_this_step: int,
    ) -> Tuple[float, float, bool, Dict[str, bool]]:
        """
        Compute the action that the full soft-governor stack would prefer for
        the current state, without actually forcing the environment to execute
        it. This serves as the imitation-learning teacher.
        """
        saved_flags = {
            "enable_post_target_fp_governor": bool(self.enable_post_target_fp_governor),
            "enable_post_target_idle_seeker": bool(self.enable_post_target_idle_seeker),
            "enable_midcourse_quality_governor": bool(self.enable_midcourse_quality_governor),
            "enable_late_concentration_keeper": bool(self.enable_late_concentration_keeper),
            "enable_late_target_compensator": bool(self.enable_late_target_compensator),
            "enable_buffer_zero_finisher": bool(self.enable_buffer_zero_finisher),
        }
        teacher_budget_capped = False
        try:
            self.enable_post_target_fp_governor = True
            self.enable_post_target_idle_seeker = True
            self.enable_midcourse_quality_governor = True
            self.enable_late_concentration_keeper = True
            self.enable_late_target_compensator = True
            self.enable_buffer_zero_finisher = True

            teacher_q_uf, teacher_q_fp, teacher_midcourse = self._apply_midcourse_quality_governor(
                commanded_q_uf,
                commanded_q_fp,
            )
            (
                teacher_q_uf,
                teacher_q_fp,
                teacher_late_target,
                _teacher_q_uf_bias,
                _teacher_q_fp_bias,
            ) = self._apply_late_target_compensator(teacher_q_uf, teacher_q_fp)
            teacher_q_uf, teacher_q_fp, teacher_late_concentration = self._apply_late_concentration_keeper(
                teacher_q_uf,
                teacher_q_fp,
            )
            teacher_q_uf, teacher_q_fp, teacher_idle_seek = self._apply_post_target_idle_seeker(
                teacher_q_uf,
                teacher_q_fp,
            )
            teacher_q_fp, teacher_post_target_fp = self._apply_post_target_fp_governor(teacher_q_fp)
            teacher_q_fp, teacher_zero_finish = self._apply_buffer_zero_finisher(
                teacher_q_uf,
                teacher_q_fp,
                minutes_this_step,
            )
            teacher_q_uf, teacher_q_fp, teacher_budget_capped = self._cap_governor_total_adjustment(
                commanded_q_uf,
                commanded_q_fp,
                teacher_q_uf,
                teacher_q_fp,
            )
        finally:
            for key, value in saved_flags.items():
                setattr(self, key, value)

        teacher_active = bool(
            abs(float(teacher_q_uf) - float(commanded_q_uf)) > 1e-6
            or abs(float(teacher_q_fp) - float(commanded_q_fp)) > 1e-6
        )
        flags = {
            "teacher_midcourse_quality": bool(teacher_midcourse),
            "teacher_late_target": bool(teacher_late_target),
            "teacher_late_concentration": bool(teacher_late_concentration),
            "teacher_post_target_idle": bool(teacher_idle_seek),
            "teacher_post_target_fp": bool(teacher_post_target_fp),
            "teacher_buffer_zero_finish": bool(teacher_zero_finish),
            "teacher_budget_capped": bool(teacher_budget_capped),
        }
        return float(teacher_q_uf), float(teacher_q_fp), teacher_active, flags

    def _apply_late_target_compensator(self, q_uf: float, q_fp: float) -> Tuple[float, float, bool, float, float]:
        """
        Very conservative tail-only helper.

        It only acts in the last part of the day when the controller is still
        slightly below the target, and it adds only a small bias instead of
        replacing the policy. This preserves the learned main trajectory while
        helping 399.x-t policies step over the 400-t line.
        """
        if not self.enable_late_target_compensator:
            return float(q_uf), float(q_fp), False, 0.0, 0.0

        remaining_minutes = max(self.total_minutes - self.timecnt, 0)
        mass_gap = max(float(self.reward_config.target_mass) - float(self.m_fp), 0.0)
        if remaining_minutes > self.late_target_window_minutes or mass_gap <= 1e-6:
            return float(q_uf), float(q_fp), False, 0.0, 0.0
        if mass_gap > self.late_target_mass_gap_limit:
            return float(q_uf), float(q_fp), False, 0.0, 0.0
        if self.last_c_uf >= self.late_target_c_uf_limit:
            return float(q_uf), float(q_fp), False, 0.0, 0.0
        if self.v_buf >= self.late_target_v_buf_limit:
            return float(q_uf), float(q_fp), False, 0.0, 0.0

        gap_ratio = float(np.clip(mass_gap / self.late_target_mass_gap_limit, 0.0, 1.0))
        # Start gently at the window entrance and get stronger near the end.
        time_ratio = float(
            np.clip(
                (self.late_target_window_minutes - remaining_minutes) / max(self.late_target_window_minutes, 1),
                0.0,
                1.0,
            )
        )
        urgency = max(0.2, gap_ratio) * (0.35 + 0.65 * time_ratio)

        q_uf_bias = self.late_target_q_uf_bias_max * urgency
        compensated_q_uf = float(np.clip(q_uf + q_uf_bias, 0.0, 50.0))

        q_fp_bias = 0.0
        compensated_q_fp = float(q_fp)
        if (
            (not self.fp_busy)
            and self.v_buf >= max(self.low_buffer_fp_threshold, self.late_target_q_fp_min_buffer)
        ):
            q_fp_bias = self.late_target_q_fp_bias_max * urgency
            compensated_q_fp = float(np.clip(q_fp + q_fp_bias, 0.0, 70.0))
            if self.q_fp_delta_max is not None:
                compensated_q_fp = float(
                    np.clip(
                        compensated_q_fp,
                        max(0.0, self.Q_fp - self.q_fp_delta_max),
                        min(70.0, self.Q_fp + self.q_fp_delta_max),
                    )
                )

        compensated = bool(compensated_q_uf > q_uf + 1e-6 or compensated_q_fp > q_fp + 1e-6)
        return compensated_q_uf, compensated_q_fp, compensated, float(compensated_q_uf - q_uf), float(compensated_q_fp - q_fp)

    def _apply_late_concentration_keeper(self, q_uf: float, q_fp: float) -> Tuple[float, float, bool]:
        """
        In the late pre-target phase, use existing buffer inventory first and
        avoid over-pulling dilute underflow if the concentration is still below
        the desired high-quality band.
        """
        if not self.enable_late_concentration_keeper:
            return float(q_uf), float(q_fp), False

        remaining_minutes = max(self.total_minutes - self.timecnt, 0)
        mass_gap = max(float(self.reward_config.target_mass) - float(self.m_fp), 0.0)
        if remaining_minutes > self.late_concentration_window_minutes:
            return float(q_uf), float(q_fp), False
        if mass_gap <= 1e-6 or mass_gap > self.late_concentration_mass_gap_limit:
            return float(q_uf), float(q_fp), False
        if self.last_c_uf >= self.late_concentration_c_uf_target:
            return float(q_uf), float(q_fp), False
        if not (self.late_concentration_buffer_min <= self.v_buf <= self.late_concentration_buffer_max):
            return float(q_uf), float(q_fp), False

        kept_q_uf = float(min(max(q_uf, 0.0), self.late_concentration_q_uf_cap))
        kept_q_fp = float(q_fp)
        if not self.fp_busy:
            kept_q_fp = float(max(q_fp, self.late_concentration_q_fp_floor))
            if self.q_fp_delta_max is not None:
                kept_q_fp = float(
                    np.clip(
                        kept_q_fp,
                        max(0.0, self.Q_fp - self.q_fp_delta_max),
                        min(70.0, self.Q_fp + self.q_fp_delta_max),
                    )
                )
        changed = bool(abs(kept_q_uf - q_uf) > 1e-6 or abs(kept_q_fp - q_fp) > 1e-6)
        return kept_q_uf, kept_q_fp, changed

    def _apply_midcourse_quality_governor(self, q_uf: float, q_fp: float) -> Tuple[float, float, bool]:
        """
        Mid/late episode quality governor.

        Once production has already progressed far enough, gently bias the
        policy toward a slightly denser operating region by capping Q_uf and
        ensuring the buffer inventory is consumed at a reasonable pace through
        Q_fp. This is intentionally weaker than the late-target compensator and
        should never violate higher-priority safety or dry-run guards.
        """
        if not self.enable_midcourse_quality_governor:
            return float(q_uf), float(q_fp), False
        if self.m_fp < self.midcourse_quality_start_mass:
            return float(q_uf), float(q_fp), False
        if self.m_fp >= float(self.reward_config.target_mass):
            return float(q_uf), float(q_fp), False
        if self.last_c_uf >= self.midcourse_quality_c_uf_target:
            return float(q_uf), float(q_fp), False
        if self.fp_busy:
            return float(q_uf), float(q_fp), False
        if not (self.midcourse_quality_buffer_min <= self.v_buf <= self.midcourse_quality_buffer_max):
            return float(q_uf), float(q_fp), False

        governed_q_uf = float(min(max(q_uf, 0.0), self.midcourse_quality_q_uf_cap))
        governed_q_fp = float(max(q_fp, self.midcourse_quality_q_fp_floor))
        if self.q_fp_delta_max is not None:
            governed_q_fp = float(
                np.clip(
                    governed_q_fp,
                    max(0.0, self.Q_fp - self.q_fp_delta_max),
                    min(70.0, self.Q_fp + self.q_fp_delta_max),
                )
            )

        changed = bool(abs(governed_q_uf - q_uf) > 1e-6 or abs(governed_q_fp - q_fp) > 1e-6)
        return governed_q_uf, governed_q_fp, changed

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._init_state()
        self._sample_feed_conditions()

        self.thickener_state = DEFAULT_THICKENER_STATE.copy()
        self.last_c_uf = float(self.thickener.d2c(self.thickener_state[-1] / 1e6))
        self.Q_uf = 0.0
        self.Q_fp = 0.0
        self.prev_q_uf = 0.0
        self.prev_q_fp = 0.0
        self.last_actual_q_fp_minute = 0.0
        self.timecnt = 0
        self.policy_stepcnt = 0
        self.m_fp = 0.0
        self.v_buf = 0.0
        self.c_aver = self.last_c_uf
        self.prev_v_buf = self.v_buf
        self.prev_c_aver = self.c_aver
        self.prev_m_fp = 0.0
        self.energy_cost_sum = 0.0
        self.episode_c_uf_sum = 0.0
        self.episode_c_uf_count = 0
        self.target_reached = False
        self.action_history = [(0.0, 0.0) for _ in range(self.ACTION_HISTORY_STEPS)]
        self.warmup_minutes_used = 0
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_applied_q_uf_delta = 0.0
        self.fp_busy = False
        self.fp_downtime_remain = 0
        self.fp_cycle_mass = 0.0
        self.fp_total_cycles = 0
        self.fp_restart_ramp_remain = 0
        self.last_fp_batch_triggered = False
        self.last_fp_prestop_ramped = False
        self.last_fp_restart_ramped = False

        c_uf = self.last_c_uf
        mass_buf = self._compute_mass_buf()
        base_obs = self._make_base_obs(c_uf, mass_buf)
        self.state_history = [
            np.asarray(base_obs, dtype=np.float32).copy()
            for _ in range(self.STATE_HISTORY_STEPS)
        ]
        obs = self._make_obs(c_uf, mass_buf)
        return obs, {}

    def step(self, action: np.ndarray):
        current_mass_buf = self._compute_mass_buf()
        current_base_obs = self._make_base_obs(self.last_c_uf, current_mass_buf)
        self._push_state_history(current_base_obs)

        prev_q_uf_cmd = float(self.Q_uf)
        self.last_raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        physical_action = self._resolve_physical_action(self.last_raw_action)
        commanded_q_uf = float(physical_action[0])
        commanded_q_fp = float(physical_action[1])
        minutes_left = max(self.total_minutes - self.timecnt, 0)
        minutes_this_step = min(self.decision_interval, minutes_left)
        (
            teacher_q_uf,
            teacher_q_fp,
            soft_teacher_active,
            soft_teacher_flags,
        ) = self._compute_soft_teacher_action(
            commanded_q_uf,
            commanded_q_fp,
            minutes_this_step,
        )
        governed_q_uf, governed_q_fp, midcourse_quality_governed = self._apply_midcourse_quality_governor(
            commanded_q_uf,
            commanded_q_fp,
        )
        (
            compensated_q_uf,
            compensated_q_fp,
            late_target_compensated,
            q_uf_comp_bias,
            q_fp_comp_bias,
        ) = self._apply_late_target_compensator(governed_q_uf, governed_q_fp)
        kept_q_uf, kept_q_fp, late_concentration_kept = self._apply_late_concentration_keeper(
            compensated_q_uf,
            compensated_q_fp,
        )
        idle_seek_q_uf, idle_seek_q_fp, idle_seeking = self._apply_post_target_idle_seeker(
            kept_q_uf,
            kept_q_fp,
        )
        if self._direct_q_fp_physical_only_enabled():
            finished_q_fp = float(idle_seek_q_fp)
            buffer_zero_finished = False
        else:
            finished_q_fp, buffer_zero_finished = self._apply_buffer_zero_finisher(
                idle_seek_q_uf,
                idle_seek_q_fp,
                minutes_this_step,
            )
        capped_q_uf, capped_q_fp, governor_budget_capped = self._cap_governor_total_adjustment(
            commanded_q_uf,
            commanded_q_fp,
            idle_seek_q_uf,
            finished_q_fp,
        )
        self.Q_uf = float(capped_q_uf)
        self.Q_fp = float(capped_q_fp)
        scheduled_q_uf = float(self.Q_uf)
        scheduled_q_fp = float(self.Q_fp)
        governor_total_delta = abs(float(scheduled_q_uf) - float(commanded_q_uf)) + abs(float(scheduled_q_fp) - float(commanded_q_fp))
        if float(self.governor_total_correction_limit) < 0.0:
            remaining_governor_budget = -1.0
        else:
            remaining_governor_budget = max(float(self.governor_total_correction_limit) - float(governor_total_delta), 0.0)
        self.last_applied_q_uf_delta = float(self.Q_uf - prev_q_uf_cmd)
        self.last_midcourse_quality_governed = bool(midcourse_quality_governed)
        self.last_late_target_compensated = bool(late_target_compensated)
        self.last_late_concentration_kept = bool(late_concentration_kept)
        self.last_late_target_q_uf_bias = float(q_uf_comp_bias)
        self.last_late_target_q_fp_bias = float(q_fp_comp_bias)
        self.last_post_target_idle_seeking = bool(idle_seeking)
        self.last_buffer_zero_finished = bool(buffer_zero_finished)
        self.last_buffer_zero_finish_q_fp = float(self.Q_fp)
        self.last_soft_teacher_q_uf = float(teacher_q_uf)
        self.last_soft_teacher_q_fp = float(teacher_q_fp)
        self.last_soft_teacher_active = bool(soft_teacher_active)
        self.last_soft_teacher_flags = dict(soft_teacher_flags)

        prev_q_uf = self.prev_q_uf
        prev_q_fp = self.prev_q_fp
        prev_v_buf = self.v_buf
        prev_c_aver = self.c_aver
        prev_m_fp = self.m_fp
        episode_step = self.policy_stepcnt
        commanded_q_fp = float(commanded_q_fp)
        applied_q_fp_sum = 0.0
        actual_q_fp_sum = 0.0
        batch_triggered_this_step = False
        post_target_governed_minutes = 0
        post_target_idle_minutes = 0
        low_buffer_guarded_minutes = 0
        mixer_idle_minutes = 0
        fp_prestop_ramped_minutes = 0
        fp_restart_ramped_minutes = 0
        q_fp_prestop_tapered_minutes = 0
        q_fp_slew_limited_minutes = 0
        minute_trace_time = []
        minute_trace_q_uf_actual = []
        minute_trace_q_fp_actual = []
        minute_trace_q_fp_env = []
        minute_trace_q_fp_cmd = []
        minute_trace_fp_busy = []
        minute_trace_v_buf = []
        minute_trace_c_uf = []
        minute_trace_c_aver = []
        minute_trace_m_fp = []
        minute_trace_total_energy_cost = []
        minute_trace_price = []

        # 执行物理模拟 (不终止，记录安全违规)
        total_energy_cost = 0.0
        safety_violations = 0
        dry_run_minutes = 0
        low_conc_minutes = 0
        final_c_uf = self.last_c_uf
        last_violations = []

        for _ in range(minutes_this_step):
            (
                step_energy,
                is_safe,
                violations,
                c_uf,
                effective_q_uf,
                effective_q_fp,
                actual_q_fp,
                batch_triggered,
                dry_run_now,
                low_conc_now,
                post_target_governed_now,
                post_target_idle_now,
                low_buffer_guarded_now,
                mixer_idle_now,
                fp_prestop_ramped_now,
                fp_restart_ramped_now,
                q_fp_prestop_tapered_now,
                q_fp_slew_limited_now,
                executed_fp_busy_now,
                step_price,
            ) = self._step_minute_physics(
                scheduled_q_uf=scheduled_q_uf,
                scheduled_q_fp=scheduled_q_fp,
                governor_budget=remaining_governor_budget,
            )
            total_energy_cost += step_energy
            final_c_uf = c_uf
            applied_q_fp_sum += effective_q_fp
            actual_q_fp_sum += actual_q_fp
            minute_trace_time.append(int(self.timecnt))
            minute_trace_q_uf_actual.append(float(effective_q_uf))
            minute_trace_q_fp_actual.append(float(actual_q_fp))
            minute_trace_q_fp_env.append(float(effective_q_fp))
            minute_trace_q_fp_cmd.append(float(commanded_q_fp))
            minute_trace_fp_busy.append(bool(executed_fp_busy_now))
            minute_trace_v_buf.append(float(self.v_buf))
            minute_trace_c_uf.append(float(c_uf))
            minute_trace_c_aver.append(float(self.c_aver))
            minute_trace_m_fp.append(float(self.m_fp))
            minute_trace_total_energy_cost.append(float(self.energy_cost_sum))
            minute_trace_price.append(float(step_price))
            batch_triggered_this_step = batch_triggered_this_step or batch_triggered
            if post_target_governed_now:
                post_target_governed_minutes += 1
            if post_target_idle_now:
                post_target_idle_minutes += 1
            if low_buffer_guarded_now:
                low_buffer_guarded_minutes += 1
            if mixer_idle_now:
                mixer_idle_minutes += 1
            if fp_prestop_ramped_now:
                fp_prestop_ramped_minutes += 1
            if fp_restart_ramped_now:
                fp_restart_ramped_minutes += 1
            if q_fp_prestop_tapered_now:
                q_fp_prestop_tapered_minutes += 1
            if q_fp_slew_limited_now:
                q_fp_slew_limited_minutes += 1
            if dry_run_now:
                dry_run_minutes += 1
            if low_conc_now:
                low_conc_minutes += 1
            if not is_safe:
                safety_violations += 1
                last_violations = violations

        avg_applied_q_fp = applied_q_fp_sum / max(minutes_this_step, 1)
        avg_actual_q_fp = actual_q_fp_sum / max(minutes_this_step, 1)
        episode_mean_c_uf = self.episode_c_uf_sum / max(self.episode_c_uf_count, 1)
        self.last_fp_batch_triggered = batch_triggered_this_step
        self.last_fp_prestop_ramped = bool(fp_prestop_ramped_minutes > 0)
        self.last_fp_restart_ramped = bool(fp_restart_ramped_minutes > 0)
        if not self.fp_busy and self.fp_restart_ramp_remain > 0:
            self.fp_restart_ramp_remain = max(self.fp_restart_ramp_remain - 1, 0)
        self.last_post_target_fp_governed = bool(post_target_governed_minutes > 0)
        self.last_post_target_idle_seeking = bool(self.last_post_target_idle_seeking or post_target_idle_minutes > 0)
        self.last_low_buffer_fp_guarded = bool(low_buffer_guarded_minutes > 0)

        delta_m_fp = self.m_fp - prev_m_fp
        is_safe = (safety_violations == 0)
        current_price = self.pricing.get_price(self.timecnt)

        # 奖励计算
        total_reward, breakdown, done = self.reward_scheme.compute(
            delta_m_fp=delta_m_fp,
            energy_cost=total_energy_cost,
            current_price=current_price,
            min_price=self.pricing.get_min_price(),
            max_price=self.pricing.get_max_price(),
            m_fp=self.m_fp,
            c_uf=final_c_uf,
            v_buf=self.v_buf,
            c_aver=self.c_aver,
            q_uf=self.Q_uf,
            q_fp=avg_applied_q_fp,
            prev_c_uf=self.last_c_uf,
            prev_q_uf=prev_q_uf,
            prev_q_fp=prev_q_fp,
            prev_v_buf=prev_v_buf,
            prev_c_aver=prev_c_aver,
            prev_m_fp=prev_m_fp,
            timecnt=self.timecnt,
            max_time_steps=self.max_steps * self.decision_interval,
            is_safe=is_safe,
            target_reached=self.target_reached,
            episode_step=episode_step,
            max_steps=self.max_steps,
            episode_mean_c_uf=episode_mean_c_uf,
            safety_violations=safety_violations,
            dry_run_minutes=dry_run_minutes,
            low_conc_minutes=low_conc_minutes,
        )

        q_fp_schedule_gap = max(float(scheduled_q_fp) - float(avg_applied_q_fp), 0.0)
        q_fp_actual_gap = max(float(avg_applied_q_fp) - float(avg_actual_q_fp), 0.0)
        q_fp_gap_tolerance = max(float(getattr(self.reward_config, "q_fp_gap_tolerance", 0.0)), 0.0)
        effective_q_fp_schedule_gap = max(q_fp_schedule_gap - q_fp_gap_tolerance, 0.0)
        effective_q_fp_actual_gap = max(q_fp_actual_gap - q_fp_gap_tolerance, 0.0)
        q_fp_env_correction_abs = abs(float(avg_applied_q_fp) - float(commanded_q_fp))
        q_fp_correction_tolerance = max(
            float(getattr(self.reward_config, "q_fp_correction_tolerance", 0.0)),
            0.0,
        )
        q_fp_correction_excess = max(q_fp_env_correction_abs - q_fp_correction_tolerance, 0.0)
        q_fp_schedule_gap_penalty = 0.0
        q_fp_actual_gap_penalty = 0.0
        guard_intervention_penalty = 0.0
        q_fp_correction_excess_penalty = 0.0
        if self.reward_config.q_fp_schedule_gap_penalty_weight > 0.0:
            q_fp_schedule_gap_penalty -= (
                float(self.reward_config.q_fp_schedule_gap_penalty_weight)
                * (effective_q_fp_schedule_gap / 70.0) ** 2
            )
        if self.reward_config.q_fp_actual_gap_penalty_weight > 0.0:
            q_fp_actual_gap_penalty -= (
                float(self.reward_config.q_fp_actual_gap_penalty_weight)
                * (effective_q_fp_actual_gap / 70.0) ** 2
            )
        if (
            float(getattr(self.reward_config, "guard_intervention_penalty_weight", 0.0)) > 0.0
            and low_buffer_guarded_minutes > 0
        ):
            guard_intervention_penalty -= float(self.reward_config.guard_intervention_penalty_weight)
        if (
            float(getattr(self.reward_config, "q_fp_correction_excess_penalty_weight", 0.0)) > 0.0
            and q_fp_correction_excess > 0.0
        ):
            corr_scale = max(q_fp_correction_tolerance, 2.0)
            normalized_corr_excess = min(q_fp_correction_excess / corr_scale, 5.0)
            q_fp_correction_excess_penalty -= (
                float(self.reward_config.q_fp_correction_excess_penalty_weight)
                * (normalized_corr_excess ** 2)
            )
        total_reward = float(
            np.clip(
                total_reward
                + q_fp_schedule_gap_penalty
                + q_fp_actual_gap_penalty
                + guard_intervention_penalty
                + q_fp_correction_excess_penalty,
                self.reward_config.reward_clip_min,
                self.reward_config.reward_clip_max,
            )
        )
        breakdown["q_fp_schedule_gap_penalty"] = float(q_fp_schedule_gap_penalty)
        breakdown["q_fp_actual_gap_penalty"] = float(q_fp_actual_gap_penalty)
        breakdown["q_fp_gap_tolerance"] = float(q_fp_gap_tolerance)
        breakdown["guard_intervention"] = float(guard_intervention_penalty)
        breakdown["q_fp_correction_excess_penalty"] = float(q_fp_correction_excess_penalty)
        breakdown["q_fp_correction_tolerance"] = float(q_fp_correction_tolerance)

        # 达标标记
        if (not self.target_reached
                and self.m_fp >= self.reward_config.target_mass
                and self.reward_config.target_completion_short_circuit):
            self.target_reached = True

        self.prev_q_uf = self.Q_uf
        self.prev_q_fp = avg_actual_q_fp if self._direct_q_fp_physical_only_enabled() else avg_applied_q_fp
        self.prev_v_buf = self.v_buf
        self.prev_c_aver = self.c_aver
        self.prev_m_fp = self.m_fp
        self.last_c_uf = final_c_uf

        if self.target_reached and self.reward_config.target_completion_short_circuit:
            self.Qf = 0.0
            self.Cf = 0.0
        else:
            self._sample_feed_conditions()

        self.policy_stepcnt += 1
        self.action_history.pop(0)
        history_q_fp = float(avg_actual_q_fp) if self._direct_q_fp_physical_only_enabled() else float(avg_applied_q_fp)
        self.action_history.append((self.Q_uf, history_q_fp))

        mass_buf = self._compute_mass_buf()
        obs = self._make_obs(final_c_uf, mass_buf)

        truncated = self.policy_stepcnt >= self.max_steps or self.timecnt >= self.total_minutes
        if truncated:
            done = True

        info = {
            "action_mode": self.mode,
            "uf_control_mode": self.uf_control_mode,
            "raw_action_q_uf": float(self.last_raw_action[0]),
            "raw_action_q_fp": float(self.last_raw_action[1]),
            "commanded_q_uf": float(commanded_q_uf),
            "applied_q_uf_delta": float(self.last_applied_q_uf_delta),
            "applied_q_uf": float(self.Q_uf),
            "commanded_q_fp": commanded_q_fp,
            "scheduled_q_uf": scheduled_q_uf,
            "scheduled_q_fp": scheduled_q_fp,
            "governor_total_correction_limit": float(self.governor_total_correction_limit),
            "governor_total_delta": float(governor_total_delta),
            "governor_budget_remaining": float(remaining_governor_budget),
            "governor_budget_capped": bool(governor_budget_capped),
            "applied_q_fp": float(avg_applied_q_fp),
            "actual_q_fp": float(avg_actual_q_fp),
            "q_fp_governor_gap": float(scheduled_q_fp - commanded_q_fp),
            "q_fp_execution_gap": float(avg_applied_q_fp - scheduled_q_fp),
            "q_fp_physical_gap": float(avg_actual_q_fp - avg_applied_q_fp),
            "q_fp_postprocess_gap": float(avg_applied_q_fp - commanded_q_fp),
            "q_fp_physical_clip_gap": float(avg_actual_q_fp - avg_applied_q_fp),
            "q_fp_env_correction_abs": float(q_fp_env_correction_abs),
            "q_fp_correction_excess": float(q_fp_correction_excess),
            "fp_prestop_ramped": bool(self.last_fp_prestop_ramped),
            "fp_prestop_ramped_minutes": int(fp_prestop_ramped_minutes),
            "fp_restart_ramped": bool(self.last_fp_restart_ramped),
            "fp_restart_ramped_minutes": int(fp_restart_ramped_minutes),
            "fp_restart_ramp_remain": int(self.fp_restart_ramp_remain),
            "q_fp_prestop_tapered": bool(q_fp_prestop_tapered_minutes > 0),
            "q_fp_prestop_tapered_minutes": int(q_fp_prestop_tapered_minutes),
            "q_fp_slew_limited": bool(q_fp_slew_limited_minutes > 0),
            "q_fp_slew_limited_minutes": int(q_fp_slew_limited_minutes),
            "direct_q_fp_physical_only": bool(self._direct_q_fp_physical_only_enabled()),
            "midcourse_quality_governed": bool(self.last_midcourse_quality_governed),
            "late_target_compensated": bool(self.last_late_target_compensated),
            "late_concentration_kept": bool(self.last_late_concentration_kept),
            "late_target_q_uf_bias": float(self.last_late_target_q_uf_bias),
            "late_target_q_fp_bias": float(self.last_late_target_q_fp_bias),
            "post_target_idle_seeking": bool(self.last_post_target_idle_seeking),
            "buffer_zero_finished": bool(self.last_buffer_zero_finished),
            "buffer_zero_finish_q_fp": float(self.last_buffer_zero_finish_q_fp),
            "soft_teacher_q_uf": float(self.last_soft_teacher_q_uf),
            "soft_teacher_q_fp": float(self.last_soft_teacher_q_fp),
            "soft_teacher_active": bool(self.last_soft_teacher_active),
            "teacher_midcourse_quality": bool(self.last_soft_teacher_flags.get("teacher_midcourse_quality", False)),
            "teacher_late_target": bool(self.last_soft_teacher_flags.get("teacher_late_target", False)),
            "teacher_late_concentration": bool(self.last_soft_teacher_flags.get("teacher_late_concentration", False)),
            "teacher_post_target_idle": bool(self.last_soft_teacher_flags.get("teacher_post_target_idle", False)),
            "teacher_post_target_fp": bool(self.last_soft_teacher_flags.get("teacher_post_target_fp", False)),
            "teacher_buffer_zero_finish": bool(self.last_soft_teacher_flags.get("teacher_buffer_zero_finish", False)),
            "total_energy_cost": self.energy_cost_sum,
            "energy_cost_step": total_energy_cost,
            "current_mass": self.m_fp,
            "current_price": current_price,
            "c_uf": float(final_c_uf),
            "buffer_volume": float(self.v_buf),
            "episode_mean_c_uf": float(episode_mean_c_uf),
            "buffer_below_idle_threshold": bool(self.v_buf <= self.mixer_power_off_volume_threshold + 1e-6),
            "safety_violation": not is_safe,
            "safety_violations": safety_violations,
            "dry_run_violation": bool(dry_run_minutes > 0),
            "dry_run_minutes": int(dry_run_minutes),
            "low_conc_violation": bool(low_conc_minutes > 0),
            "low_conc_minutes": int(low_conc_minutes),
            "violations": last_violations,
            "delta_m_fp": delta_m_fp,
            "reward_breakdown": breakdown,
            "target_reached": self.target_reached,
            "policy_step": self.policy_stepcnt,
            "startup_minutes": self.startup_minutes,
            "total_minutes": self.total_minutes,
            "available_control_minutes": self.available_control_minutes,
            "fp_batch_enabled": self.enable_fp_batching,
            "fp_busy": bool(self.fp_busy),
            "fp_downtime_remain": int(self.fp_downtime_remain),
            "fp_cycle_mass": float(self.fp_cycle_mass),
            "fp_total_cycles": int(self.fp_total_cycles),
            "fp_batch_triggered": bool(batch_triggered_this_step),
            "post_target_fp_governed": bool(post_target_governed_minutes > 0),
            "post_target_fp_governed_minutes": int(post_target_governed_minutes),
            "low_buffer_fp_guarded": bool(low_buffer_guarded_minutes > 0),
            "low_buffer_fp_guarded_minutes": int(low_buffer_guarded_minutes),
            "mixer_idle": bool(mixer_idle_minutes > 0),
            "mixer_idle_minutes": int(mixer_idle_minutes),
            "minute_trace_time": minute_trace_time,
            "minute_trace_q_uf_actual": minute_trace_q_uf_actual,
            "minute_trace_q_fp_actual": minute_trace_q_fp_actual,
            "minute_trace_q_fp_env": minute_trace_q_fp_env,
            "minute_trace_q_fp_cmd": minute_trace_q_fp_cmd,
            "minute_trace_fp_busy": minute_trace_fp_busy,
            "minute_trace_v_buf": minute_trace_v_buf,
            "minute_trace_c_uf": minute_trace_c_uf,
            "minute_trace_c_aver": minute_trace_c_aver,
            "minute_trace_m_fp": minute_trace_m_fp,
            "minute_trace_total_energy_cost": minute_trace_total_energy_cost,
            "minute_trace_price": minute_trace_price,
        }

        if self.verbose:
            bd_str = ", ".join(f"{k}={v:+.1f}" for k, v in breakdown.items() if k != 'total')
            print(
                f"Step {self.policy_stepcnt:4d} | mode={self.mode} | action=[{self.Q_uf:.1f},{self.Q_fp:.1f}] "
                f"| r={total_reward:+7.1f} | m={self.m_fp:6.1f} "
                f"| c_uf={final_c_uf:.4f} v_buf={self.v_buf:.2f} "
                f"| fp_busy={self.fp_busy}({self.fp_downtime_remain}) "
                f"| energy={total_energy_cost:.2f} | violations={safety_violations} | {bd_str}"
            )

        return obs, total_reward, done, truncated, info

    def _step_minute_physics(self, scheduled_q_uf: float, scheduled_q_fp: float, governor_budget: float):
        executed_fp_busy_now = bool(self.fp_busy)
        effective_q_uf = float(self.Q_uf)
        effective_q_fp = 0.0 if (self.enable_fp_batching and self.fp_busy) else float(self.Q_fp)
        fp_prestop_ramped_now = False
        fp_restart_ramped_now = False
        q_fp_prestop_tapered_now = False
        q_fp_slew_limited_now = False
        if not self.fp_busy:
            effective_q_fp, fp_prestop_ramped_now, fp_restart_ramped_now = self._apply_fp_batch_edge_ramps(
                effective_q_fp
            )
        post_target_idle_now = False
        if self._direct_q_fp_physical_only_enabled():
            post_target_governed_now = False
            # In direct-Q_fp mode we still allow a very narrow low-buffer guard:
            # it is the last equipment-level protection against repeated dry-run,
            # and its single-step correction is explicitly capped.
            effective_q_fp, low_buffer_guarded_now = self._apply_low_buffer_fp_guard(effective_q_fp)
        else:
            if self._is_post_target_idle_mode():
                effective_q_uf, effective_q_fp, post_target_idle_now = self._apply_post_target_idle_seeker(
                    effective_q_uf,
                    effective_q_fp,
                )
            effective_q_fp, post_target_governed_now = self._apply_post_target_fp_governor(effective_q_fp)
            effective_q_fp, low_buffer_guarded_now = self._apply_low_buffer_fp_guard(effective_q_fp)
        effective_q_uf, effective_q_fp, minute_budget_capped = self._cap_governor_total_adjustment(
            scheduled_q_uf,
            scheduled_q_fp,
            effective_q_uf,
            effective_q_fp,
            budget=governor_budget,
        )
        post_target_governed_now = bool(post_target_governed_now and abs(effective_q_fp - float(scheduled_q_fp)) > 1e-6)
        post_target_idle_now = bool(
            post_target_idle_now
            and (
                abs(effective_q_uf - float(scheduled_q_uf)) > 1e-6
                or abs(effective_q_fp - float(scheduled_q_fp)) > 1e-6
            )
        )
        low_buffer_guarded_now = bool(low_buffer_guarded_now and abs(effective_q_fp - float(scheduled_q_fp)) > 1e-6)
        physically_available_q_fp = float(max(self.v_buf + effective_q_uf / 60.0, 0.0) * 60.0)
        target_actual_q_fp = float(
            min(physically_available_q_fp, max(effective_q_fp, 0.0))
        )
        target_actual_q_fp, q_fp_prestop_tapered_now = self._apply_q_fp_actual_prestop_taper(target_actual_q_fp)
        actual_q_fp, q_fp_slew_limited_now = self._apply_q_fp_actual_slew_limit(target_actual_q_fp)
        # Final hard clip: even after restart-edge smoothing, the physically
        # realized press flow cannot exceed the inventory available in this minute.
        actual_q_fp = float(min(actual_q_fp, physically_available_q_fp))
        prev_m_fp = self.m_fp
        c_uf_floor, self.thickener_state = self.thickener.step(
            effective_q_uf, self.Qf, self.Cf, self.thickener_state
        )
        c_uf = c_uf_floor[-1]
        self.episode_c_uf_sum += float(c_uf)
        self.episode_c_uf_count += 1

        self.v_buf, self.c_aver, self.m_fp, mass_buf = (
            self.buffer_press.step(
                self.v_buf, self.m_fp, self.c_aver,
                effective_q_uf, actual_q_fp, c_uf,
            )
        )

        if self.v_buf <= self.mixer_idle_volume_threshold:
            self.v_buf = 0.0
            self.c_aver = c_uf

        # Separate "true empty" from "power-off zone":
        # - true empty still snaps to zero only at a tiny physical threshold;
        # - mixer standby power is disabled earlier once buffer inventory is
        #   already very low, even if the filter press is still allowed to run.
        low_buffer_idle_now = bool(self.v_buf <= self.mixer_power_off_volume_threshold + 1e-6)

        p_uf = 30 * (effective_q_uf / 50) ** 3 if effective_q_uf > 1e-6 else 0
        p_buff = 30 if not low_buffer_idle_now else 0
        p_fp = 90 * (actual_q_fp / 70) ** 3 if actual_q_fp > 1e-6 else 0

        current_price = self.pricing.get_price(self.timecnt)
        energy_cost = (p_uf + p_buff + p_fp) * current_price / 60
        self.energy_cost_sum += energy_cost

        is_safe, violations = self.buffer_press.check_safety(c_uf, self.v_buf)
        dry_run_now = bool(
            effective_q_fp > 1e-6
            and actual_q_fp + 1e-6 < effective_q_fp
            and self.v_buf < self._active_dry_run_threshold()
        )
        low_conc_now = bool(c_uf < self.reward_config.uf_conc_soft_low_limit)
        delta_m_fp = self.m_fp - prev_m_fp
        self.last_delta_m_fp_step = float(delta_m_fp)
        batch_triggered = self._update_fp_batch_state(delta_m_fp)
        self.timecnt += 1

        return (
            energy_cost,
            is_safe,
            violations,
            c_uf,
            effective_q_uf,
            effective_q_fp,
            actual_q_fp,
            batch_triggered,
            dry_run_now,
            low_conc_now,
            post_target_governed_now,
            post_target_idle_now,
            low_buffer_guarded_now,
            low_buffer_idle_now,
            fp_prestop_ramped_now,
            fp_restart_ramped_now,
            q_fp_prestop_tapered_now,
            q_fp_slew_limited_now,
            executed_fp_busy_now,
            current_price,
        )
