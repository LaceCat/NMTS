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
        enable_post_target_fp_governor: bool = True,
        post_target_fp_guard_level: float = 24.0,
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
        self.enable_post_target_fp_governor = bool(enable_post_target_fp_governor)
        self.post_target_fp_guard_level = float(max(post_target_fp_guard_level, 1e-6))
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
        self.fp_busy = False
        self.fp_downtime_remain = 0
        self.fp_cycle_mass = 0.0
        self.fp_total_cycles = 0
        self.last_fp_batch_triggered = False
        self.last_post_target_fp_governed = False

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
                q_uf = float(np.clip(self.Q_uf + requested_delta, 0.0, 50.0))
            else:
                q_uf = float(np.clip(arr[0], 0.0, 50.0))
            q_fp = self._discrete_on_off(float(arr[1]), 70.0)
        else:
            if self.uf_control_mode == "delta":
                requested_delta = float(np.clip(arr[0], -self.uf_delta_max, self.uf_delta_max))
                q_uf = float(np.clip(self.Q_uf + requested_delta, 0.0, 50.0))
            else:
                q_uf = float(np.clip(arr[0], 0.0, 50.0))
            q_fp = float(np.clip(arr[1], 0.0, 70.0))
            if self.q_fp_delta_max is not None:
                q_fp = float(
                    np.clip(
                        q_fp,
                        max(0.0, self.Q_fp - self.q_fp_delta_max),
                        min(70.0, self.Q_fp + self.q_fp_delta_max),
                    )
                )

        return np.array([q_uf, q_fp], dtype=np.float32)

    def _sample_feed_conditions(self):
        if self.feed_volatility == "high":
            self.Qf = float(np.random.uniform(35, 50))
            self.Cf = float(np.random.uniform(0.30, 0.45))
        else:
            self.Qf = float(np.random.uniform(40, 45))
            self.Cf = float(np.random.uniform(0.35, 0.40))

    def _compute_mass_buf(self) -> float:
        if self.v_buf <= 0.0:
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
            return False

        if delta_m_fp > 0.0:
            self.fp_cycle_mass += float(delta_m_fp)

        if self.fp_cycle_mass + 1e-9 >= self.fp_batch_mass:
            self.fp_cycle_mass = max(self.fp_cycle_mass - self.fp_batch_mass, 0.0)
            self.fp_busy = True
            self.fp_downtime_remain = self.fp_downtime_minutes
            self.fp_total_cycles += 1
            return True

        return False

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
        if self.m_fp < self.reward_config.target_mass:
            return float(q_fp), False
        if self.v_buf >= self.post_target_fp_guard_level:
            return float(q_fp), False
        return 0.0, bool(q_fp > 1e-6)

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
        self.timecnt = 0
        self.policy_stepcnt = 0
        self.m_fp = 0.0
        self.v_buf = 0.0
        self.c_aver = self.last_c_uf
        self.prev_v_buf = self.v_buf
        self.prev_c_aver = self.c_aver
        self.prev_m_fp = 0.0
        self.energy_cost_sum = 0.0
        self.target_reached = False
        self.action_history = [(0.0, 0.0) for _ in range(self.ACTION_HISTORY_STEPS)]
        self.warmup_minutes_used = 0
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_applied_q_uf_delta = 0.0
        self.fp_busy = False
        self.fp_downtime_remain = 0
        self.fp_cycle_mass = 0.0
        self.fp_total_cycles = 0
        self.last_fp_batch_triggered = False

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
        self.Q_uf = float(physical_action[0])
        self.Q_fp = float(physical_action[1])
        self.last_applied_q_uf_delta = float(self.Q_uf - prev_q_uf_cmd)

        prev_q_uf = self.prev_q_uf
        prev_q_fp = self.prev_q_fp
        prev_v_buf = self.v_buf
        prev_c_aver = self.c_aver
        prev_m_fp = self.m_fp
        episode_step = self.policy_stepcnt
        commanded_q_fp = float(self.Q_fp)
        applied_q_fp_sum = 0.0
        batch_triggered_this_step = False
        post_target_governed_minutes = 0

        # 执行物理模拟 (不终止，记录安全违规)
        total_energy_cost = 0.0
        safety_violations = 0
        dry_run_minutes = 0
        low_conc_minutes = 0
        final_c_uf = self.last_c_uf
        last_violations = []

        minutes_left = max(self.total_minutes - self.timecnt, 0)
        minutes_this_step = min(self.decision_interval, minutes_left)

        for _ in range(minutes_this_step):
            step_energy, is_safe, violations, c_uf, effective_q_fp, batch_triggered, dry_run_now, low_conc_now, post_target_governed_now = self._step_minute_physics()
            total_energy_cost += step_energy
            final_c_uf = c_uf
            applied_q_fp_sum += effective_q_fp
            batch_triggered_this_step = batch_triggered_this_step or batch_triggered
            if post_target_governed_now:
                post_target_governed_minutes += 1
            if dry_run_now:
                dry_run_minutes += 1
            if low_conc_now:
                low_conc_minutes += 1
            if not is_safe:
                safety_violations += 1
                last_violations = violations

        avg_applied_q_fp = applied_q_fp_sum / max(minutes_this_step, 1)
        self.last_fp_batch_triggered = batch_triggered_this_step
        self.last_post_target_fp_governed = bool(post_target_governed_minutes > 0)

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
            safety_violations=safety_violations,
            dry_run_minutes=dry_run_minutes,
            low_conc_minutes=low_conc_minutes,
        )

        # 达标标记
        if (not self.target_reached
                and self.m_fp >= self.reward_config.target_mass
                and self.reward_config.target_completion_short_circuit):
            self.target_reached = True

        self.prev_q_uf = self.Q_uf
        self.prev_q_fp = avg_applied_q_fp
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
        self.action_history.append((self.Q_uf, avg_applied_q_fp))

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
            "applied_q_uf_delta": float(self.last_applied_q_uf_delta),
            "applied_q_uf": float(self.Q_uf),
            "commanded_q_fp": commanded_q_fp,
            "applied_q_fp": float(avg_applied_q_fp),
            "total_energy_cost": self.energy_cost_sum,
            "energy_cost_step": total_energy_cost,
            "current_mass": self.m_fp,
            "current_price": current_price,
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

    def _step_minute_physics(self):
        effective_q_fp = 0.0 if (self.enable_fp_batching and self.fp_busy) else self.Q_fp
        effective_q_fp, post_target_governed_now = self._apply_post_target_fp_governor(effective_q_fp)
        prev_m_fp = self.m_fp
        c_uf_floor, self.thickener_state = self.thickener.step(
            self.Q_uf, self.Qf, self.Cf, self.thickener_state
        )
        c_uf = c_uf_floor[-1]

        self.v_buf, self.c_aver, self.m_fp, mass_buf = (
            self.buffer_press.step(
                self.v_buf, self.m_fp, self.c_aver,
                self.Q_uf, effective_q_fp, c_uf,
            )
        )

        if self.v_buf <= 1e-6:
            self.v_buf = 0.0
            self.c_aver = c_uf

        p_uf = 30 * (self.Q_uf / 50) ** 3 if self.Q_uf > 1e-6 else 0
        p_buff = 30 if self.v_buf > 1e-6 else 0
        p_fp = 90 * (effective_q_fp / 70) ** 3 if effective_q_fp > 1e-6 else 0

        current_price = self.pricing.get_price(self.timecnt)
        energy_cost = (p_uf + p_buff + p_fp) * current_price / 60
        self.energy_cost_sum += energy_cost

        is_safe, violations = self.buffer_press.check_safety(c_uf, self.v_buf)
        dry_run_now = bool(
            effective_q_fp > 1e-6
            and self.v_buf < self.reward_config.dry_run_buffer_threshold
        )
        low_conc_now = bool(c_uf < self.reward_config.uf_conc_soft_low_limit)
        delta_m_fp = self.m_fp - prev_m_fp
        batch_triggered = self._update_fp_batch_state(delta_m_fp)
        self.timecnt += 1

        return energy_cost, is_safe, violations, c_uf, effective_q_fp, batch_triggered, dry_run_now, low_conc_now, post_target_governed_now
