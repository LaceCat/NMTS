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

from .physics.thickener import (
    ThickenerModel,
    DEFAULT_INITIAL_CONCENTRATION_PROFILE,
    DEFAULT_THICKENER_STATE,
)
from .physics.buffer_press import BufferPressModel
from .reward.config import RewardConfig
from .reward.scheme import RewardScheme
from .reward.pricing import ElectricityPricing, PricingPresets


class ThickenerDewateringEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 288,
        decision_interval: int = 5,
        target_mass: float = 400.0,
        pricing: Optional[ElectricityPricing] = None,
        reward_config: Optional[RewardConfig] = None,
        feed_volatility: str = "normal",
        verbose: bool = False,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.decision_interval = max(1, decision_interval)

        self.thickener = ThickenerModel()
        self.buffer_press = BufferPressModel()

        self.pricing = pricing if pricing is not None else PricingPresets.daily_24h()

        if reward_config is not None:
            self.reward_config = reward_config
        else:
            self.reward_config = RewardConfig(target_mass=target_mass, max_steps=max_steps)
        self.reward_scheme = RewardScheme(self.reward_config)

        self.feed_volatility = feed_volatility

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 70.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(9,), dtype=np.float32
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
        self.c_aver = 0.66
        self.timecnt = 0
        self.policy_stepcnt = 0
        self.energy_cost_sum = 0.0
        self.target_reached = False

        self.prev_q_uf = 0.0
        self.prev_q_fp = 0.0
        self.prev_v_buf = 0.0
        self.prev_c_aver = 0.66
        self.prev_m_fp = 0.0
        self.last_c_uf = float(DEFAULT_INITIAL_CONCENTRATION_PROFILE[-1])

    def _sample_feed_conditions(self):
        if self.feed_volatility == "high":
            self.Qf = float(np.random.uniform(35, 50))
            self.Cf = float(np.random.uniform(0.30, 0.45))
        else:
            self.Qf = float(np.random.uniform(40, 45))
            self.Cf = float(np.random.uniform(0.35, 0.40))

    def _make_obs(self, c_uf: float, mass_buf: float) -> np.ndarray:
        price = self.pricing.get_price(self.timecnt)
        remaining = max(self.max_steps - self.policy_stepcnt, 0)
        return np.array([
            c_uf, self.v_buf, self.c_aver, self.m_fp,
            mass_buf, price, remaining,
            self.Qf, self.Cf,
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._init_state()
        self._sample_feed_conditions()

        c_uf = self.last_c_uf
        mass_buf = (self.c_aver * self.v_buf * self.thickener.c2d(self.c_aver)
                    if self.v_buf > 0 else 0.0)
        obs = self._make_obs(c_uf, mass_buf)
        return obs, {}

    def step(self, action: np.ndarray):
        self.Q_uf = float(np.clip(action[0], 0.0, 50.0))
        self.Q_fp = float(np.clip(action[1], 0.0, 70.0))

        prev_q_uf = self.prev_q_uf
        prev_q_fp = self.prev_q_fp
        prev_v_buf = self.v_buf
        prev_c_aver = self.c_aver
        prev_m_fp = self.m_fp
        episode_step = self.policy_stepcnt

        # 执行物理模拟 (不终止，记录安全违规)
        total_energy_cost = 0.0
        safety_violations = 0
        final_c_uf = self.last_c_uf
        last_violations = []

        for _ in range(self.decision_interval):
            step_energy, is_safe, violations, c_uf = self._step_minute_physics()
            total_energy_cost += step_energy
            final_c_uf = c_uf
            if not is_safe:
                safety_violations += 1
                last_violations = violations

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
            q_fp=self.Q_fp,
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
        )

        # 达标标记
        if (not self.target_reached
                and self.m_fp >= self.reward_config.target_mass
                and self.reward_config.target_completion_short_circuit):
            self.target_reached = True

        self.prev_q_uf = self.Q_uf
        self.prev_q_fp = self.Q_fp
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

        mass_buf = (self.c_aver * self.v_buf * self.thickener.c2d(self.c_aver)
                    if self.v_buf > 0 else 0.0)
        obs = self._make_obs(final_c_uf, mass_buf)

        truncated = self.policy_stepcnt >= self.max_steps
        if truncated:
            done = True

        info = {
            "total_energy_cost": self.energy_cost_sum,
            "energy_cost_step": total_energy_cost,
            "current_mass": self.m_fp,
            "current_price": current_price,
            "safety_violation": not is_safe,
            "safety_violations": safety_violations,
            "violations": last_violations,
            "delta_m_fp": delta_m_fp,
            "reward_breakdown": breakdown,
            "target_reached": self.target_reached,
            "policy_step": self.policy_stepcnt,
        }

        if self.verbose:
            bd_str = ", ".join(f"{k}={v:+.1f}" for k, v in breakdown.items() if k != 'total')
            print(
                f"Step {self.policy_stepcnt:4d} | action=[{self.Q_uf:.1f},{self.Q_fp:.1f}] "
                f"| r={total_reward:+7.1f} | m={self.m_fp:6.1f} "
                f"| c_uf={final_c_uf:.4f} v_buf={self.v_buf:.2f} "
                f"| energy={total_energy_cost:.2f} | violations={safety_violations} | {bd_str}"
            )

        return obs, total_reward, done, truncated, info

    def _step_minute_physics(self):
        c_uf_floor, self.thickener_state = self.thickener.step(
            self.Q_uf, self.Qf, self.Cf, self.thickener_state
        )
        c_uf = c_uf_floor[-1]

        self.v_buf, self.c_aver, self.m_fp, mass_buf = (
            self.buffer_press.step(
                self.v_buf, self.m_fp, self.c_aver,
                self.Q_uf, self.Q_fp, c_uf,
            )
        )

        if self.v_buf <= 1e-6:
            self.v_buf = 0.0
            self.c_aver = c_uf

        p_uf = 30 * (self.Q_uf / 50) ** 3 if self.Q_uf > 1e-6 else 0
        p_buff = 30 if self.v_buf > 1e-6 else 0
        p_fp = 90 * (self.Q_fp / 70) ** 3 if self.Q_fp > 1e-6 else 0

        current_price = self.pricing.get_price(self.timecnt)
        energy_cost = (p_uf + p_buff + p_fp) * current_price / 60
        self.energy_cost_sum += energy_cost

        is_safe, violations = self.buffer_press.check_safety(c_uf, self.v_buf)
        self.timecnt += 1

        return energy_cost, is_safe, violations, c_uf
