"""
Classical PID-style baseline controller for the thickener dewatering project.

The controller is intentionally simple and interpretable:
- Underflow pump loop mainly regulates underflow concentration.
- Filter-press loop mainly regulates buffer volume while considering progress
  and time-of-use electricity price.

It outputs physical actions in the same 2-D action space used by the
environment: [Q_uf, Q_fp].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDControllerConfig:
    target_mass: float = 400.0
    target_mass_high: float = 420.0
    total_steps: int = 288

    c_uf_setpoint: float = 0.715
    v_buf_setpoint: float = 1.5

    q_uf_base: float = 12.0
    q_fp_base: float = 26.0

    kp_uf: float = 85.0
    ki_uf: float = 2.5
    kd_uf: float = 12.0
    k_buf_to_uf: float = 0.12

    kp_fp: float = 5.0
    ki_fp: float = 0.25
    kd_fp: float = 1.0
    k_progress_to_fp: float = 0.20
    k_price_to_fp: float = 6.0
    k_overshoot_to_fp: float = 0.45

    q_uf_low: float = 0.0
    q_uf_high: float = 50.0
    q_fp_low: float = 0.0
    q_fp_high: float = 70.0

    integral_uf_limit: float = 0.30
    integral_fp_limit: float = 20.0


@dataclass
class PurePIDControllerConfig:
    """
    Pure PID baseline configuration.

    Compared with the engineering PID above, this controller intentionally
    removes task-progress, electricity-price, and manual tail heuristics.
    The default gains are tuned to satisfy the 400 t task requirement under
    the current CC environment while keeping the controller fully PID-based.
    """

    c_uf_setpoint: float = 0.715
    v_buf_setpoint: float = 4.0

    q_uf_base: float = 12.0
    q_fp_base: float = 22.0

    kp_uf: float = 85.0
    ki_uf: float = 2.5
    kd_uf: float = 12.0

    kp_fp: float = 4.5
    ki_fp: float = 0.20
    kd_fp: float = 0.8

    q_uf_low: float = 0.0
    q_uf_high: float = 50.0
    q_fp_low: float = 0.0
    q_fp_high: float = 70.0

    integral_uf_limit: float = 0.30
    integral_fp_limit: float = 25.0


class DualLoopPIDController:
    """
    PID baseline with light engineering feedforward terms.

    Observation convention follows env.gym_env.ThickenerDewateringEnv:
    base obs = [C_uf, V_buf, C_aver, M_FP, Mass_buf, price, remaining_steps, Qf, Cf]
    """

    def __init__(self, config: PIDControllerConfig | None = None, mode: str = "CC"):
        self.config = config or PIDControllerConfig()
        self.mode = str(mode).upper()
        self.name = "dual_loop_pid"
        self.reset()

    def reset(self) -> None:
        self.int_uf = 0.0
        self.int_fp = 0.0
        self.prev_err_uf = 0.0
        self.prev_err_fp = 0.0
        self.step_index = 0
        self.dd_uf_on = False
        self.discrete_fp_on = False

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return float(max(lower, min(upper, value)))

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.config
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        c_uf = float(obs[0])
        v_buf = float(obs[1])
        m_fp = float(obs[3])
        price = float(obs[5])
        remaining_steps = max(float(obs[6]), 1.0)

        price_mid = 0.81
        price_signal = price_mid - price

        desired_mass_now = cfg.target_mass * (self.step_index / max(cfg.total_steps, 1))
        progress_error = desired_mass_now - m_fp

        # Loop 1: regulate underflow concentration.
        # Higher c_uf than setpoint should trigger a larger Q_uf.
        err_uf = c_uf - cfg.c_uf_setpoint
        self.int_uf = self._clip(
            self.int_uf + err_uf,
            -cfg.integral_uf_limit,
            cfg.integral_uf_limit,
        )
        d_err_uf = err_uf - self.prev_err_uf
        self.prev_err_uf = err_uf

        q_uf = (
            cfg.q_uf_base
            + cfg.kp_uf * err_uf
            + cfg.ki_uf * self.int_uf
            + cfg.kd_uf * d_err_uf
            + cfg.k_buf_to_uf * max(v_buf - cfg.v_buf_setpoint, 0.0)
        )

        # Loop 2: regulate buffer volume, with progress and price feedforward.
        err_fp = v_buf - cfg.v_buf_setpoint
        self.int_fp = self._clip(
            self.int_fp + err_fp,
            -cfg.integral_fp_limit,
            cfg.integral_fp_limit,
        )
        d_err_fp = err_fp - self.prev_err_fp
        self.prev_err_fp = err_fp

        q_fp = (
            cfg.q_fp_base
            + cfg.kp_fp * err_fp
            + cfg.ki_fp * self.int_fp
            + cfg.kd_fp * d_err_fp
            + cfg.k_progress_to_fp * progress_error
            - cfg.k_price_to_fp * price_signal
        )

        # Once the task is already completed, draining should taper quickly.
        if m_fp >= cfg.target_mass:
            q_fp -= cfg.k_overshoot_to_fp * (m_fp - cfg.target_mass)
            q_uf -= 0.15 * (m_fp - cfg.target_mass)

        # Stronger taper when clearly above the descriptive target band.
        if m_fp >= cfg.target_mass_high:
            q_fp -= 0.80 * (m_fp - cfg.target_mass_high)
            q_uf -= 0.20 * (m_fp - cfg.target_mass_high)

        # Safety-oriented overrides.
        if c_uf >= 0.745:
            q_uf = max(q_uf, 14.0)
        if v_buf >= 26.0:
            q_fp = max(q_fp, 35.0)
        if v_buf >= 28.0:
            q_fp = max(q_fp, 50.0)
            q_uf = min(q_uf, 14.0)
        if v_buf <= 0.10 and m_fp >= cfg.target_mass:
            q_fp = 0.0
            q_uf = min(q_uf, 4.0)
        elif v_buf <= 0.05:
            q_fp = min(q_fp, 10.0)

        # Avoid very aggressive late production when there is plenty of time left.
        if remaining_steps > 40 and price >= 1.16 and m_fp >= cfg.target_mass * 0.85:
            q_fp *= 0.65

        q_uf = self._clip(q_uf, cfg.q_uf_low, cfg.q_uf_high)
        q_fp = self._clip(q_fp, cfg.q_fp_low, cfg.q_fp_high)

        q_uf, q_fp = self._apply_mode_mapping(q_uf, q_fp)

        self.step_index += 1
        return np.array([q_uf, q_fp], dtype=np.float32)

    def _apply_mode_mapping(self, q_uf: float, q_fp: float) -> tuple[float, float]:
        if self.mode == "CC":
            return q_uf, q_fp

        if self.mode == "CD":
            if self.discrete_fp_on:
                if q_fp <= 18.0:
                    self.discrete_fp_on = False
            else:
                if q_fp >= 24.0:
                    self.discrete_fp_on = True
            return q_uf, (70.0 if self.discrete_fp_on else 0.0)

        # DD mode: map both channels through hysteresis to avoid chattering.
        if self.dd_uf_on:
            if q_uf <= 16.0:
                self.dd_uf_on = False
        else:
            if q_uf >= 18.0:
                self.dd_uf_on = True

        if self.discrete_fp_on:
            if q_fp <= 18.0:
                self.discrete_fp_on = False
        else:
            if q_fp >= 24.0:
                self.discrete_fp_on = True

        return (50.0 if self.dd_uf_on else 0.0), (70.0 if self.discrete_fp_on else 0.0)


class PurePIDController:
    """
    Orthdox dual-loop PID baseline.

    Only uses direct feedback errors:
    - Q_uf <- PID(C_uf setpoint tracking)
    - Q_fp <- PID(V_buf setpoint tracking)

    No price feedforward, no production-progress feedforward,
    no target-mass taper logic, and no extra manual tail rules.
    """

    def __init__(self, config: PurePIDControllerConfig | None = None, mode: str = "CC"):
        self.config = config or PurePIDControllerConfig()
        self.mode = str(mode).upper()
        self.name = "pure_pid"
        self.reset()

    def reset(self) -> None:
        self.int_uf = 0.0
        self.int_fp = 0.0
        self.prev_err_uf = 0.0
        self.prev_err_fp = 0.0
        self.dd_uf_on = False
        self.discrete_fp_on = False

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return float(max(lower, min(upper, value)))

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.config
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        c_uf = float(obs[0])
        v_buf = float(obs[1])

        # Positive error should increase Q_uf when concentration is too high.
        err_uf = c_uf - cfg.c_uf_setpoint
        self.int_uf = self._clip(
            self.int_uf + err_uf,
            -cfg.integral_uf_limit,
            cfg.integral_uf_limit,
        )
        d_err_uf = err_uf - self.prev_err_uf
        self.prev_err_uf = err_uf

        q_uf = (
            cfg.q_uf_base
            + cfg.kp_uf * err_uf
            + cfg.ki_uf * self.int_uf
            + cfg.kd_uf * d_err_uf
        )

        # Positive error should increase Q_fp when the buffer is too full.
        err_fp = v_buf - cfg.v_buf_setpoint
        self.int_fp = self._clip(
            self.int_fp + err_fp,
            -cfg.integral_fp_limit,
            cfg.integral_fp_limit,
        )
        d_err_fp = err_fp - self.prev_err_fp
        self.prev_err_fp = err_fp

        q_fp = (
            cfg.q_fp_base
            + cfg.kp_fp * err_fp
            + cfg.ki_fp * self.int_fp
            + cfg.kd_fp * d_err_fp
        )

        q_uf = self._clip(q_uf, cfg.q_uf_low, cfg.q_uf_high)
        q_fp = self._clip(q_fp, cfg.q_fp_low, cfg.q_fp_high)
        q_uf, q_fp = self._apply_mode_mapping(q_uf, q_fp)
        return np.array([q_uf, q_fp], dtype=np.float32)

    def _apply_mode_mapping(self, q_uf: float, q_fp: float) -> tuple[float, float]:
        if self.mode == "CC":
            return q_uf, q_fp

        if self.mode == "CD":
            if self.discrete_fp_on:
                if q_fp <= 18.0:
                    self.discrete_fp_on = False
            else:
                if q_fp >= 24.0:
                    self.discrete_fp_on = True
            return q_uf, (70.0 if self.discrete_fp_on else 0.0)

        if self.dd_uf_on:
            if q_uf <= 16.0:
                self.dd_uf_on = False
        else:
            if q_uf >= 18.0:
                self.dd_uf_on = True

        if self.discrete_fp_on:
            if q_fp <= 18.0:
                self.discrete_fp_on = False
        else:
            if q_fp >= 24.0:
                self.discrete_fp_on = True

        return (50.0 if self.dd_uf_on else 0.0), (70.0 if self.discrete_fp_on else 0.0)
