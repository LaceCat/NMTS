"""
搅拌槽 + 压滤模型。
"""

from __future__ import annotations

from .thickener import ThickenerModel


class BufferPressModel:
    AGITATOR_POWER = 30.0

    MAX_BUFFER_VOLUME = 30.0
    MAX_UNDERFLOW_CONC = 0.75

    def __init__(self):
        self.thickener = ThickenerModel()

    def step(self, v_buf, m_fp, c_aver, q_uf, q_fp, c_uf):
        """
        执行 1 分钟的搅拌槽-压滤状态推进。

        参数中的 c_aver、c_uf 都保持为外部接口的体积分数语义。
        """
        v_in = q_uf / 60.0
        v_out = q_fp / 60.0

        mass_buf = c_aver * v_buf * self.thickener.c2d(c_aver)
        mass_in = c_uf * q_uf * self.thickener.c2d(c_uf) / 60.0

        # Minute-level perfectly mixed balance:
        # the press can consume both the existing buffer inventory and the
        # fresh underflow that arrives within the same minute.
        available_volume = max(v_buf + v_in, 0.0)
        available_mass = max(mass_buf + mass_in, 0.0)
        v_out = min(max(v_out, 0.0), available_volume)

        if available_volume <= 1e-9 or available_mass <= 0.0:
            mass_out = 0.0
            mass_buf = 0.0
            v_buf_new = max(available_volume - v_out, 0.0)
        else:
            mass_out = available_mass * (v_out / available_volume)
            mass_buf = max(available_mass - mass_out, 0.0)
            v_buf_new = max(available_volume - v_out, 0.0)

        if v_buf_new <= 1e-6:
            v_buf_new = 0.0
            c_aver_new = c_uf
            mass_buf = 0.0
        else:
            c_aver_new = (mass_buf * 4.27) / (4.27 * v_buf_new + 3.27 * mass_buf)

        m_fp_new = m_fp + mass_out
        return v_buf_new, c_aver_new, m_fp_new, mass_buf

    def check_safety(self, c_uf, v_buf):
        violations = []
        if c_uf > self.MAX_UNDERFLOW_CONC:
            violations.append(f"底流浓度超限: {c_uf:.4f} > {self.MAX_UNDERFLOW_CONC}")
        if v_buf > self.MAX_BUFFER_VOLUME:
            violations.append(f"缓冲罐超容: {v_buf:.4f} > {self.MAX_BUFFER_VOLUME}")
        return len(violations) == 0, violations
