"""
搅拌罐 + 压滤模型
质量守恒与体积平衡
"""

import numpy as np
from .thickener import ThickenerModel


class BufferPressModel:
    """
    搅拌罐与压滤段模型

    功能:
        - 接收浓密机底流 (Q_uf, C_uf)
        - 搅拌罐缓冲混合
        - 压滤泵输出 (Q_fp)
        - 追踪累计干矿量 M_FP
    """

    # 搅拌罐功率 (kW) - 当罐内有料时消耗
    AGITATOR_POWER = 30.0

    # 安全限制
    MAX_BUFFER_VOLUME = 30.0      # m³
    MAX_UNDERFLOW_CONC = 0.75     # 体积分数上限

    def __init__(self):
        self.thickener = ThickenerModel()

    def step(self, v_buf, m_fp, c_aver, q_uf, q_fp, c_uf):
        """
        执行 1 分钟的搅拌罐-压滤状态推进

        参数:
            v_buf: 搅拌罐当前体积 (m³)
            m_fp: 累计干矿量 (t)
            c_aver: 搅拌罐平均体积分数
            q_uf: 底流泵流量 (m³/h)
            q_fp: 压滤泵流量 (m³/h)
            c_uf: 底流体积分数 (来自浓密机)

        返回:
            v_buf_new: 更新后体积
            c_aver_new: 更新后平均浓度
            m_fp_new: 更新后累计干矿量
            mass_buf: 搅拌罐内干矿质量
        """
        # 进出体积 (m³/min)
        v_in = q_uf / 60.0
        v_out = q_fp / 60.0

        # 当前干矿质量
        mass_buf = c_aver * v_buf * self.thickener.c2d(c_aver)

        # 进入的干矿质量
        mass_in = c_uf * q_uf * self.thickener.c2d(c_uf) / 60.0

        # 限制出料不超过当前体积
        v_out = min(v_out, v_buf)

        # 输出的干矿质量
        mass_out = c_aver * v_out * self.thickener.c2d(c_aver)

        # 质量平衡
        mass_buf = mass_buf + mass_in - mass_out

        # 体积平衡
        v_buf_new = v_buf + v_in - v_out

        # 防止负体积
        if v_buf_new <= 1e-6:
            v_buf_new = 0.0
            c_aver_new = c_uf
            mass_buf = 0.0
        else:
            # 更新平均浓度 (质量 -> 体积分数)
            # c = rho_solid * mass / (rho_solid * V + (rho_solid - rho_water) * mass)
            c_aver_new = (mass_buf * 4.27
                          / (4.27 * v_buf_new + 3.27 * mass_buf))

        # 累计干矿量
        m_fp_new = m_fp + mass_out

        return v_buf_new, c_aver_new, m_fp_new, mass_buf

    def check_safety(self, c_uf, v_buf):
        """
        安全检查

        返回:
            violated: 是否违反安全约束
            violations: 违反列表
        """
        violations = []
        if c_uf > self.MAX_UNDERFLOW_CONC:
            violations.append(f"底流浓度超限: {c_uf:.4f} > {self.MAX_UNDERFLOW_CONC}")
        if v_buf > self.MAX_BUFFER_VOLUME:
            violations.append(f"缓冲罐超容: {v_buf:.4f} > {self.MAX_BUFFER_VOLUME}")
        return len(violations) == 0, violations
