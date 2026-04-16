"""
浓密机机理模型。

外部接口中的浓度变量 `c`、`c_uf`、`Cf` 保持为体积分数语义。
内部 10 层 PDE 状态 `thickener_state` 使用混合密度表示，单位 g/m^3。
"""

from __future__ import annotations

import numpy as np


DEFAULT_INITIAL_CONCENTRATION_PROFILE = np.array(
    [
        0.02039303,
        0.03981066,
        0.08502988,
        0.16719616,
        0.29188528,
        0.44878692,
        0.48620800,
        0.53520916,
        0.59008991,
        0.65434479,
    ],
    dtype=np.float64,
)


class ThickenerModel:
    RADIUS = 15.0
    TOTAL_HEIGHT = 5.79
    FEED_HEIGHT = 4.0
    N_LAYERS = 10
    RHO_SOLID = 4.27
    RHO_WATER = 1.0

    def __init__(self):
        self.area = np.pi * self.RADIUS**2
        self.detaz = self.TOTAL_HEIGHT / self.N_LAYERS

        lijing = 50
        v1dr = 1 / 40 * lijing + 1 / 4
        self._params = [
            0.547678836441774 * 1.5007390668665104,
            0.476947700914306 * 1.5007390668665104,
            (0.940451926648319 * 2) * 0.9682781622079192,
            0.952633313167728 * 0.9682781622079192,
            v1dr * 0.6724215861771662,
            16.5914198572464 * 1.0979700255743778,
            (1.14944107415579 * 2) * 2.4295166452320314,
            (1.73671299891156 * 1.5) * 2.4295166452320314,
            7.50829943915578e-06 * 0.5825392569061763,
            8.48008041543593e-06 * 0.9250599356670968,
            10.62486693809181e-07 * 1.3978736628247792,
            13.58704316829464e-07 * 1.3978736628247792,
        ]

    @staticmethod
    def c2d(c):
        """体积分数 -> 混合密度 (t/m^3)."""
        c = np.asarray(c, dtype=np.float64)
        rho_0 = 4.27
        rho_w = 1.0
        return rho_0 / (rho_0 - (rho_0 - rho_w) * c)

    @staticmethod
    def d2c(d):
        """密度 -> 体积分数转换 (安全闭合版)."""
        rho_0 = 4.27
        rho_water = 1.0
        d = float(d)
        if d <= rho_water:
            return 0.0
        return rho_0 * (d - rho_water) / ((rho_0 - 1) * d)

    def step(self, q_uf, qf, cf, thickener_state):
        """
        推进一步浓密机状态（1 分钟）。

        参数
        ----------
        q_uf:
            底流泵流量 (m^3/h)。
        qf:
            进料流量 (m^3/h)。
        cf:
            进料体积分数。
        thickener_state:
            10 层混合密度状态，单位 g/m^3。

        返回
        -------
        cf_floor:
            更新后 10 层体积分数。
        new_state:
            更新后 10 层混合密度状态，单位 g/m^3。
        """

        D1, D2, D3, D4 = self._params[0:4]
        v1, v2, v3, v4 = self._params[4:8]
        rv1, rv2, rv3, rv4 = self._params[8:12]

        deltat = 1 / 60
        t_final = 1 / 60
        L = int(t_final / deltat)
        A = self.area
        detaz = self.detaz

        qu = q_uf * np.ones(L)
        qf_arr = qf * np.ones(L)
        cf_arr = cf * np.ones(L)
        cf_density = self.c2d(cf_arr) * 1e6

        x1 = np.zeros(L + 1)
        x2 = np.zeros(L + 1)
        x3 = np.zeros(L + 1)
        x4 = np.zeros(L + 1)
        x5 = np.zeros(L + 1)
        x6 = np.zeros(L + 1)
        x7 = np.zeros(L + 1)
        x8 = np.zeros(L + 1)
        x9 = np.zeros(L + 1)
        x10 = np.zeros(L + 1)

        x1[0] = thickener_state[0]
        x2[0] = thickener_state[1]
        x3[0] = thickener_state[2]
        x4[0] = thickener_state[3]
        x5[0] = thickener_state[4]
        x6[0] = thickener_state[5]
        x7[0] = thickener_state[6]
        x8[0] = thickener_state[7]
        x9[0] = thickener_state[8]
        x10[0] = thickener_state[9]

        for k in range(L):
            qf_k = qf_arr[k] / A
            qu_k = qu[k] / A
            # qf_k / qu_k are already superficial velocities (m/h).
            # Upward overflow velocity should not be divided by area twice.
            qe = qf_k - qu_k
            xf = cf_density[k]

            x1[k + 1] = x1[k] + (
                (qe * x2[k] - qe * x1[k])
                - v1 * np.exp(-rv1 * x1[k]) * x1[k] / detaz
                + D1 * (x2[k] - x1[k]) / detaz**2
            ) * deltat

            x2[k + 1] = x2[k] + (
                (qe * x3[k] - qe * x2[k])
                + (v1 * np.exp(-rv1 * x1[k]) * x1[k] - v1 * np.exp(-rv1 * x2[k]) * x2[k]) / detaz
                + D1 * (x3[k] - 2 * x2[k] + x1[k]) / detaz**2
            ) * deltat

            x3[k + 1] = x3[k] + (
                (qe * x4[k] - qe * x3[k])
                + (v1 * np.exp(-rv1 * x2[k]) * x2[k] - v1 * np.exp(-rv1 * x3[k]) * x3[k]) / detaz
                + D1 * (x4[k] - 2 * x3[k] + x2[k]) / detaz**2
            ) * deltat

            x4[k + 1] = x4[k] + (
                (qe * x5[k] - qe * x4[k])
                + (v1 * np.exp(-rv1 * x3[k]) * x3[k] - v1 * np.exp(-rv1 * x4[k]) * x4[k]) / detaz
                + D1 * (x5[k] - 2 * x4[k] + x3[k]) / detaz**2
            ) * deltat

            x5[k + 1] = x5[k] + (
                (qe * x6[k] - qe * x5[k])
                + (v1 * np.exp(-rv1 * x4[k]) * x4[k] - v1 * np.exp(-rv1 * x5[k]) * x5[k]) / detaz
                + D1 * (x6[k] - 2 * x5[k] + x4[k]) / detaz**2
            ) * deltat

            x6[k + 1] = x6[k] + (
                (qf_k * xf - qe * x6[k] - qu_k * x6[k] + v1 * np.exp(-rv1 * x5[k]) * x5[k] - v2 * np.exp(-rv2 * x6[k]) * x6[k]) / detaz
            ) * deltat + (
                (D3 * (x7[k] - x6[k]) - D2 * (x6[k] - x5[k])) / detaz**2
            ) * deltat

            x7[k + 1] = x7[k] + (
                (qu_k * x6[k] - qu_k * x7[k]) + (v3 * np.exp(-rv3 * x6[k]) * x6[k] - v3 * np.exp(-rv3 * x7[k]) * x7[k])
            ) / detaz * deltat + (
                D3 * (x8[k] - 2 * x7[k] + x6[k]) / detaz**2
            ) * deltat

            x8[k + 1] = x8[k] + (
                (qu_k * x7[k] - qu_k * x8[k]) + (v3 * np.exp(-rv3 * x7[k]) * x7[k] - v3 * np.exp(-rv3 * x8[k]) * x8[k])
            ) / detaz * deltat + (
                D3 * (x9[k] - 2 * x8[k] + x7[k]) / detaz**2
            ) * deltat

            x9[k + 1] = x9[k] + (
                (qu_k * x8[k] - qu_k * x9[k]) + (v3 * np.exp(-rv3 * x8[k]) * x8[k] - v4 * np.exp(-rv4 * x9[k]) * x9[k])
            ) / detaz * deltat + (
                (D4 * (x10[k] - x9[k]) - D3 * (x9[k] - x8[k])) / detaz**2
            ) * deltat

            x10[k + 1] = x10[k] + (
                (qu_k * x9[k] - qu_k * x10[k]) + (v4 * np.exp(-rv4 * x9[k]) * x9[k])
            ) / detaz * deltat - (
                D4 * (x10[k] - x9[k]) / detaz**2
            ) * deltat

        density_floor = self.RHO_WATER * 1e6
        for arr in (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
            arr[:] = np.maximum(arr, density_floor)

        Y = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
        density_profile = Y[-1, :]
        cf_floor = np.array([self.d2c(d / 1e6) for d in density_profile], dtype=np.float64)
        new_state = density_profile.copy()

        return cf_floor, new_state


DEFAULT_THICKENER_STATE = ThickenerModel.c2d(DEFAULT_INITIAL_CONCENTRATION_PROFILE) * 1e6
