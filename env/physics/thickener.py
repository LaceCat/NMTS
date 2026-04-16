"""
Concentrate thickener physical model.

This module keeps the external concentration variables (`c`, `c_uf`, `Cf`)
in volume-fraction semantics, while the internal 10-layer PDE state uses
mixed density in g/m^3.

The implementation below is aligned with the original project file
`原项目/RL+浓密脱水/RL+浓密脱水/thickener_dewatering_env.py`.
"""

from __future__ import annotations

import numpy as np


DEFAULT_INITIAL_CONCENTRATION_PROFILE = np.array(
    [
        0.10491146,
        0.13310611,
        0.18802230,
        0.26703723,
        0.36470816,
        0.47286261,
        0.51170143,
        0.55906143,
        0.61044374,
        0.65373351,
    ],
    dtype=np.float64,
)


DEFAULT_THICKENER_STATE = np.array(
    [
        1087360.78,
        1113503.56,
        1168209.25,
        1257069.82,
        1387532.99,
        1567697.77,
        1644371.81,
        1748660.48,
        1877873.17,
        2002540.54,
    ],
    dtype=np.float64,
)


class ThickenerModel:
    RADIUS = 15.0
    TOTAL_HEIGHT = 5.79
    FEED_HEIGHT = 4.0
    N_LAYERS = 10

    def __init__(self):
        self.area = np.pi * self.RADIUS**2
        self.detaz = self.TOTAL_HEIGHT / self.N_LAYERS

        lijing = 50
        v1dr = 1 / 40 * lijing + 1 / 4
        self._params = [
            0.547678836441774,
            0.476947700914306,
            0.940451926648319,
            0.952633313167728,
            v1dr,
            16.5914198572464,
            1.14944107415579,
            1.73671299891156,
            7.50829943915578e-06,
            8.48008041543593e-06,
            10.62486693809181e-07,
            13.58704316829464e-07,
        ]

    @staticmethod
    def c2d(c):
        rho_0 = 4.27
        rho_w = 1.0
        c = np.asarray(c, dtype=np.float64)
        return rho_0 / (rho_0 - (rho_0 - rho_w) * c)

    @staticmethod
    def d2c(d):
        rho_0 = 4.27
        rho_w = 1.0
        d = np.asarray(d, dtype=np.float64)
        return rho_0 * (d - rho_w) / ((rho_0 - 1) * d)

    def step(self, q_uf, qf, cf, thickener_state):
        D1, D2, D3, D4 = self._params[0:4]
        v1, v2, v3, v4 = self._params[4:8]
        rv1, rv2, rv3, rv4 = self._params[8:12]

        t_final = 1 / 60
        deltat = 1 / 60
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
            qe = (qf_k - qu_k) / A
            xf = cf_density[k]
            qushang = qu_k * 0.8

            x1[k + 1] = (
                x1[k]
                - qushang * x1[k]
                + (
                    -v1 * np.exp(-rv1 * x1[k]) * x1[k] / detaz
                    + D1 * (x2[k] - x1[k]) / (detaz**2)
                )
                * deltat
            )

            x2[k + 1] = (
                x2[k]
                + (qushang * x1[k] - qushang * x2[k])
                + (
                    (
                        qe * x3[k]
                        - qe * x2[k]
                        + v1 * np.exp(-rv1 * x1[k]) * x1[k]
                        - v1 * np.exp(-rv1 * x2[k]) * x2[k]
                    )
                    / detaz
                    + (D1 * (x3[k] - x2[k]) - D1 * (x2[k] - x1[k])) / (detaz**2)
                )
                * deltat
            )

            x3[k + 1] = (
                x3[k]
                + (qushang * x2[k] - qushang * x3[k])
                + (
                    (
                        qe * x4[k]
                        - qe * x3[k]
                        + v1 * np.exp(-rv1 * x2[k]) * x2[k]
                        - v1 * np.exp(-rv1 * x3[k]) * x3[k]
                    )
                    / detaz
                    + (D1 * (x4[k] - x3[k]) - D1 * (x3[k] - x2[k])) / (detaz**2)
                )
                * deltat
            )

            x4[k + 1] = (
                x4[k]
                + (qushang * x3[k] - qushang * x4[k])
                + (
                    (
                        qe * x5[k]
                        - qe * x4[k]
                        + v1 * np.exp(-rv1 * x3[k]) * x3[k]
                        - v1 * np.exp(-rv1 * x4[k]) * x4[k]
                    )
                    / detaz
                    + (D1 * (x5[k] - x4[k]) - D1 * (x4[k] - x3[k])) / (detaz**2)
                )
                * deltat
            )

            x5[k + 1] = (
                x5[k]
                + (qushang * x4[k] - qushang * x5[k])
                + (
                    (
                        qe * x6[k]
                        - qe * x5[k]
                        + v1 * np.exp(-rv1 * x4[k]) * x4[k]
                        - v1 * np.exp(-rv1 * x5[k]) * x5[k]
                    )
                    / detaz
                    + (D1 * (x6[k] - x5[k]) - D1 * (x5[k] - x4[k])) / (detaz**2)
                )
                * deltat
            )

            x6[k + 1] = (
                x6[k]
                + (qushang * x5[k] - qushang * x6[k])
                + ((qf_k * xf - qe * x6[k] - qushang * x6[k]) / detaz) * deltat
                + (
                    v1 * np.exp(-rv1 * x5[k]) * x5[k]
                    - v2 * np.exp(-rv2 * x6[k]) * x6[k]
                )
                + ((D3 * (x7[k] - x6[k]) - D2 * (x6[k] - x5[k])) / (detaz**2)) * deltat
            )

            x7[k + 1] = (
                x7[k]
                + (
                    (
                        qushang * x6[k]
                        - qu_k * x7[k]
                        + v3 * np.exp(-rv3 * x6[k]) * x6[k]
                        - v3 * np.exp(-rv3 * x7[k]) * x7[k]
                    )
                    / detaz
                    + (D3 * (x8[k] - x7[k]) - D3 * (x7[k] - x6[k])) / (detaz**2)
                )
                * deltat
            )

            x8[k + 1] = (
                x8[k]
                + (
                    (
                        qu_k * x7[k]
                        - qu_k * x8[k]
                        + v3 * np.exp(-rv3 * x7[k]) * x7[k]
                        - v3 * np.exp(-rv3 * x8[k]) * x8[k]
                    )
                    / detaz
                    + (D3 * (x9[k] - x8[k]) - D3 * (x8[k] - x7[k])) / (detaz**2)
                )
                * deltat
            )

            x9[k + 1] = (
                x9[k]
                + (
                    (
                        qu_k * x8[k]
                        - qu_k * x9[k]
                        + v3 * np.exp(-rv3 * x8[k]) * x8[k]
                        - v4 * np.exp(-rv4 * x9[k]) * x9[k]
                    )
                    / detaz
                    + (D4 * (x10[k] - x9[k]) - D3 * (x9[k] - x8[k])) / (detaz**2)
                )
                * deltat
            )

            x10[k + 1] = (
                x10[k]
                + (qu_k * x9[k] - qu_k * x10[k])
                + (
                    (v4 * np.exp(-rv4 * x9[k]) * x9[k]) / detaz
                    - D4 * (x10[k] - x9[k]) / (detaz**2)
                )
                * deltat
            )

        for i in range(L + 1):
            x1[i] = max(0.0, float(self.d2c(x1[i] / 1e6)))
            x2[i] = max(0.0, float(self.d2c(x2[i] / 1e6)))
            x3[i] = max(0.0, float(self.d2c(x3[i] / 1e6)))
            x4[i] = max(0.0, float(self.d2c(x4[i] / 1e6)))
            x5[i] = max(0.0, float(self.d2c(x5[i] / 1e6)))
            x6[i] = max(0.0, float(self.d2c(x6[i] / 1e6)))
            x7[i] = max(0.0, float(self.d2c(x7[i] / 1e6)))
            x8[i] = max(0.0, float(self.d2c(x8[i] / 1e6)))
            x9[i] = max(0.0, float(self.d2c(x9[i] / 1e6)))
            x10[i] = max(0.0, float(self.d2c(x10[i] / 1e6)))

        Y = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
        cf_floor = Y[-1, :]
        new_state = self.c2d(cf_floor) * 1e6

        return cf_floor, new_state
