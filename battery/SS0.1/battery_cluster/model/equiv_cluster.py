r"""
equiv_cluster.py
================
**整簇等效电芯构造器（方案 A）**

假设一簇内所有电芯参数一致、温度一致、初始 SOC 一致、无弱组/均衡差异，
则 N_S 串 × N_P 并的整簇可以**严格**折叠为 **1 个等效电芯**：

    I_eq = I_cluster,   SOC_eq = SOC_cell,   V_eq = V_cluster
    Q_eq = N_P * Q_cell
    R0_eq = (N_S/N_P) * R0_cell,   R1_eq = (N_S/N_P) * R1_cell
    C1_eq = (N_P/N_S) * C1_cell,   R2_eq = (N_S/N_P) * R2_cell
    C2_eq = (N_P/N_S) * C2_cell,   R3_eq = (N_S/N_P) * R3_cell
    OCV_eq(SOC,T) = N_S * OCV_cell(SOC,T),  ocv_off_eq = N_S * ocv_off_cell
    alpha1/alpha2/eta/h_ref/tau_h/tau_rest 不变,  tau3 不变

推导要点：
  * 并联 N_P 个相同电芯 -> 等电压、均流：R/N_P, C*N_P, Q*N_P，时间常数不变；
  * 再串联 N_S 组 -> 同电流、电压相加：R*N_S, C/N_S；
  * 支路电压与 SOC 积分的比例项、历史项同比例缩放，故逐点严格相等。

注意
----
  * 本模块只适用于"簇内完全一致"的均值电芯模型：无法体现 SOC 分化、并联环流、
    弱组/单体过压欠压、木桶效应与单体级保护；
  * 返回对象的 V_cluster 为整簇电压，V_cell = V_cluster/N_S 为等效单体电压。

对外接口
--------
    build_equivalent_cluster(...) -> EquivalentCluster
    EquivalentCluster.reset(soc0)
    EquivalentCluster.state()          -> {'SOC','V_cluster','V_cell'}
    EquivalentCluster.step(I_A)        -> {'t','I','SOC','V_cluster','V_cell'}
    EquivalentCluster.current_limits() -> (放电允许电流, 充电允许电流) [A]
"""

import numpy as np

from battery_pack import battery_pack
from RC2_SOC import _ocv_bilinear


class EquivalentCluster:
    """整簇等效为 1 个电芯后的簇模型（含 SOC 窗口/倍率限流）。"""

    def __init__(self, pack, N_S, N_P, Q_cell, soc_lo, soc_hi, i_max,
                 T=25.0, v0=None):
        self.pack = pack
        self.N_S = int(N_S)
        self.N_P = int(N_P)
        self.Q_cell = float(Q_cell)
        self.q_stack_ah = self.N_P * self.Q_cell
        self.soc_lo = float(soc_lo)
        self.soc_hi = float(soc_hi)
        self.i_max = float(i_max)
        self.T = float(T)
        self._v0 = None if v0 is None else float(v0)
        self._v = self._v0

    # ---------------- 状态 ----------------
    def reset(self, soc0):
        """复位到给定物理 SOC（比例 0~1），端电压回到初始开路电压。"""
        self.pack.reset(soc0=float(soc0))
        self._v = self._v0
        return self

    @property
    def V_cluster(self):
        """最近一步的整簇端电压（V）；尚未推进时为初始开路电压。"""
        return self._v

    def state(self):
        st = self.pack.state()
        soc = float(np.mean(st["SOC"]))
        return {"SOC": soc,
                "V_cluster": self._v,
                "V_cell": None if self._v is None else self._v / self.N_S}

    # ---------------- 推进 ----------------
    def step(self, I_A):
        """用整簇电流 I_A（A，放电为正）推进一个 dt。"""
        out = self.pack.step(float(I_A), T=self.T)
        self._v = float(out["V_term"])
        return {"t": float(out["t"]), "I": float(I_A),
                "SOC": float(np.mean(out["SOC"])),
                "V_cluster": self._v,
                "V_cell": self._v / self.N_S}

    # ---------------- 限流 ----------------
    def current_limits(self, soc=None):
        """返回 (放电允许电流, 充电允许电流)，均为正值。

        口径与 BMSConstraint.limit 一致：SOC 窗口的单步预测 + 倍率上限 i_max；
        母线级还需要再按各簇求和。
        """
        if soc is None:
            soc = self.state()["SOC"]
        room_dis = max(float(soc) - self.soc_lo, 0.0)
        room_chg = max(self.soc_hi - float(soc), 0.0)
        i_dis = min(self.i_max,
                    room_dis * 3600.0 * self.q_stack_ah / self.pack.dt)
        i_chg = min(self.i_max,
                    room_chg * 3600.0 * self.q_stack_ah / self.pack.dt)
        return float(i_dis), float(i_chg)


def build_equivalent_cluster(*, N_S, N_P, Q_cell,
                             R0, R1, C1, alpha1, R2, C2, alpha2,
                             ocv_dis_csv, ocv_chg_csv, soc0,
                             dt=5.0, memory_limit=1500,
                             R3=0.0, tau3=0.0, ocv_off=0.0, eta=1.0,
                             T=25.0, tau_h=120.0, h_ref=0.5,
                             tau_rest=1800.0, hys_i_eps=0.1,
                             soc_lo=0.0, soc_hi=1.0, i_max=None,
                             store_history=False):
    """方案 A：构造整簇等效电芯（battery_pack(1, N_p=1) + 参数缩放）。

    Parameters 与 battery_pack.set_cell_parameters 同名同义，均为**单体**参数；
    OCV 表为**单体**表（本函数内部按 N_S 放大）。
    """
    if i_max is None:
        i_max = N_P * Q_cell                      # 默认 1C（簇电流）

    ratio = float(N_S) / float(N_P)               # R 缩放
    inv = float(N_P) / float(N_S)                 # C 缩放

    pack = battery_pack(1, N_p=1, dt=dt, memory_limit=memory_limit,
                        store_history=store_history,
                        ocv_dis_csv=ocv_dis_csv, ocv_chg_csv=ocv_chg_csv,
                        tau_h=tau_h, h_ref=h_ref, tau_rest=tau_rest,
                        hys_i_eps=hys_i_eps)

    # OCV 表按串联数放大：OCV_eq = N_S * OCV_cell（充/放两支路都要放大）
    pack.ocv_table = np.ascontiguousarray(pack.ocv_table * float(N_S))
    if pack.br_on == 1:
        pack.ocv_br = np.ascontiguousarray(pack.ocv_br * float(N_S))

    pack.set_cell_parameters(
        R0=ratio * R0, R1=ratio * R1, C1=inv * C1, alpha1=alpha1,
        R2=ratio * R2, C2=inv * C2, alpha2=alpha2,
        Q=N_P * Q_cell, soc0=soc0, eta=eta,
        ocv_off=float(N_S) * ocv_off, h0=h_ref,
        R3=ratio * R3, tau3=tau3)

    # 初始开路电压（V1=V2=0，仅 OCV 与迟滞项），不消耗状态
    ocv_d = float(_ocv_bilinear(float(soc0), float(T), pack.soc_grid,
                                pack.t_grid, pack.ocv_table))
    v0 = ocv_d + float(N_S) * float(ocv_off)
    if pack.br_on == 1:
        ocv_c = float(_ocv_bilinear(float(soc0), float(T), pack.soc_grid_b,
                                    pack.t_grid_b, pack.ocv_br))
        v0 += float(h_ref) * (ocv_c - ocv_d)

    pack.reset(soc0=float(soc0))
    return EquivalentCluster(pack, N_S, N_P, Q_cell,
                             soc_lo=soc_lo, soc_hi=soc_hi, i_max=i_max,
                             T=T, v0=v0)
