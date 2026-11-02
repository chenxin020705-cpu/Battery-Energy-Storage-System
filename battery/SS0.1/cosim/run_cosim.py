r"""
run_cosim.py
============
**4 簇并联直流母线 + PCS 联合仿真（架构 A：前馈限幅弱耦合，首个版本）**

拓扑
----
    簇1..簇4 (各 216S3P, 120 Ah, 方案 A 等效为 1 个电芯) 并联在直流母线
        直流母线 (u_dc, I_dc) ──► PCS 直流侧 ──► 交流侧 (P_ac)

时间尺度处理
------------
  * 电池：dt = 5 s（与辨识口径一致）；
  * PCS ：5 s 宏步下采用 `pcs_control_sys.steady_power_pu()` 的稳态有功律
          （10% 下垂 + 参考限幅），忽略 0.01 s 电流/参考滤波与 0.1 s 高通暂态；
          这些暂态相对 5 s 宏步的能量贡献可忽略。完整的动态 PCS 仍可用
          `ct.input_output_response` 在短时窗内单独验证。

耦合步骤（每 5 s）
------------------
    ① 由 u_dc 与各簇 SOC 求允许充/放电流 -> 母线可用功率；
    ② 限幅 P_ref（PCS 内部还会叠加频率支撑并再限 ±1.1 pu）；
    ③ PCS 稳态有功 -> P_ac；
    ④ P_dc = P_ac/η（放电）或 P_ac·η（充电），I_dc = 1000·P_dc/u_dc；
    ⑤ 各簇按 BMS 限流做 water-filling 均流，逐簇 step()，回传 V_term/SOC；
    ⑥ u_dc 取各簇端电压均值（四簇参数一致时完全相等）。

输出
----
    cosim/output/cosim_result.png   （4 联图：功率/母线电压/簇电流/母线电流）
    cosim/output/dc_ac_result.png   （4 联图：直流-交流电压/电流/功率/调制比）
    + 终端摘要（能量/限幅/Ah 一致性校验、DC→AC 换算示例）

运行
----
    cd Storage202609/Storage_system/SS0.1/cosim
    F:/Study/Anaconda3/envs/AIkun/python.exe run_cosim.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))          # SS0.1/cosim
_ROOT = os.path.dirname(_HERE)                              # SS0.1
BC_DIR = os.path.join(_ROOT, "battery_cluster")
PCS_DIR = os.path.join(_ROOT, "pcs")
for _p in (os.path.join(BC_DIR, "model"), os.path.join(BC_DIR, "params"), PCS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from equiv_cluster import build_equivalent_cluster          # noqa: E402
from save_params import load_params                         # noqa: E402
from control_sys import pcs_control_sys, pcs_per_unit       # noqa: E402

# ============ 1. 配置 ============
DT = 5.0                         # 宏步 / 电池步长 (s)
T_END_H = 4.0                    # 仿真时长 (h)
T_AMB = 25.0
ETA = 0.98                       # PCS 效率
N_CLU = 4
N_S, N_P, Q_CELL = 216, 3, 40.0  # 单簇: 216 串, 3 并, 单体 40 Ah
SOC_LO = 50.0 / 480.0            # 0.10417
SOC_DISP0 = 0.50                 # 初始"显示 SOC"
I_MAX_AC_PU = 1.1                # PCS 交流电流上限 (pu)
OUT_DIR = os.path.join(_HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

OCV_DIS = os.path.join(BC_DIR, "ocv", "ocv_branch_dis_seg.csv")
OCV_CHG = os.path.join(BC_DIR, "ocv", "ocv_branch_chg_seg.csv")

# ---- 工况: P_ref[kW] 与 f[Hz]（零阶保持） ----
P_DIS_KW = 150.0                 # 4 簇约 0.5C
P_CHG_KW = -150.0
T_DIS = (0.5, 1.0)               # h, 放电段
T_CHG = (1.5, 2.0)               # h, 充电段
F_LOW = (0.25, 0.75)             # h, f=49.8 Hz (欠频 -> 增发)
F_HIGH = (2.5, 3.0)              # h, f=50.2 Hz (过频 -> 减发)


def _in(x, lo, hi):
    return (x >= lo) & (x < hi)


def split_current(i_total, i_dis, i_chg):
    """把母线总电流按各簇允许电流分配（water-filling）。

    返回 (各簇电流, 未分配完的剩余电流)。放电为正、充电为负。
    """
    lo = -np.asarray(i_chg, dtype=float)          # 充电下限（负）
    hi = np.asarray(i_dis, dtype=float)           # 放电上限（正）
    I = np.zeros_like(hi)
    remain = float(i_total)
    active = list(range(hi.size))
    for _ in range(hi.size + 1):
        if not active or abs(remain) < 1e-12:
            break
        share = remain / len(active)
        for j in list(active):
            if share >= 0.0:
                add = min(share, hi[j] - I[j])
            else:
                add = max(share, lo[j] - I[j])
            I[j] += add
            remain -= add
        if share >= 0.0:
            active = [j for j in active if hi[j] - I[j] > 1e-9]
        else:
            active = [j for j in active if I[j] - lo[j] > 1e-9]
    return I, remain


def main(show=True):
    # ---- 2. 参数与模型 ----
    rec = load_params(script="identify_bank01.py")
    if rec is None:
        raise FileNotFoundError("未找到 identify_bank01 记录")
    P = rec["params"]
    R0 = float(P.get("R0", 1.3e-3))
    R1, C1 = float(P["R1"]), float(P["C1"])
    ALPHA1 = float(P.get("alpha1", 0.53))
    R2, C2 = float(P["R2"]), float(P["C2"])
    ALPHA2 = float(P.get("alpha2", 0.95))
    R3 = float(P.get("R3", 0.0))
    TAU3 = float(P.get("tau3", 0.0))
    OCV_OFF = float(P.get("ocv_off", 0.0))
    q_cell = float(P.get("Q", Q_CELL))
    SOC_HI = float(P.get("soc_hi", SOC_LO + 226.0 / 480.0))

    SOC0 = SOC_LO + (SOC_HI - SOC_LO) * SOC_DISP0
    print(f"[参数] {os.path.basename(rec.get('file', ''))} | "
          f"Q={q_cell:g}Ah soc_hi={SOC_HI:.4f} | SOC0={SOC0:.4f}")

    clusters = [
        build_equivalent_cluster(
            N_S=N_S, N_P=N_P, Q_cell=q_cell,
            R0=R0, R1=R1, C1=C1, alpha1=ALPHA1,
            R2=R2, C2=C2, alpha2=ALPHA2,
            R3=R3, tau3=TAU3, ocv_off=OCV_OFF,
            ocv_dis_csv=OCV_DIS, ocv_chg_csv=OCV_CHG,
            soc0=SOC0, dt=DT, memory_limit=1500, eta=1.0, T=T_AMB,
            tau_h=120.0, h_ref=0.5, tau_rest=1800.0, hys_i_eps=0.1,
            soc_lo=SOC_LO, soc_hi=SOC_HI, i_max=N_P * q_cell,
            store_history=False)
        for _ in range(N_CLU)
    ]
    q_stack_ah = clusters[0].q_stack_ah
    print(f"[拓扑] {N_CLU} 簇 × ({N_S}S{N_P}P, {q_stack_ah:g}Ah) 并联; "
          f"合计 {N_CLU * q_stack_ah:g}Ah, 1C={N_CLU * clusters[0].i_max:g}A")

    pu = pcs_per_unit()
    pcs = pcs_control_sys(q_mode="PQ", pf=0.98)
    S_BASE = pu.S_base_kVA
    print(f"[PCS ] 额定 {pu.P_rated_kW:g}kW, S_base={S_BASE:.2f}kVA; "
          f"5 s 宏步用稳态有功律（下垂+限幅）")

    # ---- 3. 工况 ----
    n = int(round(T_END_H * 3600.0 / DT)) + 1
    t_s = np.arange(n) * DT
    t_h = t_s / 3600.0
    f_Hz = np.full(n, 50.0)
    f_Hz[_in(t_h, *F_LOW)] = 49.8
    f_Hz[_in(t_h, *F_HIGH)] = 50.2
    P_ref = np.zeros(n)
    P_ref[_in(t_h, *T_DIS)] = P_DIS_KW
    P_ref[_in(t_h, *T_CHG)] = P_CHG_KW

    # ---- 4. 联合仿真 ----
    u_dc = float(clusters[0].V_cluster)
    soc = np.full(N_CLU, SOC0)
    P_ac_arr = np.zeros(n)
    P_frc_arr = np.zeros(n)
    P_bat_arr = np.zeros(n)
    P_ref_eff = np.zeros(n)
    u_dc_arr = np.zeros(n)
    I_clu = np.zeros((n, N_CLU))
    I_dc_req = np.zeros(n)
    curt = np.zeros(n, dtype=bool)
    p_ac_max = I_MAX_AC_PU * S_BASE

    for k in range(n):
        # ① 电池侧可用功率（u_dc 取上一宏步的端电压）
        lim = [c.current_limits(soc[j]) for j, c in enumerate(clusters)]
        i_dis = np.array([v[0] for v in lim])
        i_chg = np.array([v[1] for v in lim])
        p_dis = u_dc * i_dis.sum() / 1e3
        p_chg = u_dc * i_chg.sum() / 1e3
        hi = min(p_dis, p_ac_max)
        lo = -min(p_chg, p_ac_max)

        # ② 限幅 P_ref
        p_ref_k = float(np.clip(P_ref[k], lo, hi))
        P_ref_eff[k] = p_ref_k

        # ③ PCS 稳态有功（含频率支撑与 ±1.1 pu 限幅）
        p_pu, p_frc_pu = pcs.steady_power_pu(f_Hz[k] / 50.0, 1.0,
                                             p_ref_k / S_BASE)
        p_ac = p_pu * S_BASE
        P_frc_arr[k] = p_frc_pu * S_BASE

        # ④ 功率 -> 母线电流
        p_dc = p_ac / ETA if p_ac >= 0 else p_ac * ETA
        i_req = 1e3 * p_dc / u_dc
        I_dc_req[k] = i_req

        # ⑤ 各簇均流并推进
        I_j, remain = split_current(i_req, i_dis, i_chg)
        curt[k] = abs(remain) > 1e-6
        V = np.zeros(N_CLU)
        for j, c in enumerate(clusters):
            out = c.step(I_j[j])
            soc[j] = out["SOC"]
            V[j] = out["V_cluster"]
        I_clu[k] = I_j
        u_dc = float(V.mean())

        # ⑥ 记录
        P_ac_arr[k] = p_ac
        P_bat_arr[k] = u_dc * I_j.sum() / 1e3
        u_dc_arr[k] = u_dc

    # ---- 5. 统计与校验 ----
    e_ac_dis = float(np.sum(np.maximum(P_ac_arr, 0.0)) * DT / 3600.0)
    e_ac_chg = float(np.sum(np.minimum(P_ac_arr, 0.0)) * DT / 3600.0)
    e_bat_dis = float(np.sum(np.maximum(P_bat_arr, 0.0)) * DT / 3600.0)
    e_bat_chg = float(np.sum(np.minimum(P_bat_arr, 0.0)) * DT / 3600.0)
    u_mean = float(np.mean(u_dc_arr))
    ah_bat = float(np.sum(I_clu.sum(axis=1)) * DT / 3600.0)
    ah_soc = float(np.sum((SOC0 - soc) * q_stack_ah))
    print("\n========= 4 簇 + PCS 联合仿真 =========")
    print(f"功率    : P_ref {P_ref.min():.1f}~{P_ref.max():.1f} | "
          f"P_ac {P_ac_arr.min():.1f}~{P_ac_arr.max():.1f} | "
          f"P_bat {P_bat_arr.min():.1f}~{P_bat_arr.max():.1f} kW"
          f" | P_frc {P_frc_arr.min():.1f}~{P_frc_arr.max():.1f} kW")
    print(f"母线电压: {u_dc_arr.min():.2f}~{u_dc_arr.max():.2f} V | "
          f"首末 {u_dc_arr[0]:.2f} -> {u_dc_arr[-1]:.2f} V")
    print("SOC     : " + " | ".join(
        f"簇{j + 1} {soc[j] * 100:.2f}%" for j in range(N_CLU))
          + f" (初值 {SOC0 * 100:.2f}%, 窗口 [{SOC_LO * 100:.2f}, {SOC_HI * 100:.2f}]%)")
    print(f"限幅    : 触发 {int(curt.sum())} / {n} 步 | "
          f"max|ΣI_j − I_dc| = {np.max(np.abs(I_clu.sum(axis=1) - I_dc_req)):.3f} A"
          f" | max|P_ref_eff − P_ref| = {np.max(np.abs(P_ref_eff - P_ref)):.1f} kW")
    print(f"能量    : AC 放 {e_ac_dis:.2f} / 充 {e_ac_chg:.2f} kWh | "
          f"DC 放 {e_bat_dis:.2f} / 充 {e_bat_chg:.2f} kWh")
    if e_ac_dis > 0:
        print(f"          放电 DC/AC = {e_bat_dis / e_ac_dis:.4f} (期望 1/η={1 / ETA:.4f})")
    if abs(e_ac_chg) > 0:
        print(f"          充电 DC/AC = {e_bat_chg / e_ac_chg:.4f} (期望 η={ETA:.4f})")
    print(f"          由 SOC 变化对应 {ah_soc:.3f} Ah | ∫I dt = {ah_bat:.3f} Ah "
          f"| 偏差 {ah_soc - ah_bat:+.3e} Ah (均值电压 {u_mean:.1f} V)")
    print(f"一致性  : 各簇端电压极差 = "
          f"{np.ptp([c.V_cluster for c in clusters]) * 1e3:.3f} mV")

    # ---- 5b. 直流 <-> 交流换算 ----
    # 交流侧为理想电网 (U_t = 1 pu = 400 V 线电压 RMS, Q = 0):
    #   P_ac = √3·U_ac·I_ac  ->  I_ac = P_ac/(√3·U_ac)
    #   P_dc = u_dc·I_dc      ->  I_dc = P_dc/u_dc
    #   P_ac = η·P_dc (放电) / P_dc = η·P_ac (充电)
    #   两电平 SVPWM 线性区:  u_dc >= √2·U_ac,  m = √2·U_ac/u_dc <= 1
    U_AC_RMS = 400.0
    I_BASE = pu.I_base_A
    I_AC_RMS = P_ac_arr * 1e3 / (np.sqrt(3.0) * U_AC_RMS)
    I_DC_AC = (np.where(P_ac_arr >= 0.0, P_ac_arr / ETA, P_ac_arr * ETA)
               * 1e3 / u_dc_arr)
    m_idx = np.sqrt(2.0) * U_AC_RMS / u_dc_arr
    u_ac_lim = u_dc_arr / np.sqrt(2.0)

    k0 = int(round(0.90 * 3600.0 / DT))
    print(f"[DC→AC @{t_h[k0]:.2f} h] u_dc={u_dc_arr[k0]:.2f} V, "
          f"I_dc={I_clu[k0].sum():.2f} A, P_dc={P_bat_arr[k0]:.2f} kW")
    print(f"           -> P_ac={P_ac_arr[k0]:.2f} kW, "
          f"U_ac={U_AC_RMS:.0f} V (ll, RMS), "
          f"I_ac={I_AC_RMS[k0]:.2f} A ({I_AC_RMS[k0] / I_BASE:.4f} pu), "
          f"m={m_idx[k0]:.4f}, u_dc 下限={np.sqrt(2.0) * U_AC_RMS:.1f} V")

    # ---- 6. 绘图 ----
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(t_h, P_ref, "k--", lw=1.0, label="P_ref (指令)")
    axes[0].plot(t_h, P_ac_arr, "r-", lw=1.2, label="P_ac (PCS)")
    axes[0].plot(t_h, P_bat_arr, "b-", lw=1.0, label="P_bat (DC)")
    axes[0].set_ylabel("功率 [kW]")
    axes[0].grid(True); axes[0].legend(loc="upper right")

    axes[1].plot(t_h, u_dc_arr, "g-", lw=1.2, label="u_dc")
    axes[1].set_ylabel("直流母线电压 [V]")
    axes[1].grid(True); axes[1].legend(loc="upper right")

    for j in range(N_CLU):
        axes[2].plot(t_h, I_clu[:, j], lw=0.9, label=f"簇{j + 1}")
    axes[2].set_ylabel("簇电流 [A]")
    axes[2].grid(True); axes[2].legend(loc="upper right", fontsize=8, ncol=2)

    axes[3].plot(t_h, I_dc_req, "k--", lw=1.0, label="I_dc 需求")
    axes[3].plot(t_h, I_clu.sum(axis=1), "r-", lw=1.0, label="ΣI_j 实际")
    axes[3].set_ylabel("母线电流 [A]"); axes[3].set_xlabel("时间 [h]")
    axes[3].grid(True); axes[3].legend(loc="upper right")

    fig.suptitle("4 簇并联直流母线 + PCS 联合仿真（方案 A 等效簇）")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "cosim_result.png")
    fig.savefig(out_png, dpi=130)
    print(f"\n结果图已保存: {out_png}")

    # ---- 7. 直流-交流电压/电流图 ----
    fig2, ax2 = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    ax2[0].plot(t_h, u_dc_arr, "g-", lw=1.2, label="u_dc (直流母线)")
    ax2[0].plot(t_h, np.full(n, U_AC_RMS), "m--", lw=1.1,
                label="U_ac (交流线电压 RMS)")
    ax2[0].plot(t_h, u_ac_lim, "c:", lw=1.0, label="u_dc/√2 (可输出上限)")
    ax2[0].set_ylabel("电压 [V]")
    ax2[0].grid(True); ax2[0].legend(loc="upper right", fontsize=9)

    ax2[1].plot(t_h, I_clu.sum(axis=1), "b-", lw=1.0, label="I_dc (实际)")
    ax2[1].plot(t_h, I_DC_AC, "k--", lw=1.0, label="I_dc = P_dc/u_dc")
    ax2[1].plot(t_h, I_AC_RMS, "r-", lw=1.0, label="I_ac (RMS, Q=0)")
    ax2[1].set_ylabel("电流 [A]")
    ax2[1].grid(True); ax2[1].legend(loc="upper right", fontsize=9)

    ax2[2].plot(t_h, P_bat_arr, "b-", lw=1.0, label="P_dc = u_dc·I_dc")
    ax2[2].plot(t_h, P_ac_arr, "r-", lw=1.0, label="P_ac = √3·U_ac·I_ac")
    ax2[2].plot(t_h, P_ref, "k--", lw=0.9, label="P_ref")
    ax2[2].set_ylabel("功率 [kW]")
    ax2[2].grid(True); ax2[2].legend(loc="upper right", fontsize=9)

    ax2[3].plot(t_h, m_idx, color="purple", lw=1.1,
                label="m = √2·U_ac/u_dc")
    ax2[3].axhline(1.0, color="r", ls="--", lw=0.8,
                   label="线性调制上限 m=1")
    ax2[3].set_ylabel("调制比 m"); ax2[3].set_xlabel("时间 [h]")
    ax2[3].set_ylim(0.0, 1.1)
    ax2[3].grid(True); ax2[3].legend(loc="upper right", fontsize=9)

    fig2.suptitle("PCS 直流侧 ↔ 交流侧 电压/电流换算 (U_t=1 pu, Q=0)")
    fig2.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    fig2.text(0.5, 0.008,
              "P_ac=√3·U_ac·I_ac,  P_dc=u_dc·I_dc,  "
              "P_ac=η·P_dc(放电)/P_dc=η·P_ac(充电),  "
              "m=√2·U_ac/u_dc ≤ 1",
              ha="center", fontsize=9)
    out_dc_ac = os.path.join(OUT_DIR, "dc_ac_result.png")
    fig2.savefig(out_dc_ac, dpi=130)
    print(f"直流-交流图已保存: {out_dc_ac}")

    if show:
        plt.show()
    return {"t_h": t_h, "P_ref": P_ref, "P_ac": P_ac_arr,
            "P_bat": P_bat_arr, "u_dc": u_dc_arr, "I_cluster": I_clu,
            "I_ac": I_AC_RMS, "m": m_idx, "SOC_end": soc,
            "png": out_png, "png_dc_ac": out_dc_ac}


if __name__ == "__main__":
    main()
