r"""
test_bms_bank01.py
==================
**单簇(Bank01)版的 test_bms_250701.py** —— 用 2025-07-01 Bank01 的**簇级**实测数据,
与"含 BMS 约束层的 pack 模型"做对比试验。

与 test_bms_250701.py 的差异(仅"去掉 4 簇并联")
----------------------------------------------
    电堆 : N_S=216, N_P=12 (3并/模组 x 4簇并), Q_等效=480 Ah, 1C=480 A, 用"堆级"列
    单簇 : N_S=216, N_P= 3 (仅模组内 3 并)   , Q_等效=120 Ah, 1C=120 A, 用"簇级"列
    电压等级相同(216S); SOC 是强度量 -> 工作区间 SOC_LO/SOC_HI 与电堆一致,
    但其对应的 Ah 变为 1/4 (可放电量 12.5 Ah @显示0%, 可充电量 69 Ah @显示100%)。

数据列(GBK; 行尾多一逗号 -> 位置索引)
-------------------------------------
    簇级 : 簇总电压(21) 簇电流(23) 簇SOC(26) 簇最高电压(30) 簇最低电压(32)
    堆级 : 可充电量(17) 可放电量(18) —— 4 个 bank 文件里是**重复的公共量**,
           单簇对标时按"均分"折算 (/4), 并注意实测电流并非精确均分(见诊断输出)。
    模组温度: 列 740..847

模型
----
    battery_pack(N_S, N_p=3) + BMSConstraint(q_stack_ah=120, i_max=120)
    OCV : 可选 ① 三层双支路表(ocv/ocv_branch_dis|chg_seg.csv, 推荐) 或 ② 默认单表
    ECM : 可选 ① 读 params/identify_param/ 里 identify_bank01.py 的最新记录 或 ② 本文件默认值
    电流: 簇电流直接作为指令 I_ref (放电为正、充电为负), 逐点闭环: 读状态->限幅->单步

打包布局(SS0.1, 自包含: 不引用包外文件)
----------------------------------------
    SS0.1/battery_cluster/model/    battery_pack.py, bms_layer.py, RC2_SOC.py, RC2_Numba.py
    SS0.1/battery_cluster/ocv/      ocv_branch_dis_seg.csv, ocv_branch_chg_seg.csv
    SS0.1/battery_cluster/params/   save_params.py, identify_param/<辨识记录>.json
    SS0.1/battery_cluster/data/     Bank01_20250701.csv
    SS0.1/battery_cluster/run/      test_bms_bank01.py (本文件)
    SS0.1/battery_cluster/output/   bms_vs_bank01.png (运行生成)
    SS0.1/pcs/                      control_sys.py (储能变流器控制级模型, 与本簇目录同级)

运行
----
    cd Storage202609/Storage_system/SS0.1/battery_cluster/run
    F:/Study/Anaconda3/envs/AIkun/python.exe test_bms_bank01.py

输出
----
    SS0.1/battery_cluster/output/ 下: bms_vs_bank01.png (7 联图)

BMS 约束层开关
--------------
    欠压故障(低端) : UV_FAULT_ENABLE (默认 False; 该簇实测 Vmin 最低 3.086 V, 不触 2.90 V)
    高端钳位       : OV_CLAMP_ENABLE (默认 False; 实测不支持"电压触发置100%", 仅作对比)
                     触发后**锁存**: 显示SOC 置 100% 并从该锚点继续演化, 不回退。
                     另给出离线"假设"曲线 soc_work_ov(Vmax>=V_OV_CUT -> 锁存至100%), 无需重跑。
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))          # SS0.1/battery_cluster/run
_ROOT = os.path.dirname(_HERE)                              # SS0.1/battery_cluster
MODEL_DIR = os.path.join(_ROOT, "model")                    # 物理层 + BMS 约束层
PARAM_DIR = os.path.join(_ROOT, "params")                   # save_params + 辨识记录
OCV_DIR = os.path.join(_ROOT, "ocv")                        # 双支路 OCV 表
DATA_DIR = os.path.join(_ROOT, "data")                      # 实测数据
OUT_DIR = os.path.join(_ROOT, "output")                     # 图件输出
os.makedirs(OUT_DIR, exist_ok=True)
for _p in (MODEL_DIR, PARAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from battery_pack import battery_pack                       # noqa: E402
from bms_layer import BMSConstraint                         # noqa: E402
from save_params import load_params                         # noqa: E402

# ============ 1. 配置 ============
DATA = os.path.join(DATA_DIR, "Bank01_20250701.csv")

DT = 5.0
MEMORY_LIMIT = 1500
T_AMB = 25.0                 # 体温口径: 25℃ 常数; 若要实测 T(t) 见下方 T_USE_MEASURED
T_USE_MEASURED = False       # True -> 用列 740..847 的模块温度均值作为 T(t)
SKIP_H = 20.0                # 开头 20 h 电流恒 0

# ---- 拓扑: 单簇 ----
N_S, N_P, Q_CELL = 216, 3, 40     # 单簇: 216 串, 3 并
Q_CLU = N_P * Q_CELL                # 120 Ah
I_1C = Q_CLU                        # 120 A
# ---- 工作区间(物理SOC, 与电堆一致); 单簇对应 Ah: 可放 12.5 / 可充 69 ----
SOC_LO = 50.0 / 480.0               # 0.10417 (显示0% 时可放电量 50 Ah /480)
FULL_SCALE_AH = 226.0               # 显示 0%~100% 对应物理电量(堆级); 单簇 = /4
SOC_HI = SOC_LO + FULL_SCALE_AH / 480.0     # 0.575

# ---- OCV / ECM 来源 ----
USE_3LAYER_OCV = True        # True: 三层双支路表(与辨识一致, 推荐)
ECM_FROM_RECORD = True       # True: 读 params/identify_param/ 中 identify_bank01.py 的最新记录

# 分段保序回归修正后的 OCV 表(由 fix_ocv_segmented.py 生成; 全 SOC 区间非减)
OCV_DIS = os.path.join(OCV_DIR, "ocv_branch_dis_seg.csv")
OCV_CHG = os.path.join(OCV_DIR, "ocv_branch_chg_seg.csv")
TAU_H, H_REF, TAU_REST, HYS_EPS = 120.0, 0.5, 1800.0, 0.1

# ---- ECM 默认值(ECM_FROM_RECORD=False 或读不到记录时使用; 同 test_bms_250701.py) ----
R0, R1, C1, ALPHA1 = 2.1e-3, 8e-4, 5e4, 0.8
R2, C2, ALPHA2 = 1.2e-3, 2.5e5, 0.7
R3, TAU3, OCV_OFF = 0.0, 0.0, 0.0     # 第三条慢支路(辨识记录可覆盖)
_soc_hi_ecm = SOC_HI                # 读记录时会被记录的 soc_hi 覆盖

if ECM_FROM_RECORD:
    try:
        _rec = load_params(script="identify_bank01.py")
    except Exception:
        _rec = None
    if _rec:
        _P = _rec["params"]
        R0 = float(_P.get("R0", R0)); R1 = float(_P["R1"]); C1 = float(_P["C1"])
        ALPHA1 = float(_P.get("alpha1", 1.0))
        R2 = float(_P["R2"]); C2 = float(_P["C2"]); ALPHA2 = float(_P.get("alpha2", 1.0))
        R3 = float(_P.get("R3", 0.0)); TAU3 = float(_P.get("tau3", 0.0))
        OCV_OFF = float(_P.get("ocv_off", 0.0))
        _soc_hi_ecm = float(_P.get("soc_hi", SOC_HI)); Q_CELL = float(_P.get("Q", Q_CELL))
        Q_CLU = N_P * Q_CELL; I_1C = Q_CLU
        SOC_HI = _soc_hi_ecm          # 让 BMS 工作上限与辨识口径一致(显示100% 对应的物理SOC)
        print(f"[ECM] 读入 identify_bank01 记录: {os.path.basename(_rec.get('file', ''))} | "
              f"Q={Q_CELL:g}Ah, soc_hi={SOC_HI:.4f}")
    else:
        print("[ECM] 未找到 identify_bank01 记录 -> 用本文件默认值")

# ---- 弱组与欠压故障(单簇专用) ----
# 诊断结论: 250701 的欠压故障**不在 Bank01** —— 该簇"簇最低电压"最低仅 3.086 V,
#           从未触及 2.90 V 阈值(而堆级数据在 1.913 h 确实触发)。故本簇**不做故障模拟**。
WEAK_ENABLE = False          # False: 全簇单体一致(无弱组); True: 才在 WEAK_GROUP 上抬高 R0
R0_WEAK = R0                 # WEAK_ENABLE=False 时必须等于 R0
UV_FAULT_ENABLE = False      # False: 关闭 BMS 欠压故障状态机(v_uv_cut 置 0 -> 永不触发)
WEAK_GROUP = 0
V_UV_CUT = 2.90              # 仅 UV_FAULT_ENABLE=True 时生效
V_RECOVER = 3.20
T_HOLD = 0.94 * 3600.0

# ---- 高端钳位(锁存式) ----
# 判据: max(单体电压) >= V_OV_CUT -> 显示SOC 重整定为 SOC_OV(100%)并锁存;
#       同步重整定显示容量: 可充电量 -> 0 Ah、可放电量 -> 69 Ah, 之后随显示状态继续演化。
OV_CLAMP_ENABLE = True
V_OV_CUT = 3.40              # 仅 OV_CLAMP_ENABLE=True 时生效(候选阈值)
SOC_OV = 1.0
CAP_TRIG_HI_AH = 69.0        # 触发时 显示可放电量 重整定值 (Ah)
CAP_TRIG_LO_AH = 0.0         # 触发时 显示可充电量 重整定值 (Ah)
CAP_TOTAL_AH = 69.0          # 显示容量总量(与 SOC 显示解耦; = 物理满容量 0.575*Q)
Q_DISP_AH = 83.0             # 方案3: 显示SOC 独立积分容量基准 (Ah)

# ---- 静置重估(可选): 长放电后静置期, 显示SOC/容量向目标值一阶弛豫 ----
REST_REEST_ENABLE = True
REST_I_EPS = 2.0             # 静置判定电流阈值 (A)
REST_T_MIN = 300.0           # 最短连续静置 (s)
REST_LOOKBACK_S = 1800.0     # 前置放电回看窗口 (s)
REST_DISCHARGE_AH_MIN = 20.0 # 回看窗口净放电门槛 (Ah)
REST_TAU = 600.0             # 弛豫时间常数 (s)
REST_SOC_TARGET = 0.24       # 重估目标显示SOC (标定自实测 ~24%)
REST_CAP_DIS_TARGET_AH = 13.5    # 目标可放电量 (实测 ~13.25)
REST_CAP_CHG_TARGET_AH = 55.25   # 目标可充电量 (实测 ~55.25)

CLU = {"时间": 0, "簇总电压": 21, "簇电流": 23, "簇SOC": 26,
       "簇最高电压": 30, "簇最低电压": 32}
STK = {"堆可充电量": 17, "堆可放电量": 18}
T_POS = list(range(740, 848))
V_COL0, N_CELL_V = 40, 216        # 单体电压 V1..V216 (位置索引 40..255)

# ============ 2. 读取并重采样当日数据 ============
raw = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                  usecols=list(CLU.values()), names=list(CLU.keys()))
raw["时间"] = pd.to_datetime(raw["时间"])
for c in list(CLU.keys())[1:]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.set_index("时间").sort_index()

_sr = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                  usecols=list(STK.values()), names=list(STK.keys()))
for c in list(STK.keys()):
    _sr[c] = pd.to_numeric(_sr[c], errors="coerce")
_sr.index = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                        usecols=[0], names=["t"])["t"].pipe(pd.to_datetime)

# 单体电压 V1..V216 (用于"实际全簇均值单体电压")
_cv = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                  usecols=list(range(V_COL0, V_COL0 + N_CELL_V)))
_cv = _cv.apply(pd.to_numeric, errors="coerce")
_cv.index = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                        usecols=[0], names=["t"])["t"].pipe(pd.to_datetime)
_cv = _cv.sort_index()

_t0 = raw.index[0]
_cut = _t0 + pd.Timedelta(hours=SKIP_H)
raw, _sr, _cv = raw[raw.index >= _cut], _sr[_sr.index >= _cut].sort_index(), _cv[_cv.index >= _cut]

df = raw.resample(f"{DT:g}s").mean().interpolate(limit_direction="both").dropna()
_cap = _sr.resample(f"{DT:g}s").mean().reindex(df.index).interpolate(
    limit_direction="both").fillna(0.0)
_Vc = _cv.resample(f"{DT:g}s").mean().reindex(df.index).interpolate(
    limit_direction="both").to_numpy(dtype=float)
Vmean_data_cells = np.nanmean(_Vc, axis=1)          # 实际全簇均值单体电压(216 只均值)

# 0h 实测可放电量(单簇) —— 作为显示容量初值锚点, 传给 BMSConstraint 由其内部演化
CAP_DIS0_AH = float(_cap["堆可放电量"].to_numpy(dtype=float)[0] / 4.0)

t_h = (df.index - df.index[0]).total_seconds() / 3600.0
I_ref = df["簇电流"].to_numpy(dtype=float)                 # 单簇: 用簇电流作指令
soc_start_disp = float(df["簇SOC"].iloc[0]) / 100.0
print(f"[数据] {os.path.basename(DATA)} | {len(raw)} 行 -> {len(df)} 行 (dt={DT:g}s)")
print(f"[拓扑] N_S={N_S}, N_P={N_P} (单簇) | Q_clu={Q_CLU:g} Ah | 1C={I_1C:g} A | "
      f"窗口物理SOC [{SOC_LO:.4f}, {SOC_HI:.4f}] (簇 Ah: {SOC_LO*Q_CLU:.1f}~{SOC_HI*Q_CLU:.1f})")
print(f"[起始] 显示SOC {soc_start_disp*100:.0f}%")
print(f"[OCV] {'三层双支路表' if USE_3LAYER_OCV else '默认单表'} | "
      f"ECM: R0={R0*1e3:.3f}mΩ R1={R1*1e3:.3f}mΩ C1={C1:g} α1={ALPHA1:g} "
      f"R2={R2*1e3:.3f}mΩ C2={C2:g} α2={ALPHA2:g}")

# T: 常数 或 实测均值
if T_USE_MEASURED:
    _tb = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                      usecols=T_POS).apply(pd.to_numeric, errors="coerce")
    _tb.index = pd.read_csv(DATA, encoding="gbk", header=None, skiprows=1,
                            usecols=[0], names=["t"])["t"].pipe(pd.to_datetime)
    T_in = (_tb[_tb.index >= _cut].resample(f"{DT:g}s").mean().mean(axis=1)
            .reindex(df.index).interpolate(limit_direction="both").to_numpy(dtype=float))
    print(f"[温度] 实测 T(t): {np.nanmin(T_in):.1f}~{np.nanmax(T_in):.1f}℃")
else:
    T_in = T_AMB
    print(f"[温度] 常数 {T_AMB:g}℃")

# ============ 3. 物理层 + BMS 约束层 ============
_ocv_kw = ({"ocv_dis_csv": OCV_DIS, "ocv_chg_csv": OCV_CHG,
            "tau_h": TAU_H, "h_ref": H_REF, "tau_rest": TAU_REST, "hys_i_eps": HYS_EPS}
           if USE_3LAYER_OCV else {})
pack = battery_pack(N_S, N_p=N_P, dt=DT, memory_limit=MEMORY_LIMIT,
                    store_history=False, **_ocv_kw)
R0_arr = np.full(N_S * N_P, R0, dtype=float)
if WEAK_ENABLE:
    R0_arr[WEAK_GROUP * N_P:(WEAK_GROUP + 1) * N_P] = R0_WEAK
print(f"[弱组/故障] WEAK_ENABLE={WEAK_ENABLE} | UV_FAULT_ENABLE={UV_FAULT_ENABLE}"
      + (f" | R0_WEAK={R0_WEAK*1e3:.3f} mΩ" if WEAK_ENABLE else " (全簇单体一致, 不做故障模拟)"))
print(f"[高端钳位] OV_CLAMP_ENABLE={OV_CLAMP_ENABLE}"
      + (f" | 阈值 {V_OV_CUT:.3f} V -> 显示 {SOC_OV*100:.0f}%" if OV_CLAMP_ENABLE else " (关闭)"))
print(f"[显示SOC] 独立积分基准 Q_disp={Q_DISP_AH:.1f} Ah (方案3)")
print(f"[静置重估] REST_REEST_ENABLE={REST_REEST_ENABLE}"
      + (f" | 目标 SOC={REST_SOC_TARGET*100:.0f}%, 可放={REST_CAP_DIS_TARGET_AH:.2f} Ah, "
         f"可充={REST_CAP_CHG_TARGET_AH:.2f} Ah, tau={REST_TAU:g}s"
         if REST_REEST_ENABLE else " (关闭)"))
pack.set_cell_parameters(R0=R0_arr, R1=R1, C1=C1, alpha1=ALPHA1,
                         R2=R2, C2=C2, alpha2=ALPHA2,
                         Q=Q_CELL, soc0=SOC_HI, eta=1.0,
                         R3=R3, tau3=TAU3, ocv_off=OCV_OFF, h0=H_REF)

bms = BMSConstraint(pack, soc_lo=SOC_LO, soc_hi=SOC_HI,
                    q_stack_ah=Q_CLU, i_max=I_1C, soc_taper=0.0,
                    v_uv_cut=(V_UV_CUT if UV_FAULT_ENABLE else 0.0),
                    v_recover=V_RECOVER, t_hold=T_HOLD,
                    i_fault=0.0, soc_fault=0.0,
                    uv_confirm=2, uv_discharge_only=True,
                    v_ov_cut=(V_OV_CUT if OV_CLAMP_ENABLE else None), soc_ov=SOC_OV,
                    cap_total_ah=CAP_TOTAL_AH, cap_dis0_ah=CAP_DIS0_AH,
                    cap_ov_dis_ah=CAP_TRIG_HI_AH, cap_ov_chg_ah=CAP_TRIG_LO_AH,
                    rest_reest=REST_REEST_ENABLE, i_rest_eps=REST_I_EPS,
                    t_rest_min=REST_T_MIN, rest_lookback_s=REST_LOOKBACK_S,
                    rest_discharge_ah_min=(REST_DISCHARGE_AH_MIN
                                           if REST_REEST_ENABLE else None),
                    tau_rest=REST_TAU, rest_soc_target=REST_SOC_TARGET,
                    rest_cap_dis_target_ah=REST_CAP_DIS_TARGET_AH,
                    rest_cap_chg_target_ah=REST_CAP_CHG_TARGET_AH,
                    q_disp_ah=Q_DISP_AH, soc_disp0=soc_start_disp)

phy0 = float(bms.to_phy_soc(soc_start_disp))
print(f"[起始] 物理SOC = {phy0:.4f}")
res = bms.run(I_ref, T=T_in, soc0=phy0)

# ============ 4. 对齐 + 派生量 ============
# step() 返回 t=(k+1)*dt(步末), 而实测 t_h[0]=0(步首); 回移一个 dt,
# 使"应用 I_ref[k] 后的第 k 个输出"与实测第 k 个采样在时间上对齐。
t_m = (res["t"] - DT) / 3600.0
n = min(len(t_h), t_m.size)
t_h, t_m = t_h[:n], t_m[:n]
I_ref, I_act = I_ref[:n], res["I"][:n]
SOC_phy = res["SOC_phy"][:n]
SOC_work = res["SOC_work"][:n] * 100.0
V_mod = res["V_term"][:n]
Vmin_mod = res["V_cell_min"][:n]
Vmax_mod = res["V_cell_max"][:n]
Vmean_mod = V_mod / N_S                        # 模型"均值单体"电压(等效单体, 全簇一致)
fault = res["fault"][:n]

# 未钳位基准显示SOC(物理SOC -> 工作SOC 映射; 本簇故障关闭, 不含故障置零)
SOC_work_base = np.clip((SOC_phy - SOC_LO) / (SOC_HI - SOC_LO), 0.0, 1.0) * 100.0

# 高端钳位"假设"推演(离线, 锁存式, 不改模型): 首次 Vmax >= V_OV_CUT 即置显示SOC=SOC_OV,
# 并**锁存**: 之后以该步物理SOC 为锚、按物理SOC 增量继续演化(从100%起安时积分), 不回退。
# 该钳位只改显示、不反馈物理层, 故与开启 OV_CLAMP_ENABLE 时严格一致, 无需重跑。
_ov_idx = np.where(Vmax_mod >= V_OV_CUT)[0]
ov_hit = np.zeros(n, dtype=bool)
SOC_work_ov = SOC_work_base.copy()
if _ov_idx.size:
    _k0 = int(_ov_idx[0])
    ov_hit[_k0:] = True
    _latched = np.clip(SOC_OV + (SOC_phy - SOC_phy[_k0]) / (SOC_HI - SOC_LO),
                       0.0, 1.0) * 100.0
    SOC_work_ov[_k0:] = _latched[_k0:]          # 仅触发后锁存, 触发前仍用映射曲线

V_data = df["簇总电压"].to_numpy(dtype=float)[:n]
I_data = df["簇电流"].to_numpy(dtype=float)[:n]
SOC_data = df["簇SOC"].to_numpy(dtype=float)[:n]
Vmax_data = df["簇最高电压"].to_numpy(dtype=float)[:n]
Vmin_data = df["簇最低电压"].to_numpy(dtype=float)[:n]
# 堆级可充/放电量 -> 单簇折算(均分假设, /4)
capdis_data = _cap["堆可放电量"].to_numpy(dtype=float)[:n] / 4.0
capchg_data = _cap["堆可充电量"].to_numpy(dtype=float)[:n] / 4.0
# 显示容量由 BMSConstraint 输出(与 SOC 解耦; 初值锚定与高压触发重整定均在其内部完成)
capdis_mod = res["Cap_dis"][:n]
capchg_mod = res["Cap_chg"][:n]
_k0_mod = int(_ov_idx[0]) if _ov_idx.size else None

# ============ 5. 统计 ============
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

print("\n========= 单簇 Bank01: 实测 vs 模型 =========")
print(f"电压 : 实测 {V_data.min():.1f}~{V_data.max():.1f} V | "
      f"模型 {V_mod.min():.1f}~{V_mod.max():.1f} V | RMSE {rmse(V_mod, V_data):.1f} V")
print(f"电流 : 实测 {I_data.min():.1f}~{I_data.max():.1f} A | "
      f"模型 {I_act.min():.1f}~{I_act.max():.1f} A")
print(f"  电流对齐: t[0] 模型 {t_m[0]:.6f} h / 实测 {t_h[0]:.6f} h | "
      f"argmax 模型 {int(np.argmax(I_act))} / 实测 {int(np.argmax(I_data))} | "
      f"max|I_act−I_data| {np.max(np.abs(I_act - I_data)):.3e} A")
print(f"SOC  : 实测(显示) {SOC_data.min():.0f}~{SOC_data.max():.0f} % | "
      f"模型(工作) {SOC_work.min():.1f}~{SOC_work.max():.1f} % | RMSE {rmse(SOC_work, SOC_data):.2f} %")
print(f"物理SOC(模型): {SOC_phy.min():.4f} ~ {SOC_phy.max():.4f} (窗口 [{SOC_LO:.4f}, {SOC_HI:.4f}])")
print(f"簇最高电压: 实测 {Vmax_data.min():.3f}~{Vmax_data.max():.3f} V | "
      f"模型 {Vmax_mod.min():.3f}~{Vmax_mod.max():.3f} V | RMSE {rmse(Vmax_mod, Vmax_data):.3f} V")
print(f"簇最低电压: 实测 {Vmin_data.min():.3f}~{Vmin_data.max():.3f} V | "
      f"模型 {Vmin_mod.min():.3f}~{Vmin_mod.max():.3f} V | RMSE {rmse(Vmin_mod, Vmin_data):.3f} V")
print(f"簇极差(高-低): 实测 {(Vmax_data-Vmin_data).mean()*1e3:.1f} mV(均) | "
      f"模型 {(Vmax_mod-Vmin_mod).mean()*1e3:.1f} mV(均)")
# ---- 模型"均值单体" vs 实测全簇均值 ----
_Vmc = Vmean_data_cells[:n]
_gap = (Vmean_mod - _Vmc) * 1e3
print(f"均值单体   : 模型 {Vmean_mod.min():.4f}~{Vmean_mod.max():.4f} V | "
      f"实测(216只均值) {_Vmc.min():.4f}~{_Vmc.max():.4f} V")
print(f"均值单体差距: RMSE {np.sqrt(np.mean(_gap**2)):.2f} mV | 偏置 {_gap.mean():+.2f} mV | "
      f"max|差| {np.abs(_gap).max():.2f} mV @ h={t_h[int(np.argmax(np.abs(_gap)))]:.2f}")
print(f"  校核: 簇总电压/216 = {V_data.min()/N_S:.4f}~{V_data.max()/N_S:.4f} V (应≈实测全簇均值)")
print(f"可放电量(/4): 实测 {capdis_data.min():.1f}~{capdis_data.max():.1f} | "
      f"模型 {capdis_mod.min():.1f}~{capdis_mod.max():.1f} Ah")
print(f"可充电量(/4): 实测 {capchg_data.min():.1f}~{capchg_data.max():.1f} | "
      f"模型 {capchg_mod.min():.1f}~{capchg_mod.max():.1f} Ah")
print(f"  触发重整定 : 可放电量->{CAP_TRIG_HI_AH:.1f} Ah, 可充电量->{CAP_TRIG_LO_AH:.1f} Ah"
      + (f" @ h={t_m[_k0_mod]:.3f}" if _k0_mod is not None else " (未触发)"))
print(f"  可放电量 RMSE: {rmse(capdis_mod, capdis_data):.2f} Ah")
print(f"  可充电量 RMSE: {rmse(capchg_mod, capchg_data):.2f} Ah")

# ---- 0 h(分析窗口起点)显示容量偏差诊断 ----
_i0 = 0
print(f"[0h] 显示SOC 实测 {SOC_data[_i0]:.1f}% | 模型 {SOC_work[_i0]:.1f}% | "
      f"物理SOC {SOC_phy[_i0]:.4f}")
print(f"[0h] 可放电量 实测 {capdis_data[_i0]:.2f} | 模型 {capdis_mod[_i0]:.2f} Ah | "
      f"偏差 {capdis_mod[_i0]-capdis_data[_i0]:+.2f}")
print(f"[0h] 可充电量 实测 {capchg_data[_i0]:.2f} | 模型 {capchg_mod[_i0]:.2f} Ah | "
      f"偏差 {capchg_mod[_i0]-capchg_data[_i0]:+.2f}")
print(f"[0h] 可放+可充 实测 {capdis_data[_i0]+capchg_data[_i0]:.2f} | "
      f"模型 {capdis_mod[_i0]+capchg_mod[_i0]:.2f} Ah (应≈CAP_TOTAL_AH={CAP_TOTAL_AH:.2f})")

# ---- 静置段诊断: 长放电后无电流期间, 实测是否重估 SOC / 可充放电量 ----
# 若 |I|<2 A 的静置段内实测 SOC/容量仍在变化, 而模型(安时积分, 静置 ΔAh=0)不变,
# 则说明实测存在"静置重估"机制。
I_REST_EPS = 2.0
_rest = np.abs(I_data) < I_REST_EPS
_edges = np.diff(np.r_[False, _rest, False].astype(int))
_s0 = np.where(_edges == 1)[0]
_e0 = np.where(_edges == -1)[0] - 1
_segs = [(int(s), int(e)) for s, e in zip(_s0, _e0) if (e - s) * DT >= 300]   # >=5 min
_segs.sort(key=lambda se: se[1] - se[0], reverse=True)
print("\n---- 静置段(>=5 min): 实测 vs 模型 的段内变化 ----")
print("  #  起止[h]          时长[min] 前30min放电Ah | ΔSOC实测 ΔSOC模型 | "
      "Δ可放实测 Δ可放模型 | Δ可充实测 Δ可充模型")
_pre = int(30 * 60 / DT)
for j, (s, e) in enumerate(_segs[:6]):
    _a = max(0, s - _pre)
    _pre_ah = float(np.sum(I_data[_a:s]) * DT / 3600.0)
    print(f"  {j}  {t_h[s]:.3f}~{t_h[e]:.3f} {(e-s)*DT/60:8.1f} {_pre_ah:+12.2f} | "
          f"{SOC_data[e]-SOC_data[s]:+7.2f} {SOC_work[e]-SOC_work[s]:+7.2f} | "
          f"{capdis_data[e]-capdis_data[s]:+8.2f} {capdis_mod[e]-capdis_mod[s]:+8.2f} | "
          f"{capchg_data[e]-capchg_data[s]:+8.2f} {capchg_mod[e]-capchg_mod[s]:+8.2f}")

# 段0 内部轨迹(判断重估是阶跃还是渐变)
if _segs:
    _s, _e = _segs[0]
    _step = max(1, int(300 / DT))      # 每 5 min 采样
    print("\n---- 段0 内部轨迹(每5min) ----")
    print("  t[h]    I[A]   SOC实测  capdis实测 capchg实测   Vmax    Vmin")
    for kk in range(_s, _e + 1, _step):
        print(f"  {t_h[kk]:.3f} {I_data[kk]:+7.1f} {SOC_data[kk]:7.2f} "
              f"{capdis_data[kk]:10.2f} {capchg_data[kk]:10.2f} "
              f"{Vmax_data[kk]:7.3f} {Vmin_data[kk]:7.3f}")

# 静置结束 -> 充电开始 的衔接检查(每1min)
print("\n---- 静置结束->充电开始 衔接(每1min) ----")
print("  t[h]    I[A]  SOC实测 SOC模型  capdis实测 capdis模型  capchg实测 capchg模型")
for kk in range(max(0, int(2.80 * 3600 / DT)), min(n, int(3.05 * 3600 / DT)),
                max(1, int(60 / DT))):
    print(f"  {t_h[kk]:.3f} {I_data[kk]:+6.1f} {SOC_data[kk]:7.2f} {SOC_work[kk]:7.2f} "
          f"{capdis_data[kk]:10.2f} {capdis_mod[kk]:10.2f} "
          f"{capchg_data[kk]:10.2f} {capchg_mod[kk]:10.2f}")

# ---- 充/放电段斜率 -> 反推显示容量(较长段, 降低 SOC 1% 量化影响) ----
I_ON = 10.0
_on = np.abs(I_data) >= I_ON
_ed2 = np.diff(np.r_[False, _on, False].astype(int))
_segs_i = [(int(s), int(e)) for s, e in zip(np.where(_ed2 == 1)[0],
                                            np.where(_ed2 == -1)[0] - 1)
           if (e - s) * DT >= 600]                     # >=10 min
_segs_i.sort(key=lambda se: se[1] - se[0], reverse=True)
print("\n---- 充/放电段斜率反推显示容量(>=10min) ----")
print("  #  起止[h]        时长[min] 平均I[A]  净Ah  ΔSOC实测 ΔSOC模型 | Q实测Ah Q模型Ah")
for j, (s, e) in enumerate(_segs_i[:6]):
    _dur = (e - s) * DT
    _ah = float(np.sum(I_data[s:e + 1]) * DT / 3600.0)
    _dm = SOC_data[e] - SOC_data[s]
    _dk = SOC_work[e] - SOC_work[s]
    _qm = abs(_ah) / (abs(_dm) / 100.0) if abs(_dm) > 0.5 else float("nan")
    _qk = abs(_ah) / (abs(_dk) / 100.0) if abs(_dk) > 0.5 else float("nan")
    print(f"  {j}  {t_h[s]:.3f}~{t_h[e]:.3f} {_dur/60:8.1f} {_ah*3600/_dur:+8.1f} "
          f"{_ah:+7.2f} {_dm:+8.2f} {_dk:+8.2f} | {_qm:7.2f} {_qk:7.2f}")
print(f"  当前设定: Q_clu={Q_CLU:.1f} Ah | 显示SOC Q_disp={Q_DISP_AH:.1f} Ah | "
      f"CAP_TOTAL={CAP_TOTAL_AH:.1f} Ah")
print(f"BMS 限幅步数 : {sum(1 for r in res['reasons'] if r)} / {n}")

# ---- 分段显示SOC RMSE(定位斜率段与静置段的误差差异) ----
print("\n---- 分段显示SOC RMSE ----")
for _name, _a, _b in [("触发前 0~0.739", 0.0, 0.739),
                      ("放电 0.739~1.914", 0.739, 1.914),
                      ("静置 1.914~2.854", 1.914, 2.854),
                      ("充电 2.854~3.232", 2.854, 3.232),
                      ("静置 3.232~3.994", 3.232, 3.994)]:
    _m = (t_h >= _a) & (t_h < _b)
    if _m.any():
        print(f"  {_name:20s}: RMSE {rmse(SOC_work[_m], SOC_data[_m]):5.2f} %  "
              f"| 偏置 {np.mean(SOC_work[_m]-SOC_data[_m]):+5.2f} %  (n={int(_m.sum())})")

# ---- 高端钳位统计(基于模型最高单体 Vmax_mod; 阈值 V_OV_CUT; 锁存式) ----
_ov_n = int(ov_hit.sum())
print(f"高端钳位(阈值 {V_OV_CUT:.3f} V, 锁存): 锁存 {_ov_n} / {n} 步"
      + (f" | 首次触发 h={t_m[int(np.argmax(ov_hit))]:.3f}, 锚点物理SOC "
         f"{SOC_phy[int(np.argmax(ov_hit))]:.4f}" if _ov_n else " (未触发)"))
print(f"  实测对照    : 实测Vmax最大 {Vmax_data.max():.3f} V; 100%期间Vmax峰值 "
      f"{Vmax_data[SOC_data >= 99.5].max() if (SOC_data >= 99.5).any() else float('nan'):.3f} V")
print(f"  (参考)物理映射 RMSE: 锁存 {rmse(SOC_work_ov, SOC_data):.2f} % | "
      f"未锁存 {rmse(SOC_work_base, SOC_data):.2f} %")

if UV_FAULT_ENABLE:
    cross = np.where(Vmin_mod <= V_UV_CUT)[0]
    if cross.size:
        print(f"Vmin<=阈值首次 : h={t_m[cross[0]]:.3f} | Vmin={Vmin_mod[cross[0]]:.3f} V (阈值 {V_UV_CUT:.2f})")
    if fault.any():
        kf = int(np.argmax(fault)); kl = n - 1 - int(np.argmax(fault[::-1]))
        print(f"欠压故障(FAULT): 触发 h={t_m[kf]:.3f} -> 恢复 h={t_m[kl]:.3f} | "
              f"持续 {(kl - kf) * DT / 3600.0:.2f} h ({int(fault.sum())} 步)")
    else:
        print("欠压故障(FAULT): 未触发")
else:
    print("欠压故障(FAULT): 已关闭 (该簇实测 Vmin 最低 "
          f"{Vmin_data.min():.3f} V, 未触及 {V_UV_CUT:.2f} V; 故障不在本簇)")
print(f"Vmin(模型)    : {Vmin_mod.min():.3f} V @ h={t_m[int(np.argmin(Vmin_mod))]:.3f}")
print(f"Vmin(实测)    : {Vmin_data.min():.3f} V @ h={t_h[int(np.argmin(Vmin_data))]:.3f}")

# ---- 弱组 R0 标定参考(单簇口径; 仅 WEAK_ENABLE=True 时给出) ----
if WEAK_ENABLE:
    I_c = I_act / N_P
    V_other = (V_mod + I_c * (R0_WEAK - R0)) / N_S
    i19 = int(np.argmin(np.abs(t_m - 1.907)))
    if abs(I_c[i19]) > 1.0:
        dR_need = (V_other[i19] - V_UV_CUT) / I_c[i19]
        print(f"弱组标定参考  : h=1.907 处 V_other={V_other[i19]:.4f} V, 单体电流={I_c[i19]:.1f} A "
              f"-> 需 ΔR={dR_need * 1e3:.2f} mΩ (即 R0_WEAK = {R0 + dR_need:.5g} Ω)")
    else:
        print("弱组标定参考  : h=1.907 处单体电流≈0, 无法反解 ΔR; 请用该簇 Vmin 首触时刻重标")

# ---- SOC 相位对齐诊断 ----
M = 120
seg_m = SOC_work[M:-M]
best_lag, best_r, best_e = 0, -2.0, float("nan")
for lag in range(-M, M + 1):
    seg_d = SOC_data[M + lag:len(SOC_data) - M + lag]
    if seg_d.size != seg_m.size:
        continue
    r = float(np.corrcoef(seg_m, seg_d)[0, 1])
    if r > best_r:
        best_lag, best_r, best_e = lag, r, rmse(seg_m, seg_d)
print(f"SOC 相位对齐 : 最佳滞后 {best_lag * DT:+.0f} s | 相关 {best_r:.3f} | 对齐后 RMSE {best_e:.2f} %")
print("===========================================")

# ============ 6. 绘图 ============
fig, axes = plt.subplots(7, 1, figsize=(13, 20), sharex=True)

axes[0].plot(t_h, V_data, "b-", lw=1.0, label="实测 簇总电压")
axes[0].plot(t_m, V_mod, "r-", lw=1.0, label="模型 簇电压")
axes[0].set_ylabel("电压 [V]"); axes[0].grid(True); axes[0].legend(loc="upper right")

axes[1].plot(t_h, I_data, "b-", lw=1.0, label="实测/指令 簇电流")
axes[1].plot(t_m, I_act, "r--", lw=1.0, label="模型 实际电流")
axes[1].set_ylabel("电流 [A]"); axes[1].grid(True); axes[1].legend(loc="upper right")

axes[2].plot(t_h, SOC_data, "b-", lw=1.2, label="实测 显示SOC")
axes[2].plot(t_m, SOC_work, "r-", lw=1.2, label="模型 工作SOC")
if _ov_n > 0 and not OV_CLAMP_ENABLE:
    axes[2].plot(t_m, SOC_work_ov, "g--", lw=1.0,
                 label=f"模型 高端钳位假设·锁存 (≥{V_OV_CUT:.2f}V→{SOC_OV*100:.0f}%)")
axes[2].set_ylabel("SOC [%]"); axes[2].set_ylim(-5, 105); axes[2].grid(True)
axes[2].legend(loc="upper right")

axes[3].plot(t_m, SOC_phy * 100.0, "k-", lw=1.2, label="模型 物理SOC")
axes[3].axhline(SOC_LO * 100.0, color="darkorange", ls="--", lw=1, label=f"工作下限 {SOC_LO*100:.1f}%")
axes[3].axhline(SOC_HI * 100.0, color="darkorange", ls="--", lw=1, label=f"工作上限 {SOC_HI*100:.1f}%")
axes[3].set_ylabel("物理SOC [%]"); axes[3].grid(True); axes[3].legend(loc="upper right")

axes[4].plot(t_h, capdis_data, "b-", lw=1.0, label="实测 可放电量(/4)")
axes[4].plot(t_m, capdis_mod, "r--", lw=1.0, label="模型 可放电量(解耦整定)")
axes[4].plot(t_h, capchg_data, "c-", lw=1.0, label="实测 可充电量(/4)")
axes[4].plot(t_m, capchg_mod, "m--", lw=1.0, label="模型 可充电量(解耦整定)")
axes[4].set_ylabel("可充/放电量 [Ah]")
axes[4].grid(True); axes[4].legend(loc="upper right")

# (6) 簇最高/最低单体电压 (实测 vs 模型)
axes[5].plot(t_h, Vmax_data, "b-", lw=1.0, label="实测 簇最高电压")
axes[5].plot(t_m, Vmax_mod, "r--", lw=1.0, label="模型 最高单体电压")
axes[5].plot(t_h, Vmin_data, "c-", lw=1.0, label="实测 簇最低电压")
axes[5].plot(t_m, Vmin_mod, "m--", lw=1.0, label="模型 最低单体电压")
axes[5].set_ylabel("单体电压 [V]")
axes[5].grid(True); axes[5].legend(loc="upper right", fontsize=8)

# (7) 模型"均值单体" vs 实测全簇均值: 差值
axes[6].plot(t_h, (Vmean_mod - Vmean_data_cells[:n]) * 1e3, "k-", lw=1.0,
             label=f"模型均值单体 − 实测全簇均值 (RMSE "
                   f"{np.sqrt(np.mean(_gap**2)):.2f} mV, 偏置 {_gap.mean():+.2f} mV)")
axes[6].axhline(0, color="gray", lw=0.8)
axes[6].axhline(_gap.mean(), color="r", ls="--", lw=0.9,
                label=f"平均偏置 {_gap.mean():+.2f} mV")
axes[6].set_ylabel("单体均值差 [mV]"); axes[6].set_xlabel("Time [h]")
axes[6].grid(True); axes[6].legend(loc="upper right", fontsize=8)

if fault.any():
    kf = int(np.argmax(fault)); kl = n - 1 - int(np.argmax(fault[::-1]))
    for i, ax in enumerate(axes):
        ax.axvspan(t_m[kf], t_m[kl], color="red", alpha=0.10,
                   label="模型欠压故障" if i == 2 else None)

fig.suptitle(f"含 BMS 约束层的单簇(Bank01)模型 vs 2025-07-01 实测数据 "
             f"(N_S={N_S}, N_P={N_P}, Q={Q_CLU:g}Ah)")
fig.tight_layout()
out_png = os.path.join(OUT_DIR, "bms_vs_bank01.png")
fig.savefig(out_png, dpi=130)
print(f"\n对比图已保存: {out_png}")
plt.show()
