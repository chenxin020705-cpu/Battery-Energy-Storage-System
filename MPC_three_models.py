"""
=============================================================================
 多储能联合AGC调频 —— MPC完整实现（三模型整合 / 电流为控制变量）
 对应文档：4.5 策略B：MPC滚动优化策略

 储能类型与模型：
   1. supercap   —— 二阶RC等效电路（EnergyStorage）
   2. li_batt    —— 分数阶二阶RC（FractionalOrderRCStorage，GL离散+短记忆）
   3. aofb_flow  —— 水系有机液流电池（AofBFlowBattery，能斯特+ASR+浓差）

 核心设计：以电流 I 为唯一优化变量
   - 电压 U 由各电池模型精确计算
   - 功率 P = I × U，工作点线性化 P ≈ P₀ + kₚ·(I−I₀) 保持QP凸性
   - SOC 由法拉第定律直接递推
   - 分数阶RC递推在预测时域内展开为符号表达式（历史常数+变量线性组合）

 双模型架构：
   - MPC预测/优化阶段：线性化降阶模型（保持QP，毫秒级求解）
   - MPC执行/状态更新：精确非线性模型（分数阶递推 / 能斯特+ASR+浓差）

 依赖安装：pip install cvxpy numpy
=============================================================================
"""

import numpy as np
try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

FARADAY = 96485.3
GAS_CONSTANT = 8.31
EPS = 1.0e-12


# =============================================================================
#  一、二阶RC储能设备（超级电容）
# =============================================================================
@dataclass
class EnergyStorage:
    """二阶RC等效电路模型储能设备。以电流为控制变量，功率=I×U为派生量。"""
    dev_id: str
    dev_type: str

    q_nom: float
    eta: float
    r0: float
    r1: float
    c1: float
    r2: float
    c2: float
    ocv_coeff: np.ndarray

    soc: float
    soc_ref: float
    soc_min: float
    soc_max: float
    v_min: float
    v_max: float
    i_min: float
    i_max: float
    delta_i_max: float
    voltage: float

    v_rc1_max: float = 0.5
    v_rc2_max: float = 0.3
    pcs_pmin: float = -1e9
    pcs_pmax: float = 1e9
    temp: float = 25.0
    t_min: float = 0.0
    t_max: float = 50.0
    w_soc: float = 10.0
    w_dp: float = 0.1
    w_life: float = 0.001
    w_vrc: float = 0.0
    soh: float = 0.95
    last_i: float = 0.0
    last_p: float = 0.0
    v_rc1: float = 0.0
    v_rc2: float = 0.0

    alpha1: float = field(init=False, default=0.0)
    beta1: float = field(init=False, default=0.0)
    alpha2: float = field(init=False, default=0.0)
    beta2: float = field(init=False, default=0.0)

    def __post_init__(self):
        dt = 1.0
        tau1 = self.r1 * self.c1
        tau2 = self.r2 * self.c2
        self.alpha1 = float(np.exp(-dt / tau1)) if tau1 > 0 else 0.0
        self.beta1 = self.r1 * (1.0 - self.alpha1)
        self.alpha2 = float(np.exp(-dt / tau2)) if tau2 > 0 else 0.0
        self.beta2 = self.r2 * (1.0 - self.alpha2)

    def ocv(self, soc: float) -> float:
        c = self.ocv_coeff
        return float(c[0] + c[1] * soc + c[2] * soc ** 2 + c[3] * soc ** 3)

    def ocv_derivative(self, soc: float) -> float:
        c = self.ocv_coeff
        return float(c[1] + 2.0 * c[2] * soc + 3.0 * c[3] * soc ** 2)

    def terminal_voltage(self, current: float, soc: float, v_rc1: float, v_rc2: float) -> float:
        return self.ocv(soc) - current * self.r0 - v_rc1 - v_rc2

    def power_linearization(self, current: float, soc: float, v_rc1: float, v_rc2: float):
        """P=I×U工作点线性化，返回(P0_W, kp_W/A, U0)"""
        U0 = self.terminal_voltage(current, soc, v_rc1, v_rc2)
        P0 = current * U0
        dU_dI = -self.r0 - self.beta1 - self.beta2
        k_p = U0 + current * dU_dI
        return P0, k_p, U0


# =============================================================================
#  二、分数阶二阶RC储能设备（锂电池）
# =============================================================================
@dataclass
class FractionalOrderRCStorage:
    """
    分数阶二阶RC等效电路模型（Grünwald-Letnikov离散 + 短记忆截断）。

    状态：x = [SOC, V1, V2]
    递推（对应README §4.4）：
      SOC[k] = SOC[k-1] − η·I[k]·dt / (3600·Q)
      V_i[k] = ( I[k]/C_i − dt^(−α_i)·Σ_{l=1..L} w_i[l]·V_i[k−l] )
               / ( dt^(−α_i) + 1/(R_i·C_i) )
      w_i[l] = (−1)^l · C(α_i, l)   （GL权重）
      V_cell = OCV(SOC) − I·R₀ − V1 − V2

    MPC预测中，分数阶递推展开为：历史常数部分（预求和）+ 变量线性组合。
    执行阶段用精确分数阶递推更新状态与历史缓冲。
    """
    dev_id: str

    # 分数阶RC参数（堆级等效）
    r0: float
    r1: float
    c1: float
    alpha1: float
    r2: float
    c2: float
    alpha2: float
    q_pack: float  # 堆容量 Ah
    eta: float
    dt: float
    memory_limit: int  # 短记忆窗口（步）

    # OCV多项式（堆级，代替查表；若有OCV(SOC,T)表可替换为插值+线性化）
    ocv_coeff: np.ndarray

    # 状态
    soc: float
    soc_ref: float
    soc_min: float
    soc_max: float
    v_min: float
    v_max: float
    i_min: float
    i_max: float
    delta_i_max: float
    voltage: float

    # 历史缓冲（最近在前：v1_history[0]=V[k-1], v1_history[1]=V[k-2], ...）
    v1_history: np.ndarray = field(default=None)
    v2_history: np.ndarray = field(default=None)

    # 有默认值的参数
    pcs_pmin: float = -1e9
    pcs_pmax: float = 1e9
    temp: float = 25.0
    t_min: float = 0.0
    t_max: float = 50.0
    soh: float = 0.95
    last_i: float = 0.0
    last_p: float = 0.0
    w_soc: float = 15.0
    w_dp: float = 0.05
    w_life: float = 0.002
    w_vrc: float = 0.0
    dev_type: str = "fractional_rc"

    # 预计算GL权重
    w1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)

    def __post_init__(self):
        # 预计算GL权重 w[l] = (-1)^l * C(alpha, l), l=1..L
        self.w1 = self._gl_weights(self.alpha1, self.memory_limit)
        self.w2 = self._gl_weights(self.alpha2, self.memory_limit)
        # 初始化历史缓冲为零（冷启动；实际运行前应预热填充）
        if self.v1_history is None or len(self.v1_history) < self.memory_limit:
            self.v1_history = np.zeros(self.memory_limit)
            self.v2_history = np.zeros(self.memory_limit)

    @staticmethod
    def _gl_weights(alpha: float, L: int) -> np.ndarray:
        """Grünwald-Letnikov权重 w[l] = (-1)^l · C(α, l), l=1..L。
        递推：w[l] = w[l-1] · (1 − (α+1)/l)"""
        w = np.zeros(L + 1)
        w[0] = 1.0
        for l in range(1, L + 1):
            w[l] = w[l - 1] * (1.0 - (alpha + 1.0) / l)
        return w[1:]  # 返回 w[1..L]

    def ocv(self, soc: float) -> float:
        c = self.ocv_coeff
        return float(c[0] + c[1] * soc + c[2] * soc ** 2 + c[3] * soc ** 3)

    def ocv_derivative(self, soc: float) -> float:
        c = self.ocv_coeff
        return float(c[1] + 2.0 * c[2] * soc + 3.0 * c[3] * soc ** 2)

    def _fractional_rc_next(self, current: float, v_hist: np.ndarray,
                            w: np.ndarray, R: float, C: float, alpha: float) -> float:
        """精确分数阶RC一步递推（数值计算，执行阶段用）。"""
        dt_alpha = self.dt ** alpha
        denom = 1.0 / dt_alpha + 1.0 / (R * C)
        L = min(len(v_hist), self.memory_limit)
        hist_sum = float(np.sum(w[:L] * v_hist[:L]))
        return (current / C - hist_sum / dt_alpha) / denom

    def terminal_voltage(self, current: float, soc: float, v1: float, v2: float) -> float:
        return self.ocv(soc) - current * self.r0 - v1 - v2

    def power_linearization(self, current: float, soc: float, v1: float, v2: float):
        """
        P=I×U工作点线性化。
        分数阶一步瞬态增益：dV_i/dI = 1 / (C_i · (dt^(−α_i) + 1/(R_i·C_i)))
        返回 (P0_W, kp_W/A, U0)
        """
        U0 = self.terminal_voltage(current, soc, v1, v2)
        P0 = current * U0
        dt_a1 = self.dt ** self.alpha1
        dt_a2 = self.dt ** self.alpha2
        dV1_dI = 1.0 / (self.c1 * (1.0 / dt_a1 + 1.0 / (self.r1 * self.c1)))
        dV2_dI = 1.0 / (self.c2 * (1.0 / dt_a2 + 1.0 / (self.r2 * self.c2)))
        dU_dI = -self.r0 - dV1_dI - dV2_dI
        kp = U0 + current * dU_dI
        return P0, kp, U0

    def step(self, current: float) -> float:
        """
        执行阶段：精确分数阶模型单步更新。
        更新 SOC、V1、V2（含历史缓冲滚动）、端电压。返回端电压。
        """
        # SOC递推（法拉第定律，放电I>0→SOC减小）
        q_eff = self.q_pack * self.soh
        self.soc = float(np.clip(
            self.soc - self.eta * current * self.dt / (q_eff * 3600.0),
            self.soc_min, self.soc_max
        ))
        # 分数阶RC递推（用当前历史缓冲）
        v1_new = self._fractional_rc_next(current, self.v1_history, self.w1,
                                          self.r1, self.c1, self.alpha1)
        v2_new = self._fractional_rc_next(current, self.v2_history, self.w2,
                                          self.r2, self.c2, self.alpha2)
        # 历史缓冲左移滚动：新值放最前，最旧值丢弃
        self.v1_history = np.roll(self.v1_history, 1)
        self.v1_history[0] = v1_new
        self.v2_history = np.roll(self.v2_history, 1)
        self.v2_history[0] = v2_new
        # 端电压
        self.voltage = self.terminal_voltage(current, self.soc, v1_new, v2_new)
        return self.voltage

    def precompute_const_history(self, Np: int) -> tuple:
        """
        预计算预测时域内每一步的分数阶历史常数和。
        对于预测步 j，常数部分 = Σ_{l=j+1..L} w[l-1] · v_history[l-j-1]
        （即 V[k+j-l] 落在预测起点之前的那些项）
        返回 (const_hist1[Np], const_hist2[Np])
        """
        L = self.memory_limit
        const1 = np.zeros(Np)
        const2 = np.zeros(Np)
        for j in range(Np):
            # l 从 j+1 到 L：V[k+j-l] 是历史常数，索引 = l-j-1
            for l in range(j + 1, L + 1):
                idx = l - j - 1
                if idx < len(self.v1_history):
                    const1[j] += self.w1[l - 1] * self.v1_history[idx]
                    const2[j] += self.w2[l - 1] * self.v2_history[idx]
        return const1, const2


# =============================================================================
#  三、水系有机液流电池（AofB）
# =============================================================================
@dataclass
class AofBFlowBattery:
    """水系有机液流电池（MV/TEMPTMA）MPC接口。以电流为控制变量。"""
    dev_id: str

    n_cells: int
    formal_potential: float
    temperature: float
    total_negative_active: float
    total_positive_active: float
    electrode_area: float
    flow_cross_section: float
    cell_resistivity_charge: float
    cell_resistivity_discharge: float
    diffusion_electrolyte: np.ndarray

    soc: float
    soc_ref: float
    soc_min: float
    soc_max: float
    v_min: float
    v_max: float
    i_min: float
    i_max: float
    delta_i_max: float
    voltage: float
    flow_rate: float
    flow_min: float
    flow_max: float
    capacity_ah: float

    pcs_pmin: float = -1e9
    pcs_pmax: float = 1e9
    asr_reference_temperature: float = 298.15
    asr_activation_energy: float = 15000.0
    asr_soc_reference: float = 0.5
    asr_soc_coefficient: float = 0.20
    asr_aging_per_cycle: float = 0.001
    electrons: int = 1
    temp_celsius: float = 25.0
    t_min: float = 0.0
    t_max: float = 50.0
    cycle_index: float = 1.0
    soh: float = 0.95
    last_i: float = 0.0
    last_p: float = 0.0
    w_soc: float = 20.0
    w_dp: float = 0.1
    w_life: float = 0.005
    w_flow: float = 0.0
    dev_type: str = "aofb_flow"

    lin_ocv0: float = field(init=False, default=0.0)
    lin_k_soc: float = field(init=False, default=0.0)
    lin_k_i: float = field(init=False, default=0.0)
    lin_i_lim: float = field(init=False, default=1e9)
    lin_u0: float = field(init=False, default=0.0)
    lin_i0: float = field(init=False, default=0.0)
    lin_soc0: float = field(init=False, default=0.0)
    lin_p0: float = field(init=False, default=0.0)
    lin_kp: float = field(init=False, default=0.0)

    def _kelvin(self) -> float:
        return self.temp_celsius + 273.15

    def dynamic_asr(self, process: str, soc: float) -> float:
        reference = self.cell_resistivity_charge if process == "charge" else self.cell_resistivity_discharge
        T = self._kelvin()
        temp_factor = np.exp(
            self.asr_activation_energy / GAS_CONSTANT * (1.0 / T - 1.0 / self.asr_reference_temperature))
        soc_factor = 1.0 + self.asr_soc_coefficient * (float(np.clip(soc, 0, 1)) - self.asr_soc_reference) ** 2
        aging_factor = 1.0 + self.asr_aging_per_cycle * (self.cycle_index - 1.0)
        return float(reference * temp_factor * soc_factor * aging_factor)

    def ocv_single_cell(self, soc: float) -> float:
        soc_c = float(np.clip(soc, EPS, 1.0 - EPS))
        return self.formal_potential + (GAS_CONSTANT * self._kelvin() / (self.electrons * FARADAY)) * np.log(
            soc_c / (1.0 - soc_c))

    def stack_ocv(self, soc: float) -> float:
        return self.n_cells * self.ocv_single_cell(soc)

    def mass_transfer_coeff(self, flow_rate: float) -> np.ndarray:
        velocity = flow_rate / self.flow_cross_section
        km = np.empty(6, dtype=float)
        km[:3] = 1.6e-4 * velocity ** 0.4 * self.diffusion_electrolyte[:3] / 2.4e-10
        km[3:] = 2.6e-4 * velocity ** 0.4 * self.diffusion_electrolyte[3:] / 3.9e-10
        return km

    def limiting_current(self, soc: float, flow_rate: float, process: str) -> float:
        km = self.mass_transfer_coeff(flow_rate)
        if process == "discharge":
            c_neg = self.total_negative_active * soc
            c_pos = self.total_positive_active * soc
            i_lim_neg = self.electrons * FARADAY * km[0] * c_neg * self.electrode_area
            i_lim_pos = self.electrons * FARADAY * km[4] * c_pos * self.electrode_area
        else:
            c_neg = self.total_negative_active * (1.0 - soc)
            c_pos = self.total_positive_active * (1.0 - soc)
            i_lim_neg = self.electrons * FARADAY * km[1] * c_neg * self.electrode_area
            i_lim_pos = self.electrons * FARADAY * km[3] * c_pos * self.electrode_area
        return float(min(i_lim_neg, i_lim_pos))

    def concentration_overpotential(self, current: float, i_lim: float) -> float:
        if i_lim <= 0:
            return 1e3
        ratio = abs(current) / i_lim
        if ratio >= 0.999:
            ratio_clipped = 0.99
            eta_at_099 = GAS_CONSTANT * self._kelvin() / (self.electrons * FARADAY) * np.log(1.0 - ratio_clipped)
            slope_at_099 = GAS_CONSTANT * self._kelvin() / (self.electrons * FARADAY) / (i_lim * (1.0 - ratio_clipped))
            eta = eta_at_099 + slope_at_099 * (ratio - ratio_clipped) * i_lim
            return abs(eta) * 2.0
        eta = GAS_CONSTANT * self._kelvin() / (self.electrons * FARADAY) * np.log(1.0 - ratio)
        return abs(eta) * 2.0

    def stack_voltage(self, current: float, soc: float, process: str, flow_rate: float) -> float:
        i_density = abs(current) / self.electrode_area
        asr = self.dynamic_asr(process, soc)
        i_lim = self.limiting_current(soc, flow_rate, process)
        eta_conc = self.concentration_overpotential(current, i_lim)
        loss_single = i_density * asr + eta_conc
        direction = 1.0 if process == "charge" else -1.0
        return self.n_cells * (self.ocv_single_cell(soc) + direction * loss_single)

    def linearize_at_operating_point(self, i_work: float, dt: float = 1.0):
        soc0 = self.soc
        process = "discharge" if i_work >= 0 else "charge"
        u0 = self.stack_voltage(i_work, soc0, process, self.flow_rate)
        p0 = i_work * u0
        ds = 0.01
        u_p = self.stack_voltage(i_work, soc0 + ds, process, self.flow_rate)
        u_m = self.stack_voltage(i_work, soc0 - ds, process, self.flow_rate)
        k_soc = (u_p - u_m) / (2.0 * ds)
        di = max(abs(i_work) * 0.01, 0.1)
        u_ip = self.stack_voltage(i_work + di, soc0, process, self.flow_rate)
        u_im = self.stack_voltage(i_work - di, soc0, process, self.flow_rate)
        k_i = (u_ip - u_im) / (2.0 * di)
        k_p = u0 + i_work * k_i
        i_lim = self.limiting_current(soc0, self.flow_rate, process)
        self.lin_soc0 = soc0
        self.lin_i0 = i_work
        self.lin_u0 = u0
        self.lin_p0 = p0
        self.lin_ocv0 = self.stack_ocv(soc0)
        self.lin_k_soc = float(k_soc)
        self.lin_k_i = float(k_i)
        self.lin_kp = float(k_p)
        self.lin_i_lim = float(i_lim)

    def soc_step(self, current: float, dt: float) -> float:
        q_eff = self.capacity_ah * self.soh
        dsoc = -current * dt / (q_eff * 3600.0)
        return float(np.clip(self.soc + dsoc, self.soc_min, self.soc_max))


# =============================================================================
#  四、MPC控制器（三模型 / 电流为控制变量）
# =============================================================================
class MPCController:
    """
    多储能联合AGC调频 MPC 控制器。
    唯一优化变量：各储能电流 var_i (n_sto × Np)
    支持三种模型：二阶RC(EnergyStorage)、分数阶RC(FractionalOrderRCStorage)、AofB(AofBFlowBattery)
    """

    def __init__(self, predict_horizon: int, weight_agc: float, dt: float = 1.0):
        self.Np = predict_horizon
        self.w_e = weight_agc
        self.dt = dt
        self.history: List[dict] = []

    @staticmethod
    def _is_aofb(sto) -> bool:
        return isinstance(sto, AofBFlowBattery)

    @staticmethod
    def _is_fractional(sto) -> bool:
        return isinstance(sto, FractionalOrderRCStorage)

    def build_optimization_problem(
            self,
            sto_list: List,
            p_unit: float,
            pred_p_agc: np.ndarray,
            comm_error: bool = False,
            strategy_switch: bool = False,
    ):
        if cp is None:
            raise RuntimeError("旧版MPC需要安装cvxpy。")
        n_sto = len(sto_list)
        Np = self.Np
        var_i = cp.Variable((n_sto, Np))

        state_exprs: List[Dict[str, list]] = []
        power_lin: List[Dict[str, list]] = []

        # ===== 预计算：所有设备的功率线性化参数 =====
        for sto in sto_list:
            if self._is_aofb(sto):
                sto.linearize_at_operating_point(sto.last_i, self.dt)
                power_lin.append({
                    "p0": [sto.lin_p0 / 1000.0] * Np,
                    "kp": [sto.lin_kp / 1000.0] * Np,
                    "i0": [sto.lin_i0] * Np,
                })
            elif self._is_fractional(sto):
                # 分数阶：用当前V1/V2（历史缓冲最前值）计算工作点
                v1_now = sto.v1_history[0] if len(sto.v1_history) > 0 else 0.0
                v2_now = sto.v2_history[0] if len(sto.v2_history) > 0 else 0.0
                P0, kp, U0 = sto.power_linearization(sto.last_i, sto.soc, v1_now, v2_now)
                power_lin.append({
                    "p0": [P0 / 1000.0] * Np,
                    "kp": [kp / 1000.0] * Np,
                    "i0": [sto.last_i] * Np,
                })
            else:
                i_work = sto.last_i
                u_work = sto.terminal_voltage(i_work, sto.soc, sto.v_rc1, sto.v_rc2)
                p_work = i_work * u_work / 1000.0
                dU_dI = -sto.r0 - sto.beta1 - sto.beta2
                kp_work = (u_work + i_work * dU_dI) / 1000.0
                power_lin.append({
                    "p0": [p_work] * Np,
                    "kp": [kp_work] * Np,
                    "i0": [i_work] * Np,
                })

        cost = 0
        constraints = []

        for i, sto in enumerate(sto_list):
            # 温度前置校验
            t_cur = sto.temp if hasattr(sto, 'temp') else sto.temp_celsius
            if not (sto.t_min <= t_cur <= sto.t_max):
                print(f"[警告] {sto.dev_id} 当前温度 {t_cur} 越限")

            if self._is_aofb(sto):
                # ============================================================
                #  AofB 分支
                # ============================================================
                soc0 = sto.lin_soc0
                u0 = sto.lin_u0
                k_soc = sto.lin_k_soc
                k_i = sto.lin_k_i
                i_lim = sto.lin_i_lim
                i0 = sto.lin_i0
                p0 = sto.lin_p0
                kp = sto.lin_kp

                soc_e, u_e, p_e = [], [], []
                q_ah = sto.capacity_ah * sto.soh

                for j in range(Np):
                    i_ij = var_i[i, j]
                    if j == 0:
                        soc_e.append(soc0 - i_ij * self.dt / (q_ah * 3600.0))
                    else:
                        soc_e.append(soc_e[j - 1] - i_ij * self.dt / (q_ah * 3600.0))
                    u_ij = u0 + k_soc * (soc_e[j] - soc0) + k_i * (i_ij - i0)
                    u_e.append(u_ij)
                    p_ij = p0 + kp * (i_ij - i0)
                    p_e.append(p_ij)

                state_exprs.append({"soc": soc_e, "voltage": u_e, "power": p_e,
                                    "current": [var_i[i, j] for j in range(Np)]})

                for j in range(Np):
                    i_ij = var_i[i, j]
                    soc_ij = soc_e[j]
                    u_ij = u_e[j]

                    if i == 0:
                        total_p = cp.sum(
                            [power_lin[k]["p0"][j] + power_lin[k]["kp"][j] * (var_i[k, j] - power_lin[k]["i0"][j]) for k
                             in range(n_sto)])
                        cost += self.w_e * cp.square(p_unit + total_p - pred_p_agc[j])

                    cost += sto.w_soc * cp.square(soc_ij - sto.soc_ref)
                    di = i_ij - sto.last_i if j == 0 else var_i[i, j] - var_i[i, j - 1]
                    cost += sto.w_dp * cp.square(di)
                    cost += sto.w_life * (cp.square(i_ij) + 5.0 * cp.square(soc_ij - sto.soc_ref)) / sto.soh

                    constraints += [sto.i_min <= i_ij, i_ij <= sto.i_max]
                    if j == 0:
                        constraints.append(cp.abs(i_ij - sto.last_i) <= sto.delta_i_max)
                    else:
                        constraints.append(cp.abs(var_i[i, j] - var_i[i, j - 1]) <= sto.delta_i_max)
                    if not isinstance(soc_ij, (int, float, np.floating)):
                        constraints += [sto.soc_min <= soc_ij, soc_ij <= sto.soc_max]
                    if not isinstance(u_ij, (int, float, np.floating)):
                        constraints += [sto.v_min <= u_ij, u_ij <= sto.v_max]
                    constraints.append(cp.abs(i_ij) <= i_lim * 0.95)

            elif self._is_fractional(sto):
                # ============================================================
                #  分数阶RC 分支（核心：历史常数预求和 + 变量线性组合展开）
                # ============================================================
                soc0 = sto.soc
                ocv0 = sto.ocv(soc0)
                k_ocv = sto.ocv_derivative(soc0)
                q_eff = sto.q_pack * sto.soh

                # 分数阶递推系数
                dt_a1 = sto.dt ** sto.alpha1
                denom1 = 1.0 / dt_a1 + 1.0 / (sto.r1 * sto.c1)
                dt_a2 = sto.dt ** sto.alpha2
                denom2 = 1.0 / dt_a2 + 1.0 / (sto.r2 * sto.c2)

                # 预计算每步的历史常数和（预测起点之前的V值，全是常数）
                const_hist1, const_hist2 = sto.precompute_const_history(Np)

                soc_e, v1_e, v2_e, u_e, p_e = [], [], [], [], []

                for j in range(Np):
                    i_ij = var_i[i, j]

                    # --- SOC递推（法拉第定律）---
                    if j == 0:
                        soc_e.append(soc0 - sto.eta * i_ij * self.dt / (q_eff * 3600.0))
                    else:
                        soc_e.append(soc_e[j - 1] - sto.eta * i_ij * self.dt / (q_eff * 3600.0))

                    # --- V1 分数阶递推（符号展开）---
                    # V1[j] = (I[j]/C1 - dt^-a1 * (变量历史和 + 常数历史和)) / denom1
                    # 变量历史和：l=1..j → w1[l-1] * v1_e[j-l]
                    if j > 0:
                        var_hist1 = cp.sum([sto.w1[l - 1] * v1_e[j - l] for l in range(1, j + 1)])
                    else:
                        var_hist1 = 0.0
                    hist1_sum = var_hist1 + const_hist1[j]
                    v1_next = (i_ij / sto.c1 - hist1_sum / dt_a1) / denom1
                    v1_e.append(v1_next)

                    # --- V2 分数阶递推 ---
                    if j > 0:
                        var_hist2 = cp.sum([sto.w2[l - 1] * v2_e[j - l] for l in range(1, j + 1)])
                    else:
                        var_hist2 = 0.0
                    hist2_sum = var_hist2 + const_hist2[j]
                    v2_next = (i_ij / sto.c2 - hist2_sum / dt_a2) / denom2
                    v2_e.append(v2_next)

                    # --- 端电压（OCV线性化 + 欧姆 + 分数阶极化）---
                    u_ij = ocv0 + k_ocv * (soc_e[j] - soc0) - i_ij * sto.r0 - v1_e[j] - v2_e[j]
                    u_e.append(u_ij)

                    # --- 功率线性化 ---
                    p_work = power_lin[i]["p0"][j]
                    kp_work = power_lin[i]["kp"][j]
                    i_work = power_lin[i]["i0"][j]
                    p_ij = p_work + kp_work * (i_ij - i_work)
                    p_e.append(p_ij)

                state_exprs.append({"soc": soc_e, "v1": v1_e, "v2": v2_e,
                                    "voltage": u_e, "power": p_e,
                                    "current": [var_i[i, j] for j in range(Np)]})

                for j in range(Np):
                    i_ij = var_i[i, j]
                    soc_ij = soc_e[j]
                    u_ij = u_e[j]

                    # AGC跟踪（全局项，i=0累加一次）
                    if i == 0:
                        total_p = cp.sum(
                            [power_lin[k]["p0"][j] + power_lin[k]["kp"][j] * (var_i[k, j] - power_lin[k]["i0"][j]) for k
                             in range(n_sto)])
                        cost += self.w_e * cp.square(p_unit + total_p - pred_p_agc[j])

                    # SOC保持
                    cost += sto.w_soc * cp.square(soc_ij - sto.soc_ref)
                    # 电流平滑
                    di = i_ij - sto.last_i if j == 0 else var_i[i, j] - var_i[i, j - 1]
                    cost += sto.w_dp * cp.square(di)
                    # 寿命损耗
                    cost += sto.w_life * (cp.square(i_ij) + 5.0 * cp.square(soc_ij - sto.soc_ref)) / sto.soh
                    # 极化电压软约束（可选）
                    if sto.w_vrc > 0:
                        cost += sto.w_vrc * (cp.square(v1_e[j]) + cp.square(v2_e[j]))

                    # 约束：电流边界
                    constraints += [sto.i_min <= i_ij, i_ij <= sto.i_max]
                    # 电流爬坡
                    if j == 0:
                        constraints.append(cp.abs(i_ij - sto.last_i) <= sto.delta_i_max)
                    else:
                        constraints.append(cp.abs(var_i[i, j] - var_i[i, j - 1]) <= sto.delta_i_max)
                    # SOC窗口
                    if not isinstance(soc_ij, (int, float, np.floating)):
                        constraints += [sto.soc_min <= soc_ij, soc_ij <= sto.soc_max]
                    # 端电压约束
                    if not isinstance(u_ij, (int, float, np.floating)):
                        constraints += [sto.v_min <= u_ij, u_ij <= sto.v_max]

            else:
                # ============================================================
                #  二阶RC 分支（超级电容）
                # ============================================================
                soc0_rc = sto.soc
                ocv0 = sto.ocv(soc0_rc)
                k_ocv = sto.ocv_derivative(soc0_rc)

                soc_e, v1_e, v2_e, u_e, p_e = [], [], [], [], []

                for j in range(Np):
                    i_ij = var_i[i, j]

                    if j == 0:
                        soc_e.append(soc0_rc - sto.eta * i_ij * self.dt / (sto.q_nom * 3600.0))
                        v1_e.append(sto.alpha1 * sto.v_rc1 + sto.beta1 * i_ij)
                        v2_e.append(sto.alpha2 * sto.v_rc2 + sto.beta2 * i_ij)
                    else:
                        soc_e.append(soc_e[j - 1] - sto.eta * i_ij * self.dt / (sto.q_nom * 3600.0))
                        v1_e.append(sto.alpha1 * v1_e[j - 1] + sto.beta1 * i_ij)
                        v2_e.append(sto.alpha2 * v2_e[j - 1] + sto.beta2 * i_ij)

                    u_ij = ocv0 + k_ocv * (soc_e[j] - soc0_rc) - i_ij * sto.r0 - v1_e[j] - v2_e[j]
                    u_e.append(u_ij)

                    p_work = power_lin[i]["p0"][j]
                    kp_work = power_lin[i]["kp"][j]
                    i_work = power_lin[i]["i0"][j]
                    p_ij = p_work + kp_work * (i_ij - i_work)
                    p_e.append(p_ij)

                state_exprs.append({"soc": soc_e, "v_rc1": v1_e, "v_rc2": v2_e,
                                    "voltage": u_e, "power": p_e,
                                    "current": [var_i[i, j] for j in range(Np)]})

                for j in range(Np):
                    i_ij = var_i[i, j]
                    soc_ij = soc_e[j]
                    u_ij = u_e[j]

                    if i == 0:
                        total_p = cp.sum(
                            [power_lin[k]["p0"][j] + power_lin[k]["kp"][j] * (var_i[k, j] - power_lin[k]["i0"][j]) for k
                             in range(n_sto)])
                        cost += self.w_e * cp.square(p_unit + total_p - pred_p_agc[j])

                    cost += sto.w_soc * cp.square(soc_ij - sto.soc_ref)
                    di = i_ij - sto.last_i if j == 0 else var_i[i, j] - var_i[i, j - 1]
                    cost += sto.w_dp * cp.square(di)
                    cost += sto.w_life * (cp.square(i_ij) + 5.0 * cp.square(soc_ij - sto.soc_ref)) / sto.soh
                    if sto.w_vrc > 0:
                        cost += sto.w_vrc * (cp.square(v1_e[j]) + cp.square(v2_e[j]))

                    constraints += [sto.i_min <= i_ij, i_ij <= sto.i_max]
                    if j == 0:
                        constraints.append(cp.abs(i_ij - sto.last_i) <= sto.delta_i_max)
                    else:
                        constraints.append(cp.abs(var_i[i, j] - var_i[i, j - 1]) <= sto.delta_i_max)
                    if not isinstance(soc_ij, (int, float, np.floating)):
                        constraints += [sto.soc_min <= soc_ij, soc_ij <= sto.soc_max]
                    if not isinstance(u_ij, (int, float, np.floating)):
                        constraints += [sto.v_min <= u_ij, u_ij <= sto.v_max]
                    constraints += [cp.abs(v1_e[j]) <= sto.v_rc1_max]
                    constraints += [cp.abs(v2_e[j]) <= sto.v_rc2_max]

        # 策略切换防冲击
        if comm_error or strategy_switch:
            for i, sto in enumerate(sto_list):
                constraints.append(cp.abs(var_i[i, 0] - sto.last_i) <= sto.delta_i_max * 0.2)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        return problem, var_i, state_exprs, power_lin

    def pcs_bms_verify_and_derate(self, sto_list, i_candidate, u_now):
        """PCS/BMS校验：电流→功率(kW)校验边界，越限降额。"""
        i_verified = np.copy(i_candidate)
        for idx, sto in enumerate(sto_list):
            p_now_kw = i_candidate[idx] * u_now[idx] / 1000.0
            if p_now_kw > sto.pcs_pmax:
                i_verified[idx] = sto.pcs_pmax * 1000.0 / u_now[idx] if u_now[idx] > 0 else i_candidate[idx]
            elif p_now_kw < sto.pcs_pmin:
                i_verified[idx] = sto.pcs_pmin * 1000.0 / u_now[idx] if u_now[idx] > 0 else i_candidate[idx]
            i_verified[idx] = np.clip(i_verified[idx], sto.i_min, sto.i_max)
        return i_verified

    def step(self, sto_list, p_unit, pred_p_agc, comm_error=False, strategy_switch=False):
        prob, var_i, state_exprs, power_lin = self.build_optimization_problem(
            sto_list, p_unit, pred_p_agc, comm_error, strategy_switch
        )

        solver_used = None
        # CLARABEL首选（凸QP稳定）→ OSQP → SCS 三级回退
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            solver_used = "CLARABEL"
        except Exception:
            pass
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            try:
                prob.solve(solver=cp.OSQP, verbose=False, max_iter=500000, eps_abs=1e-5, eps_rel=1e-5, polish=True)
                solver_used = "OSQP"
            except Exception:
                pass
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=200000)
                solver_used = "SCS"
            except Exception:
                pass

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"[警告] MPC求解异常 status={prob.status}，保持上一时刻输出")
            current_i = np.array([s.last_i for s in sto_list])
            pred_i_full = np.zeros((len(sto_list), self.Np))
            pred_soc_full = np.zeros((len(sto_list), self.Np))
            trigger_flag = {"optimize_fail": True, "solver_status": prob.status, "solver": solver_used}
        else:
            pred_i_full = var_i.value
            pred_soc_full = np.zeros((len(sto_list), self.Np))
            for i in range(len(sto_list)):
                for j in range(self.Np):
                    expr = state_exprs[i]["soc"][j]
                    pred_soc_full[i, j] = float(expr) if isinstance(expr, (int, float, np.floating)) else (
                        expr.value if expr.value is not None else 0.0)
            candidate_i = pred_i_full[:, 0]
            u_now = np.array([s.voltage for s in sto_list])
            current_i = self.pcs_bms_verify_and_derate(sto_list, candidate_i, u_now)
            trigger_flag = {"optimize_fail": False, "solver_status": prob.status, "solver": solver_used}

        # ===== 更新设备状态（用各模型精确更新）=====
        p_now = np.zeros(len(sto_list))
        for idx, sto in enumerate(sto_list):
            sto.last_i = current_i[idx]
            current = current_i[idx]

            if self._is_aofb(sto):
                sto.soc = sto.soc_step(current, self.dt)
                process = "discharge" if current >= 0 else "charge"
                sto.voltage = sto.stack_voltage(current, sto.soc, process, sto.flow_rate)
            elif self._is_fractional(sto):
                # 分数阶：精确递推更新SOC+V1+V2+历史缓冲+电压
                sto.step(current)
            else:
                # 二阶RC
                sto.soc = float(
                    np.clip(sto.soc - sto.eta * current * self.dt / (sto.q_nom * 3600.0), sto.soc_min, sto.soc_max))
                sto.v_rc1 = sto.alpha1 * sto.v_rc1 + sto.beta1 * current
                sto.v_rc2 = sto.alpha2 * sto.v_rc2 + sto.beta2 * current
                sto.voltage = sto.terminal_voltage(current, sto.soc, sto.v_rc1, sto.v_rc2)

            p_now[idx] = current * sto.voltage / 1000.0
            sto.last_p = p_now[idx]

        self.history.append({
            "p_unit": p_unit, "pred_agc": pred_p_agc.copy(),
            "cmd_i": current_i.copy(), "cmd_p": p_now.copy(),
            "pred_i_seq": pred_i_full.copy(),
            "pred_soc_seq": pred_soc_full.copy(), "flag": trigger_flag,
        })
        return current_i, p_now, pred_i_full, pred_soc_full, trigger_flag


# =============================================================================
#  五、系统集成用四设备功率MPC
# =============================================================================
@dataclass
class PowerMPCConfig:
    """并网点功率MPC配置；功率统一使用W，时间统一使用s。"""

    predict_horizon: int = 12
    tracking_weight: float = 100.0
    soc_weight: float = 4.0
    smooth_weight: float = 1.0
    throughput_weight: float = 0.01
    default_ramp_fraction_per_s: float = 0.25
    solver_order: Sequence[str] = ("CLARABEL", "OSQP", "SCS")

    def validate(self) -> None:
        if self.predict_horizon <= 0:
            raise ValueError("predict_horizon must be positive.")
        for name in (
            "tracking_weight", "soc_weight", "smooth_weight",
            "throughput_weight", "default_ramp_fraction_per_s",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative.")


@dataclass
class PowerMPCResult:
    """四设备功率MPC的一次滚动优化结果。"""

    setpoints_W: Dict[str, float]
    predicted_power_W: Dict[str, List[float]]
    predicted_soc: Dict[str, List[float]]
    target_power_W: List[float]
    residual_W: float
    solver_status: str
    solver: Optional[str]
    used_fallback: bool = False


class PowerMPCController:
    """直接使用EMS的BMS状态快照进行四设备并网功率优化。

    与旧 ``MPCController`` 不同，本控制器不维护或推进任何储能物理状态。
    设备状态只能由PCS执行层推进一次，随后通过 ``synchronize_actual`` 把实际
    功率反馈给MPC，从而避免原先MPC和设备模型各推进一次的问题。
    """

    def __init__(
        self,
        device_names: Sequence[str],
        soc_targets: Optional[Mapping[str, float]] = None,
        rated_power_W: Optional[Mapping[str, float]] = None,
        config: Optional[PowerMPCConfig] = None,
    ) -> None:
        self.device_names = tuple(device_names)
        if not self.device_names or len(set(self.device_names)) != len(self.device_names):
            raise ValueError("device_names must be non-empty and unique.")
        self.config = config if config is not None else PowerMPCConfig()
        self.config.validate()
        self.soc_targets = {
            name: float((soc_targets or {}).get(name, 0.5))
            for name in self.device_names
        }
        self.rated_power_W = {
            name: float((rated_power_W or {}).get(name, 0.0))
            for name in self.device_names
        }
        self.last_actual_power_W = {name: 0.0 for name in self.device_names}

    def synchronize_actual(self, actual_power_W: Mapping[str, float]) -> None:
        """用PCS实际功率同步滚动优化的上一控制量。"""
        for name in self.device_names:
            if name in actual_power_W:
                value = float(actual_power_W[name])
                if not np.isfinite(value):
                    raise ValueError(f"Actual power for {name} must be finite.")
                self.last_actual_power_W[name] = value

    def _forecast(self, target_power_W: float, forecast_power_W: Optional[Sequence[float]]) -> np.ndarray:
        if forecast_power_W is None:
            values = np.full(self.config.predict_horizon, float(target_power_W))
        else:
            values = np.asarray(forecast_power_W, dtype=float).reshape(-1)
            if values.size == 0:
                raise ValueError("forecast_power_W cannot be empty.")
            if values.size < self.config.predict_horizon:
                values = np.pad(values, (0, self.config.predict_horizon-values.size), mode="edge")
            values = values[:self.config.predict_horizon]
        if not np.all(np.isfinite(values)):
            raise ValueError("Power forecast must contain finite values.")
        return values

    def _fallback(self, target_W: float, states: Mapping[str, Any], dt_s: float) -> Dict[str, float]:
        """求解器不可用时使用受边界和爬坡约束的确定性分配。"""
        result = {name: 0.0 for name in self.device_names}
        remaining = float(target_W)
        active = list(self.device_names)
        for _ in range(len(active) + 1):
            if not active or abs(remaining) <= 1.0e-9:
                break
            weights = []
            for name in active:
                state = states[name]
                target = self.soc_targets[name]
                directional = max(0.05, state.soc-target+0.5) if remaining > 0 else max(0.05, target-state.soc+0.5)
                weights.append(directional)
            total_weight = sum(weights)
            before = remaining
            next_active = []
            for name, weight in zip(active, weights):
                state = states[name]
                rated = self.rated_power_W[name] or max(
                    state.max_discharge_power_W, state.max_charge_power_W, 1.0
                )
                ramp = self.config.default_ramp_fraction_per_s * rated * dt_s
                lower = max(-state.max_charge_power_W, self.last_actual_power_W[name]-ramp)
                upper = min(state.max_discharge_power_W, self.last_actual_power_W[name]+ramp)
                desired = before * weight / total_weight
                add = float(np.clip(desired, lower-result[name], upper-result[name]))
                result[name] += add
                remaining -= add
                if (remaining > 0 and result[name] < upper-1.0e-9) or (
                    remaining < 0 and result[name] > lower+1.0e-9
                ):
                    next_active.append(name)
            if abs(before-remaining) <= 1.0e-9:
                break
            active = next_active
        return result

    def calculate(
        self,
        target_power_W: float,
        bms_states: Mapping[str, Any],
        dt_s: float,
        forecast_power_W: Optional[Sequence[float]] = None,
    ) -> PowerMPCResult:
        """计算四设备功率指令，但不推进任何设备状态。"""
        if not np.isfinite(target_power_W) or not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("target_power_W must be finite and dt_s must be positive.")
        missing = set(self.device_names) - set(bms_states)
        if missing:
            raise ValueError(f"Missing BMS states: {sorted(missing)}")
        forecast = self._forecast(target_power_W, forecast_power_W)
        n, horizon = len(self.device_names), self.config.predict_horizon
        if cp is None:
            setpoints = self._fallback(float(forecast[0]), bms_states, dt_s)
            return PowerMPCResult(
                setpoints_W=setpoints,
                predicted_power_W={name: [setpoints[name]]*horizon for name in self.device_names},
                predicted_soc={name: [float(bms_states[name].soc)]*horizon for name in self.device_names},
                target_power_W=forecast.tolist(),
                residual_W=float(forecast[0]-sum(setpoints.values())),
                solver_status="cvxpy_unavailable",
                solver=None,
                used_fallback=True,
            )
        p_dis = cp.Variable((n, horizon), nonneg=True)
        p_chg = cp.Variable((n, horizon), nonneg=True)
        net = p_dis - p_chg
        constraints = []
        cost = 0
        soc_expressions: Dict[str, List[Any]] = {}

        # 以kW缩放功率目标，避免W量级平方造成病态数值。
        for k in range(horizon):
            cost += self.config.tracking_weight * cp.square(cp.sum(net[:, k])/1000.0 - forecast[k]/1000.0)

        for i, name in enumerate(self.device_names):
            state = bms_states[name]
            available = bool(state.available) and not bool(state.fault)
            dis_limit = float(state.max_discharge_power_W) if available else 0.0
            chg_limit = float(state.max_charge_power_W) if available else 0.0
            constraints += [p_dis[i, :] <= dis_limit, p_chg[i, :] <= chg_limit]
            rated = self.rated_power_W[name] or max(dis_limit, chg_limit, 1.0)
            ramp_W = self.config.default_ramp_fraction_per_s * rated * dt_s
            previous = self.last_actual_power_W[name]
            constraints += [cp.abs(net[i, 0]-previous) <= ramp_W]
            if horizon > 1:
                constraints += [cp.abs(net[i, 1:]-net[i, :-1]) <= ramp_W]
            changes = [net[i, 0]-previous]
            if horizon > 1:
                changes.append(net[i, 1:]-net[i, :-1])
            cost += self.config.smooth_weight * cp.sum_squares(
                cp.hstack(changes)/1000.0
            )
            cost += self.config.throughput_weight * cp.sum(
                (p_dis[i, :]+p_chg[i, :])/1000.0
            )

            energy_Wh = getattr(state, "energy_capacity_Wh", None)
            soc_values: List[Any] = []
            if energy_Wh is not None and float(energy_Wh) > 0.0:
                soc = float(getattr(state, "soe", None) if getattr(state, "soe", None) is not None else state.soc)
                eta_dis = max(float(getattr(state, "discharge_efficiency", 1.0)), 1.0e-6)
                eta_chg = max(float(getattr(state, "charge_efficiency", 1.0)), 1.0e-6)
                lower = float(state.soc_min if state.soc_min is not None else 0.0)
                upper = float(state.soc_max if state.soc_max is not None else 1.0)
                for k in range(horizon):
                    soc = soc - (
                        p_dis[i, k]/eta_dis - p_chg[i, k]*eta_chg
                    ) * dt_s / (float(energy_Wh)*3600.0)
                    constraints += [soc >= lower, soc <= upper]
                    cost += self.config.soc_weight * cp.square(soc-self.soc_targets[name])
                    soc_values.append(soc)
            else:
                soc_values = [float(state.soc)] * horizon
            soc_expressions[name] = soc_values

        problem = cp.Problem(cp.Minimize(cost), constraints)
        solver_used: Optional[str] = None
        for solver in self.config.solver_order:
            try:
                kwargs = {"verbose": False}
                if solver == "OSQP":
                    kwargs.update({"max_iter": 200000, "eps_abs": 1.0e-5, "eps_rel": 1.0e-5})
                problem.solve(solver=solver, **kwargs)
                solver_used = solver
            except Exception:
                continue
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break

        solved = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and net.value is not None
        if solved:
            matrix = np.asarray(net.value, dtype=float)
            setpoints = {name: float(matrix[i, 0]) for i, name in enumerate(self.device_names)}
            predicted = {name: matrix[i, :].tolist() for i, name in enumerate(self.device_names)}
            predicted_soc = {}
            for name, expressions in soc_expressions.items():
                predicted_soc[name] = [
                    float(value) if isinstance(value, (int, float, np.floating))
                    else float(value.value)
                    for value in expressions
                ]
            status = str(problem.status)
            used_fallback = False
        else:
            setpoints = self._fallback(float(forecast[0]), bms_states, dt_s)
            predicted = {name: [setpoints[name]]*horizon for name in self.device_names}
            predicted_soc = {name: [float(bms_states[name].soc)]*horizon for name in self.device_names}
            status = str(problem.status or "solver_error")
            used_fallback = True

        return PowerMPCResult(
            setpoints_W=setpoints,
            predicted_power_W=predicted,
            predicted_soc=predicted_soc,
            target_power_W=forecast.tolist(),
            residual_W=float(forecast[0]-sum(setpoints.values())),
            solver_status=status,
            solver=solver_used,
            used_fallback=used_fallback,
        )


# =============================================================================
#  五、演示程序
# =============================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    """演示2：三模型MPC滚动闭环 —— 超级电容(二阶RC)+锂电池(分数阶)+AofB"""
    print("=" * 70)
    print(" 演示：MPC滚动闭环（三模型 / 电流控制）—— 超容+分数阶锂电+AofB 联合AGC调频")
    print("=" * 70)

    sto_devices: List = [
        # ---- 超级电容（二阶RC）：高功率、小容量 ----
        EnergyStorage(
            dev_id="supercap_01", dev_type="supercap",
            q_nom=200.0, eta=0.99,
            r0=0.005, r1=0.01, c1=800.0, r2=0.02, c2=3000.0,
            ocv_coeff=np.array([320.0, 160.0, 0.0, 0.0]),
            soc=0.5, soc_ref=0.5, soc_min=0.1, soc_max=0.9,
            v_min=336.0, v_max=464.0,
            i_min=-1300, i_max=1300, delta_i_max=800,
            voltage=400.0,
            v_rc1_max=15.0, v_rc2_max=30.0,
            pcs_pmin=-500, pcs_pmax=500,
            w_soc=5, w_dp=2, w_life=0.5, w_vrc=0.5,
        ),
        # ---- 锂电池（分数阶RC）：中功率、中容量 ----
        FractionalOrderRCStorage(
            dev_id="li_frac_01",
            r0=0.015, r1=0.025, c1=4000.0, alpha1=0.9,
            r2=0.06, c2=25000.0, alpha2=0.7,
            q_pack=500.0, eta=0.98, dt=1.0, memory_limit=200,
            ocv_coeff=np.array([600.0, 240.0, -50.0, 30.0]),
            soc=0.5, soc_ref=0.5, soc_min=0.15, soc_max=0.9,
            v_min=620.0, v_max=780.0,
            i_min=-1100, i_max=1100, delta_i_max=300,
            voltage=720.0,
            pcs_pmin=-800, pcs_pmax=800,
            w_soc=15, w_dp=5, w_life=0.5, w_vrc=0.3,
        ),
        # ---- 水系液流电池（AofB）：低功率、大容量 ----
        AofBFlowBattery(
            dev_id="aofb_flow_01",
            n_cells=40, formal_potential=1.2, temperature=298.15,
            total_negative_active=800.0, total_positive_active=800.0,
            electrode_area=0.5, flow_cross_section=0.01,
            cell_resistivity_charge=0.0003, cell_resistivity_discharge=0.0003,
            diffusion_electrolyte=np.array([5e-10] * 6),
            soc=0.5, soc_ref=0.5, soc_min=0.05, soc_max=0.95,
            v_min=35.0, v_max=100.0,
            i_min=-1200, i_max=1200, delta_i_max=200,
            voltage=48.0,
            flow_rate=5e-4, flow_min=1e-5, flow_max=5e-3,
            capacity_ah=800.0,
            pcs_pmin=-100, pcs_pmax=100,
            temp_celsius=25.0, soh=1.0,
            w_soc=20, w_dp=1, w_life=0.5,
        ),
    ]

    mpc = MPCController(predict_horizon=12, weight_agc=30, dt=1.0)
    sim_steps = 30
    p_unit_static = 2000.0
    np.random.seed(42)

    all_agc, all_total_p = [], []
    all_soc = {dev.dev_id: [] for dev in sto_devices}
    all_voltage = {dev.dev_id: [] for dev in sto_devices}
    all_control_i = {dev.dev_id: [] for dev in sto_devices}
    all_each_p = {dev.dev_id: [] for dev in sto_devices}

    model_tag = {}
    for i, dev in enumerate(sto_devices):
        if isinstance(dev, FractionalOrderRCStorage):
            model_tag[i] = "FracRC"
        elif isinstance(dev, AofBFlowBattery):
            model_tag[i] = "AofB"
        else:
            model_tag[i] = "2ndRC"

    print(f"  预测时域 Np={mpc.Np}, 控制周期={mpc.dt}s, 机组出力={p_unit_static}kW")
    print(f"  设备：0=超级电容(二阶RC), 1=锂电池(分数阶α1=0.9/α2=0.7), 2=液流电池(AofB)")
    print(f"  控制变量：电流 I(A)；功率 P=I×U 为派生量")
    print("-" * 70)

    for t in range(sim_steps):
        agc_now = 2200 + 400 * np.sin(0.2 * t) + np.random.randn() * 50
        pred_agc_seq = np.full(mpc.Np, agc_now) + np.linspace(0, 100, mpc.Np)
        switch_flag = (t == 10)

        cmd_i, cmd_p, _, _, flags = mpc.step(
            sto_devices, p_unit=p_unit_static, pred_p_agc=pred_agc_seq,
            strategy_switch=switch_flag,
        )

        total_p_out = np.sum(cmd_p)
        all_agc.append(agc_now)
        all_total_p.append(total_p_out)
        for dev in sto_devices:
            all_soc[dev.dev_id].append(dev.soc * 100)
            all_voltage[dev.dev_id].append(dev.voltage)

        switch_mark = " [策略切换]" if switch_flag else ""
        print(
            f"  t={t:2d}{switch_mark} | AGC={agc_now:7.1f} "
            f"储能总出力={total_p_out:7.2f}kW 合成={p_unit_static + total_p_out:7.1f}"
        )
        for i, dev in enumerate(sto_devices):
            extra = ""
            if isinstance(dev, AofBFlowBattery):
                extra = f" I_lim={dev.lin_i_lim:.0f}A"
            elif isinstance(dev, FractionalOrderRCStorage):
                extra = f" V1={dev.v1_history[0]:.2f}V V2={dev.v2_history[0]:.2f}V"
            print(
                f"      {dev.dev_id:14s}[{model_tag[i]}] I={cmd_i[i]:7.1f}A "
                f"P={cmd_p[i]:7.2f}kW SOC={dev.soc:.4f} U={dev.voltage:.2f}V{extra}"
            )
            all_control_i[dev.dev_id].append(cmd_i[i])
            all_each_p[dev.dev_id].append(cmd_p[i])

    print("-" * 70)
    all_agc = np.array(all_agc)
    all_total_p = np.array(all_total_p)
    rmse = np.sqrt(np.mean((all_agc - (p_unit_static + all_total_p)) ** 2))
    print(f"  调频跟踪 RMSE = {rmse:.2f} kW")
    for dev_id in all_soc:
        print(
            f"  {dev_id:16s} SOC[{min(all_soc[dev_id]):.3f},{max(all_soc[dev_id]):.3f}] "
            f"U[{min(all_voltage[dev_id]):.2f},{max(all_voltage[dev_id]):.2f}]V"
        )

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 跟踪效果图
    plt.figure(figsize=(10, 6))
    plt.plot(all_agc, 'r--', label="AGC指令")
    plt.plot(p_unit_static + all_total_p, 'b-', label="合成出力")
    plt.legend()
    plt.xlabel("时间 (s)")
    plt.ylabel("功率 (kW)")
    plt.title("调频跟踪")
    plt.show()
    # 各储能设备指标
    model_name = ['超级电容', '锂电池', '液流电池']
    colors = ['r', 'g', 'b']
    fig_dics = [all_control_i, all_voltage, all_each_p, all_soc]
    fig_titles = ['电流(A)', '电压(V)', '功率(kW)', 'SOC(%)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for j in range(4):
        for i, dev in enumerate(sto_devices):
            axes[j // 2, j % 2].plot(fig_dics[j][dev.dev_id], color=colors[i], label=f"{model_name[i]}")

            axes[j // 2, j % 2].legend()
            axes[j // 2, j % 2].set_xlabel("时间 (s)")
            axes[j // 2, j % 2].set_ylabel(fig_titles[j])

    plt.show()


