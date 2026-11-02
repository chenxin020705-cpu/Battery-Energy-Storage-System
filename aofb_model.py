"""MV/TEMPTMA 水系有机液流电池堆模型。

模型逐项对应 Guan 等论文第二章式 (1)–(53)。所有输入均采用 SI 单位；
第二章没有给出的实验参数必须由使用者显式提供，代码不会偷偷引入其他章节数据。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
except ImportError:  # 保留没有安装SciPy时的NumPy兼容路径。
    csr_matrix = None
    spsolve = None


FARADAY = 96485.3  # 法拉第常数，C mol^-1
GAS_CONSTANT = 8.31  # 摩尔气体常数，J mol^-1 K^-1
EPS = 1.0e-12


class Process(str, Enum):
    """电池运行过程。"""

    CHARGE = "charge"
    DISCHARGE = "discharge"


# 通用的六状态顺序：两侧有效氧化还原态，以及跨膜后位于错误一侧的物种。
NEG_REDUCED, NEG_OXIDIZED, POS_CROSSOVER_IN_NEG = 0, 1, 2
POS_REDUCED, POS_OXIDIZED, NEG_CROSSOVER_IN_POS = 3, 4, 5
STATE_NAMES = (
    "负极还原态",
    "负极氧化态",
    "正极物种_负极侧",
    "正极还原态",
    "正极氧化态",
    "负极物种_正极侧",
)

# 旧版 MV/TEMPTMA 名称作为兼容别名保留；新代码应优先使用上面的通用名称。
MV1_NEG = NEG_REDUCED
MV2_NEG = NEG_OXIDIZED
T1_NEG = POS_CROSSOVER_IN_NEG
T1_POS = POS_REDUCED
T2_POS = POS_OXIDIZED
MV2_POS = NEG_CROSSOVER_IN_POS


@dataclass(frozen=True)
class ModelParameters:
    """质量守恒、传质和电压方程所需参数（全部为 SI 单位）。"""

    n_cells: int  # 电池堆中串联单电池的数量，无量纲。
    tank_volume_neg: float  # 负极电解液储液罐体积，单位为 m³。
    tank_volume_pos: float  # 正极电解液储液罐体积，单位为 m³。
    cell_volume: float  # 单个完整电池的电解液体积，单位为 m³；每个半电池取其一半。
    membrane_area: float  # 单个电池中用于离子扩散的有效膜面积，单位为 m²。
    membrane_thickness: float  # 离子跨膜扩散距离，即膜厚度，单位为 m。
    electrode_area: float  # 单个电极的有效反应面积，用于计算电流密度，单位为 m²。
    flow_cross_section: float  # 垂直于电解液流向的半电池横截面积，单位为 m²。
    total_negative_active: float  # 负极有效氧化还原物种总浓度，单位为 mol/m³。
    total_positive_active: float  # 正极有效氧化还原物种总浓度，单位为 mol/m³。
    formal_potential: float  # MV/TEMPTMA 电池的形式电势 E0，单位为 V。
    temperature: float  # 电池运行的绝对温度，单位为 K。
    cell_resistivity_charge: float  # 参考温度/SOC/初始循环下的充电ASR，单位为 Ω·m²。
    cell_resistivity_discharge: float  # 参考温度/SOC/初始循环下的放电ASR，单位为 Ω·m²。
    diffusion_electrolyte: NDArray[np.float64]  # 六种状态物种在电解液中的扩散系数，单位为 m²/s。
    diffusion_membrane: NDArray[np.float64]  # 六种状态物种在膜内的扩散系数，单位为 m²/s。
    asr_reference_temperature: float = 298.15  # 参考ASR对应的绝对温度，单位为 K。
    asr_activation_energy: float = 15000.0  # ASR温度修正的表观活化能，单位为 J/mol。
    asr_soc_reference: float = 0.5  # ASR的参考SOC。
    asr_soc_coefficient: float = 0.20  # 弱SOC二次修正系数，无量纲。
    asr_aging_per_cycle: float = 0.001  # 每完成一个循环的ASR相对增长率，默认0.1%/循环。
    degradation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(6))  # 六种物种的分数降解率，单位为 s⁻¹。
    electrons: int = 1  # 每次氧化还原反应转移的电子数 z；MV/TEMPTMA 体系取 1。
    activation_overpotential_neg: float = 0.0  # 负极活化过电位，单位为 V；默认按论文假设忽略。
    activation_overpotential_pos: float = 0.0  # 正极活化过电位，单位为 V；默认按论文假设忽略。

    # 作用：提供旧版参数名兼容入口，返回负极活性物质总浓度。
    @property
    def total_mv(self) -> float:
        """旧版兼容属性；等同于 ``total_negative_active``。"""
        return self.total_negative_active

    # 作用：提供旧版参数名兼容入口，返回正极活性物质总浓度。
    @property
    def total_temptma(self) -> float:
        """旧版兼容属性；等同于 ``total_positive_active``。"""
        return self.total_positive_active

    # 作用：在数据类创建后校验参数形状、范围和物理合理性。
    def __post_init__(self) -> None:
        if self.n_cells < 1:
            raise ValueError("n_cells 必须至少为 1。")
        positive = (
            self.tank_volume_neg,
            self.tank_volume_pos,
            self.cell_volume,
            self.membrane_area,
            self.membrane_thickness,
            self.electrode_area,
            self.flow_cross_section,
            self.total_negative_active,
            self.total_positive_active,
            self.temperature,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("体积、面积、厚度、浓度和温度必须为正数。")
        if self.asr_reference_temperature <= 0.0 or self.asr_activation_energy < 0.0:
            raise ValueError("ASR参考温度必须为正，活化能不允许为负。")
        if not 0.0 <= self.asr_soc_reference <= 1.0:
            raise ValueError("asr_soc_reference 必须位于 [0, 1]。")
        if self.asr_soc_coefficient < 0.0 or self.asr_aging_per_cycle < 0.0:
            raise ValueError("ASR的SOC修正系数和老化增长率不允许为负。")
        for name in ("diffusion_electrolyte", "diffusion_membrane", "degradation"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (6,):
                raise ValueError(f"{name} 必须按 STATE_NAMES 顺序提供 6 个值。")
            if np.any(value < 0.0):
                raise ValueError(f"{name} 不允许出现负值。")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CircuitParameters:
    """式 (34)–(47) 中的正/负电解液流道和歧管电阻（Ω）。"""

    channel_pos: float
    channel_neg: float
    manifold_pos: float
    manifold_neg: float

    # 作用：在数据类创建后校验参数形状、范围和物理合理性。
    def __post_init__(self) -> None:
        if min(self.channel_pos, self.channel_neg, self.manifold_pos, self.manifold_neg) <= 0:
            raise ValueError("流道和歧管电阻必须为正数。")


@dataclass
class CircuitSolution:
    """式 (35)–(47) 的电流解。"""

    cell: NDArray[np.float64]
    channel_pos: NDArray[np.float64]
    channel_neg: NDArray[np.float64]
    manifold_pos: NDArray[np.float64]
    manifold_neg: NDArray[np.float64]


@dataclass
class SimulationResult:
    """一次恒流充电或放电的求解结果。"""

    time: NDArray[np.float64]
    state: NDArray[np.float64]
    process: Process
    flow_rate: float
    terminated_by_cutoff: bool
    message: str


@dataclass(frozen=True)
class PerformanceReference:
    """SOE、SOH 和功率保持率的健康基准。"""

    rated_capacity_ah: float  # 新电池的额定容量，单位为 Ah。
    rated_energy_wh: float  # 新电池的额定可用能量，单位为 Wh。
    rated_power_w: float  # 新电池在相同工况下的参考功率，单位为 W。
    rated_resistance_ohm: Optional[float] = None  # 新电池等效内阻，单位为 Ω；可不提供。

    # 作用：在数据类创建后校验参数形状、范围和物理合理性。
    def __post_init__(self) -> None:
        if min(self.rated_capacity_ah, self.rated_energy_wh, self.rated_power_w) <= 0.0:
            raise ValueError("额定容量、额定能量和额定功率必须为正数。")
        if self.rated_resistance_ohm is not None and self.rated_resistance_ohm <= 0.0:
            raise ValueError("额定内阻必须为正数。")


@dataclass
class PerformanceSeries:
    """一次充/放电过程中用户关心的全部输出时间序列。"""

    time: NDArray[np.float64]  # 时间，s。
    cell_voltage: NDArray[np.float64]  # 各单电池负载电压，形状为 时间点×电池数，V。
    stack_voltage: NDArray[np.float64]  # 电池堆端电压，V。
    stack_ocv: NDArray[np.float64]  # 电池堆开路电压，V。
    soc_neg: NDArray[np.float64]  # 各单电池负极 SOC。
    soc_pos: NDArray[np.float64]  # 各单电池正极 SOC。
    soc: NDArray[np.float64]  # 按论文 2.11 节定义的电池堆过程 SOC。
    remaining_capacity_ah: NDArray[np.float64]  # 当前状态可继续放出的电量，Ah。
    remaining_energy_wh: NDArray[np.float64]  # 当前状态估算的可放电能量，Wh。
    soe: NDArray[np.float64]  # State of Energy，当前能量/额定能量。
    soh: NDArray[np.float64]  # 容量型 State of Health，理论容量/额定容量。
    capacity_fade: NDArray[np.float64]  # 容量衰减比例，1-SOH。
    power_w: NDArray[np.float64]  # 当前充电输入功率或放电输出功率，W。
    power_retention: NDArray[np.float64]  # 相对参考功率的保持率。
    power_fade: NDArray[np.float64]  # 相对参考功率的衰减比例。
    equivalent_resistance_ohm: NDArray[np.float64]  # 由 OCV 与负载电压差得到的等效内阻，Ω。
    resistance_soh: NDArray[np.float64]  # 内阻型 SOH；未提供额定内阻时为 NaN。


@dataclass
class CycleMetrics:
    """一个完整充放电循环的汇总指标。"""

    cycle: int
    charge_capacity_ah: float
    discharge_capacity_ah: float
    charge_energy_wh: float
    discharge_energy_wh: float
    coulombic_efficiency: float
    energy_efficiency: float
    voltage_efficiency: float
    peak_discharge_power_w: float
    average_discharge_power_w: float
    mean_discharge_resistance_ohm: float
    capacity_retention: float
    capacity_fade: float
    power_retention: float
    power_fade: float
    soh: float


@dataclass
class CycleResult:
    """单次循环的原始积分结果、时间序列输出和汇总指标。"""

    charge: SimulationResult
    discharge: SimulationResult
    charge_performance: PerformanceSeries
    discharge_performance: PerformanceSeries
    metrics: CycleMetrics


@dataclass
class MultiCycleResult:
    """多循环仿真结果。"""

    cycles: list[CycleResult]
    reference: PerformanceReference


# 作用：根据电流、反应物浓度和电子数计算半电池化学计量流量。
def stoichiometric_flow_rate(current: float, reactant_concentration: float, electrons: int = 1) -> float:
    """式 (3)：单个半电池的化学计量流量。"""
    if reactant_concentration <= 0.0:
        raise ValueError("反应物浓度必须为正数。")
    return abs(current) / (electrons * FARADAY * reactant_concentration)


# 作用：综合充放电及SOC上下限，计算乘以流量因子后的公共恒定流量。
def applied_constant_flow_rate(
    charge_current: float,
    discharge_current: float,
    total_temptma: float,
    total_mv: float,
    soc_lower: float,
    soc_upper: float,
    flow_factor: float,
    electrons: int = 1,
) -> float:
    """式 (4)–(5)：在给定 SOC 区间内取四种工况所需流量的最大值。"""
    if not 0.0 < soc_lower < soc_upper < 1.0:
        raise ValueError("应满足 0 < soc_lower < soc_upper < 1。")
    if flow_factor <= 0.0:
        raise ValueError("流量因子 FF 必须为正数。")
    candidates = (
        abs(charge_current) / (electrons * FARADAY * total_temptma * (1.0 - soc_upper)),
        abs(charge_current) / (electrons * FARADAY * total_mv * (1.0 - soc_upper)),
        abs(discharge_current) / (electrons * FARADAY * total_temptma * soc_lower),
        abs(discharge_current) / (electrons * FARADAY * total_mv * soc_lower),
    )
    return flow_factor * max(candidates)


# 作用：以通用正负极命名调用恒定流量计算，避免绑定具体电解液名称。
def required_constant_flow_rate(
    charge_current: float,
    discharge_current: float,
    total_positive_active: float,
    total_negative_active: float,
    soc_lower: float,
    soc_upper: float,
    flow_factor: float,
    electrons: int = 1,
) -> float:
    """按正、负极活性物质浓度计算满足 SOC 窗口的公共流量。

    这是 ``applied_constant_flow_rate`` 的体系无关接口；旧函数继续保留，
    以兼容已有的 MV/TEMPTMA 实验代码。
    """
    return applied_constant_flow_rate(
        charge_current,
        discharge_current,
        total_positive_active,
        total_negative_active,
        soc_lower,
        soc_upper,
        flow_factor,
        electrons,
    )


# 作用：由给定初始SOC生成储液罐及全部单体六种物种的初始浓度状态向量。
def initial_state(params: ModelParameters, soc0: float) -> NDArray[np.float64]:
    """式 (48)：由初始 SOC 构造平衡的储液罐和各单电池浓度。"""
    if not 0.0 <= soc0 <= 1.0:
        raise ValueError("soc0 必须位于 [0, 1]。")
    base = np.array(
        [
            params.total_negative_active * soc0,
            params.total_negative_active * (1.0 - soc0),
            0.0,  # 初始负极侧没有跨膜 TEMPTMA+
            params.total_positive_active * (1.0 - soc0),
            params.total_positive_active * soc0,
            0.0,  # 初始正极侧没有跨膜 MV2+
        ],
        dtype=float,
    )
    # 状态布局：[负罐3种，正罐3种，cell(1)六种，...，cell(n)六种]
    return np.concatenate((base[:3], base[3:], np.tile(base, params.n_cells)))


# 作用：把一维状态向量拆分为负极储液罐、正极储液罐和各单体浓度矩阵。
def unpack_state(y: NDArray[np.float64], n_cells: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """把一维积分状态拆成负罐、正罐和 n×6 单电池浓度。"""
    expected = 6 + 6 * n_cells
    if np.asarray(y).shape != (expected,):
        raise ValueError(f"状态长度应为 {expected}。")
    return y[:3], y[3:6], y[6:].reshape(n_cells, 6)


# 作用：由流量、流道截面积和电解液扩散系数计算六种物种的传质系数。
def mass_transfer_coefficients(params: ModelParameters, flow_rate: float) -> NDArray[np.float64]:
    """式 (22)、(24)：由电解液扩散系数估计局部传质系数。"""
    if flow_rate <= 0.0:
        raise ValueError("flow_rate 必须为正数。")
    velocity = flow_rate / params.flow_cross_section
    km = np.empty(6, dtype=float)
    # 式 (22)：负极侧三类有机离子。
    km[:3] = 1.6e-4 * velocity**0.4 * params.diffusion_electrolyte[:3] / 2.4e-10
    # 式 (24)：正极侧三类有机离子。
    km[3:] = 2.6e-4 * velocity**0.4 * params.diffusion_electrolyte[3:] / 3.9e-10
    return km


# 作用：由反应物浓度和传质系数计算局部极限电流密度。
def limiting_current_density(concentration: float, mass_transfer_coefficient: float, electrons: int = 1) -> float:
    """式 (25)：极限电流密度。"""
    return electrons * FARADAY * mass_transfer_coefficient * max(concentration, 0.0)


# 作用：根据工作电流密度与极限电流密度计算浓差过电位。
def concentration_overpotential(
    current_density: float,
    limiting_density: float,
    temperature: float,
    electrons: int = 1,
) -> float:
    """式 (26)：浓差过电位；返回带符号的 Nernst 对数值。"""
    if limiting_density <= 0.0 or current_density >= limiting_density:
        raise ValueError("电流密度必须严格小于正的极限电流密度。")
    return GAS_CONSTANT * temperature / (electrons * FARADAY) * np.log(1.0 - current_density / limiting_density)


# 作用：用单节电池六种物种浓度和Nernst关系计算单体开路电压。
def open_circuit_voltage(cell_concentration: NDArray[np.float64], params: ModelParameters) -> float:
    """式 (31)：单电池开路电压。"""
    c = np.maximum(np.asarray(cell_concentration, dtype=float), EPS)
    quotient = c[MV1_NEG] * c[T2_POS] / (c[MV2_NEG] * c[T1_POS])
    return params.formal_potential + GAS_CONSTANT * params.temperature / (params.electrons * FARADAY) * np.log(quotient)


# 作用：向量化计算电堆中每一节单体的开路电压。
def open_circuit_voltages(
    cell_concentrations: NDArray[np.float64], params: ModelParameters
) -> NDArray[np.float64]:
    """批量计算各节单电池OCV；每一行仍代表一节独立单电池。"""
    c = np.maximum(np.asarray(cell_concentrations, dtype=float), EPS)
    if c.ndim != 2 or c.shape[1] != 6:
        raise ValueError("cell_concentrations 必须是 n×6 数组。")
    quotient = c[:, MV1_NEG] * c[:, T2_POS] / (
        c[:, MV2_NEG] * c[:, T1_POS]
    )
    return params.formal_potential + (
        GAS_CONSTANT * params.temperature / (params.electrons * FARADAY)
    ) * np.log(quotient)


# 作用：用参考ASR乘温度、SOC和循环老化修正得到当前面积比电阻。
def dynamic_cell_resistivity(
    process: Process,
    soc: float,
    cycle_index: float,
    params: ModelParameters,
) -> float:
    """计算参考ASR经温度、弱SOC和老化修正后的动态面积比电阻。

    ASR = ASR_ref × exp[Ea/R(1/T-1/Tref)]
                  × [1+k_soc(SOC-SOC_ref)^2]
                  × [1+k_age(N-1)]

    该函数刻意不接收流量；流量只通过传质系数和浓差过电位参与电压计算，
    从而避免把传质损失重复计入ASR。
    """
    if cycle_index < 1.0:
        raise ValueError("cycle_index 必须不小于 1。")
    reference = (
        params.cell_resistivity_charge
        if process is Process.CHARGE
        else params.cell_resistivity_discharge
    )
    temperature_factor = np.exp(
        params.asr_activation_energy
        / GAS_CONSTANT
        * (1.0 / params.temperature - 1.0 / params.asr_reference_temperature)
    )
    soc_factor = 1.0 + params.asr_soc_coefficient * (
        float(np.clip(soc, 0.0, 1.0)) - params.asr_soc_reference
    ) ** 2
    aging_factor = 1.0 + params.asr_aging_per_cycle * (cycle_index - 1.0)
    return float(reference * temperature_factor * soc_factor * aging_factor)


# 作用：综合开路电压、欧姆极化、浓差极化和活化过电位计算单体负载电压。
def cell_voltage(
    cell_concentration: NDArray[np.float64],
    cell_current: float,
    process: Process,
    params: ModelParameters,
    flow_rate: float,
    cycle_index: float = 1.0,
) -> float:
    """式 (27)–(33)：开路电压、欧姆降和两侧浓差/活化过电位。"""
    c = np.maximum(np.asarray(cell_concentration, dtype=float), EPS)
    current_density = abs(cell_current) / params.electrode_area
    km = mass_transfer_coefficients(params, flow_rate)
    soc_neg = c[MV1_NEG] / max(c[MV1_NEG] + c[MV2_NEG], EPS)
    soc_pos = c[T2_POS] / max(c[T1_POS] + c[T2_POS], EPS)
    local_soc = 0.5 * (soc_neg + soc_pos)
    if process is Process.CHARGE:
        reactants = ((MV2_NEG, km[MV2_NEG]), (T1_POS, km[T1_POS]))
        direction = 1.0
    else:
        reactants = ((MV1_NEG, km[MV1_NEG]), (T2_POS, km[T2_POS]))
        direction = -1.0
    resistivity = dynamic_cell_resistivity(process, local_soc, cycle_index, params)

    overpotentials = []
    for index, coefficient in reactants:
        i_lim = limiting_current_density(c[index], coefficient, params.electrons)
        overpotentials.append(abs(concentration_overpotential(current_density, i_lim, params.temperature, params.electrons)))

    losses = (
        current_density * resistivity
        + abs(params.activation_overpotential_neg)
        + abs(params.activation_overpotential_pos)
        + sum(overpotentials)
    )
    return open_circuit_voltage(c, params) + direction * losses


# 作用：同时计算各单体负载电压及其对电流的解析斜率，供边界和旁路求解使用。
def cell_voltages_and_slopes(
    cell_concentrations: NDArray[np.float64],
    cell_currents: float | NDArray[np.float64],
    process: Process,
    params: ModelParameters,
    flow_rate: float,
    cycle_index: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """批量返回各节电压及其对电流的解析斜率 dV/dI。"""
    if cycle_index < 1.0:
        raise ValueError("cycle_index 必须不小于 1。")
    c = np.maximum(np.asarray(cell_concentrations, dtype=float), EPS)
    if c.ndim != 2 or c.shape[1] != 6:
        raise ValueError("cell_concentrations 必须是 n×6 数组。")
    currents = np.asarray(cell_currents, dtype=float)
    if currents.ndim == 0:
        currents = np.full(c.shape[0], abs(float(currents)))
    elif currents.shape == (c.shape[0],):
        currents = np.abs(currents)
    else:
        raise ValueError("cell_currents 必须是标量或长度等于单电池数的一维数组。")

    current_density = currents / params.electrode_area
    km = mass_transfer_coefficients(params, flow_rate)
    soc_neg = c[:, MV1_NEG] / np.maximum(c[:, MV1_NEG] + c[:, MV2_NEG], EPS)
    soc_pos = c[:, T2_POS] / np.maximum(c[:, T1_POS] + c[:, T2_POS], EPS)
    local_soc = 0.5 * (soc_neg + soc_pos)

    if process is Process.CHARGE:
        reactant_indices = (MV2_NEG, T1_POS)
        direction = 1.0
        reference_asr = params.cell_resistivity_charge
    else:
        reactant_indices = (MV1_NEG, T2_POS)
        direction = -1.0
        reference_asr = params.cell_resistivity_discharge

    temperature_factor = np.exp(
        params.asr_activation_energy
        / GAS_CONSTANT
        * (1.0 / params.temperature - 1.0 / params.asr_reference_temperature)
    )
    soc_factor = 1.0 + params.asr_soc_coefficient * (
        np.clip(local_soc, 0.0, 1.0) - params.asr_soc_reference
    ) ** 2
    aging_factor = 1.0 + params.asr_aging_per_cycle * (cycle_index - 1.0)
    resistivity = reference_asr * temperature_factor * soc_factor * aging_factor

    coefficient = GAS_CONSTANT * params.temperature / (params.electrons * FARADAY)
    concentration_losses = np.zeros(c.shape[0], dtype=float)
    concentration_slopes = np.zeros(c.shape[0], dtype=float)
    for index in reactant_indices:
        limiting_density = (
            params.electrons * FARADAY * km[index] * c[:, index]
        )
        if np.any(limiting_density <= 0.0) or np.any(current_density >= limiting_density):
            raise ValueError("电流密度必须严格小于正的极限电流密度。")
        concentration_losses += np.abs(
            coefficient * np.log(1.0 - current_density / limiting_density)
        )
        concentration_slopes += coefficient / (
            params.electrode_area
            * limiting_density
            * (1.0 - current_density / limiting_density)
        )

    losses = (
        current_density * resistivity
        + abs(params.activation_overpotential_neg)
        + abs(params.activation_overpotential_pos)
        + concentration_losses
    )
    voltages = open_circuit_voltages(c, params) + direction * losses
    slopes = direction * (
        resistivity / params.electrode_area + concentration_slopes
    )
    return voltages, slopes


# 作用：批量返回各单体在给定电流下的负载电压。
def cell_voltages(
    cell_concentrations: NDArray[np.float64],
    cell_currents: float | NDArray[np.float64],
    process: Process,
    params: ModelParameters,
    flow_rate: float,
    cycle_index: float = 1.0,
) -> NDArray[np.float64]:
    """向量化计算各节电压，物理方程与 :func:`cell_voltage` 完全相同。"""
    voltages, _ = cell_voltages_and_slopes(
        cell_concentrations,
        cell_currents,
        process,
        params,
        flow_rate,
        cycle_index,
    )
    return voltages


# 作用：根据长度、截面积和电导率计算电解液通道电阻。
def electrolyte_resistance(length: float, cross_section: float, conductivity: float) -> float:
    """式 (34)：流道或歧管内电解液的电阻。"""
    if min(length, cross_section, conductivity) <= 0.0:
        raise ValueError("长度、横截面积和电导率必须为正数。")
    return length / (cross_section * conductivity)


# 作用：建立并求解电堆流道、歧管与单体组成的旁路电流线性方程组。
def solve_equivalent_circuit(
    stack_current: float,
    voltage_intercept: NDArray[np.float64],
    voltage_slope: NDArray[np.float64],
    resistance: CircuitParameters,
) -> CircuitSolution:
    """式 (35)–(47)：求解线性化后的 5n-2 元等效电路。

    单电池电压采用 E_j(I_j) = voltage_intercept[j] + voltage_slope[j] * I_j。
    这正是 2.10 节所述“围绕单电池电流线性化”的矩阵形式。
    """
    e0 = np.asarray(voltage_intercept, dtype=float)
    slope = np.asarray(voltage_slope, dtype=float)
    if e0.ndim != 1 or slope.shape != e0.shape:
        raise ValueError("voltage_intercept 与 voltage_slope 必须是一维等长数组。")
    n = e0.size
    if n == 1:
        return CircuitSolution(
            cell=np.array([stack_current]),
            channel_pos=np.zeros(1),
            channel_neg=np.zeros(1),
            manifold_pos=np.zeros(0),
            manifold_neg=np.zeros(0),
        )
    if n < 2:
        raise ValueError("至少需要一个单电池。")

    # 未知量顺序：Icell(n), Ich+(n), Ich-(n), Im+(n-1), Im-(n-1)
    size = 5 * n - 2
    b = np.zeros(size, dtype=float)
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    cell0, chp0, chn0, mp0, mn0 = 0, n, 2 * n, 3 * n, 4 * n - 1
    row = 0

    # 作用：把一条等效电路约束写入稀疏线性方程组。
    def equation(terms: tuple[tuple[int, float], ...], rhs: float = 0.0) -> None:
        nonlocal row
        for column, coefficient in terms:
            matrix_rows.append(row)
            matrix_columns.append(column)
            matrix_values.append(coefficient)
        b[row] = rhs
        row += 1

    # 端部 KCL：式 (35)–(40)。
    equation(((cell0, 1.0), (chn0, -2.0)), stack_current)
    equation(((chn0, 1.0), (mn0, 1.0)))
    equation(((chp0, 1.0), (mp0, 1.0)))
    equation(((cell0 + n - 1, -1.0), (chp0 + n - 1, -2.0)), -stack_current)
    equation(((chn0 + n - 1, 1.0), (mn0 + n - 2, -1.0)))
    equation(((chp0 + n - 1, 1.0), (mp0 + n - 2, -1.0)))

    # 中间节点 KCL：式 (43)–(45)，论文下标 N=1,...,n-2。
    for k in range(n - 2):
        equation(((cell0 + k + 1, 1.0), (cell0 + k, -1.0), (chp0 + k, -2.0), (chn0 + k + 1, -2.0)))
        equation(((chp0 + k + 1, 1.0), (mp0 + k + 1, 1.0), (mp0 + k, -1.0)))
        equation(((chn0 + k + 1, 1.0), (mn0 + k + 1, 1.0), (mn0 + k, -1.0)))

    # 端部 KVL：式 (41)、(42)。电压斜率移入系数矩阵，截距移至右端。
    equation(
        ((chp0 + n - 2, resistance.channel_pos), (chp0 + n - 1, -resistance.channel_pos),
         (mp0 + n - 2, -resistance.manifold_pos), (cell0 + n - 1, slope[n - 1])),
        -e0[n - 1],
    )
    equation(
        ((chn0 + n - 2, resistance.channel_neg), (chn0 + n - 1, -resistance.channel_neg),
         (mn0 + n - 2, -resistance.manifold_neg), (cell0 + n - 2, slope[n - 2])),
        -e0[n - 2],
    )

    # 中间网孔 KVL：式 (46)、(47)。
    for k in range(n - 2):
        equation(
            ((chp0 + k, resistance.channel_pos), (chp0 + k + 1, -resistance.channel_pos),
             (mp0 + k, -resistance.manifold_pos), (cell0 + k + 1, slope[k + 1])),
            -e0[k + 1],
        )
        equation(
            ((chn0 + k, resistance.channel_neg), (chn0 + k + 1, -resistance.channel_neg),
             (mn0 + k, -resistance.manifold_neg), (cell0 + k, slope[k])),
            -e0[k],
        )

    if row != size:
        raise RuntimeError(f"内部错误：应生成 {size} 个方程，实际为 {row}。")
    # 该 5n-2 阶KCL/KVL矩阵非常稀疏。使用稀疏求解器不改变方程，
    # 但可避免在每个RK阶段反复执行昂贵的稠密分解。
    if csr_matrix is not None and spsolve is not None:
        matrix = csr_matrix(
            (matrix_values, (matrix_rows, matrix_columns)),
            shape=(size, size),
        )
        x = np.asarray(spsolve(matrix, b), dtype=float)
    else:
        a = np.zeros((size, size), dtype=float)
        np.add.at(a, (matrix_rows, matrix_columns), matrix_values)
        x = np.linalg.solve(a, b)
    return CircuitSolution(x[cell0:chp0], x[chp0:chn0], x[chn0:mp0], x[mp0:mn0], x[mn0:])


# 作用：根据开路截距和电压斜率求各单体及流道、歧管中的电流。
def linearized_circuit_currents(
    cells: NDArray[np.float64],
    stack_current: float,
    process: Process,
    params: ModelParameters,
    flow_rate: float,
    resistance: Optional[CircuitParameters],
    cycle_index: float = 1.0,
) -> CircuitSolution:
    """按 2.10 节在 I=堆电流附近对各单电池电压作数值线性化。"""
    n = params.n_cells
    if resistance is None or n == 1:
        # 不启用分流网络时，各串联单电池电流就是堆电流。
        return CircuitSolution(
            cell=np.full(n, abs(stack_current)),
            channel_pos=np.zeros(n),
            channel_neg=np.zeros(n),
            manifold_pos=np.zeros(max(n - 1, 0)),
            manifold_neg=np.zeros(max(n - 1, 0)),
        )
    reference = np.full(n, abs(stack_current), dtype=float)
    delta = max(abs(stack_current) * 1.0e-6, 1.0e-8)
    e_ref = cell_voltages(cells, reference, process, params, flow_rate, cycle_index)
    e_plus = cell_voltages(
        cells, reference + delta, process, params, flow_rate, cycle_index
    )
    slopes = (e_plus - e_ref) / delta
    intercepts = e_ref - slopes * reference
    return solve_equivalent_circuit(abs(stack_current), intercepts, slopes, resistance)


# 作用：实现论文质量守恒微分方程，返回储液罐和各单体全部浓度的时间导数。
def rhs_mass_balance(
    _time: float,
    y: NDArray[np.float64],
    process: Process,
    stack_current: float,
    flow_rate: float,
    params: ModelParameters,
    circuit: Optional[CircuitParameters] = None,
    cycle_index: float = 1.0,
) -> NDArray[np.float64]:
    """式 (6)–(17) 的完整常微分方程右端。"""
    tank_neg, tank_pos, cells = unpack_state(y, params.n_cells)
    currents = linearized_circuit_currents(
        cells, stack_current, process, params, flow_rate, circuit, cycle_index
    ).cell
    result = np.zeros_like(y, dtype=float)
    d_tank_neg, d_tank_pos, d_cells = unpack_state(result, params.n_cells)

    # 式 (6)–(11)：各储液罐的连续搅拌釜质量守恒。
    d_tank_neg[:] = flow_rate * (cells[:, :3].sum(axis=0) - params.n_cells * tank_neg)
    d_tank_neg[:] -= params.degradation[:3] * params.tank_volume_neg * tank_neg
    d_tank_neg[:] /= params.tank_volume_neg
    d_tank_pos[:] = flow_rate * (cells[:, 3:].sum(axis=0) - params.n_cells * tank_pos)
    d_tank_pos[:] -= params.degradation[3:] * params.tank_volume_pos * tank_pos
    d_tank_pos[:] /= params.tank_volume_pos

    membrane = params.diffusion_membrane * params.membrane_area / params.membrane_thickness
    half_volume = params.cell_volume / 2.0
    current_sign = 1.0 if process is Process.CHARGE else -1.0
    molar_current = currents / (params.electrons * FARADAY)
    tank_concentrations = np.concatenate((tank_neg, tank_pos))
    rates = flow_rate * (tank_concentrations[None, :] - cells)

    # 式 (12)–(17)：对40节独立状态批量施加相同形式的法拉第反应项。
    rates[:, MV1_NEG] += current_sign * molar_current
    rates[:, MV2_NEG] -= current_sign * molar_current
    rates[:, T1_POS] -= current_sign * molar_current
    rates[:, T2_POS] += current_sign * molar_current

    # Fick 扩散及式 (2) 的瞬时自放电反应；每个数组元素仍属于对应单电池。
    flux_mv1 = membrane[MV1_NEG] * cells[:, MV1_NEG]
    flux_mv2 = membrane[MV2_NEG] * (cells[:, MV2_NEG] - cells[:, MV2_POS])
    flux_t1 = membrane[T1_POS] * (cells[:, T1_POS] - cells[:, T1_NEG])
    flux_t2 = membrane[T2_POS] * cells[:, T2_POS]
    rates[:, MV1_NEG] -= flux_mv1 + flux_t2
    rates[:, MV2_NEG] += -flux_mv2 + flux_t2
    rates[:, T1_NEG] += flux_t1 + flux_t2
    rates[:, T1_POS] += -flux_t1 + flux_mv1
    rates[:, T2_POS] -= flux_t2 + flux_mv1
    rates[:, MV2_POS] += flux_mv2 + flux_mv1

    # 论文中的 φ 是每秒分数衰减率；若手头数据为 %/s，输入前需除以 100。
    rates -= params.degradation[None, :] * half_volume * cells
    d_cells[:] = rates / half_volume
    return result


# 作用：分别由各单体正负极氧化还原物种浓度计算局部SOC。
def soc_from_concentration(cells: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """式 (49)–(50)：分别计算负极和正极 SOC。"""
    c = np.maximum(np.asarray(cells, dtype=float), 0.0)
    soc_neg = c[:, MV1_NEG] / np.maximum(c[:, MV1_NEG] + c[:, MV2_NEG], EPS)
    soc_pos = c[:, T2_POS] / np.maximum(c[:, T1_POS] + c[:, T2_POS], EPS)
    return soc_neg, soc_pos


# 作用：按充电取较高侧、放电取较低侧的论文判据计算过程SOC。
def process_soc(cells: NDArray[np.float64], process: Process) -> float:
    """2.11 节：充电取两侧 SOC 较大值，放电取较小值；再取堆内限制单元。"""
    neg, pos = soc_from_concentration(cells)
    per_cell = np.maximum(neg, pos) if process is Process.CHARGE else np.minimum(neg, pos)
    return float(np.max(per_cell) if process is Process.CHARGE else np.min(per_cell))


# 作用：在恒定电流和流量下积分质量守恒方程，直至时长结束或达到截止SOC。
def simulate_constant_current(
    y0: NDArray[np.float64],
    process: Process,
    stack_current: float,
    flow_rate: float,
    params: ModelParameters,
    duration: float,
    soc_cutoff: Optional[float] = None,
    circuit: Optional[CircuitParameters] = None,
    max_step: float = np.inf,
    cycle_index: float = 1.0,
) -> SimulationResult:
    """按 2.11 节积分一次恒流过程，并可在指定 SOC 处终止。"""
    if stack_current <= 0.0 or flow_rate <= 0.0 or duration <= 0.0:
        raise ValueError("电流、流量和仿真时长必须为正数。")
    cutoff_function: Optional[Callable[[NDArray[np.float64]], float]] = None
    cutoff_direction = 0.0
    if soc_cutoff is not None:
        if not 0.0 < soc_cutoff < 1.0:
            raise ValueError("soc_cutoff 必须位于 (0, 1)。")

        # 作用：把当前状态换算成相对截止SOC的事件函数值。
        def cutoff_function(y: NDArray[np.float64]) -> float:
            _, _, cells = unpack_state(y, params.n_cells)
            return process_soc(cells, process) - soc_cutoff

        cutoff_direction = 1.0 if process is Process.CHARGE else -1.0

    time, state, terminated = _adaptive_rk45(
        lambda t, y: rhs_mass_balance(
            t, y, process, stack_current, flow_rate, params, circuit, cycle_index
        ),
        np.asarray(y0, dtype=float),
        duration,
        max_step,
        cutoff_function,
        cutoff_direction,
    )
    message = "达到 SOC 截止条件。" if terminated else "达到设定仿真时长。"
    return SimulationResult(time, state, process, flow_rate, terminated, message)


# 作用：使用自适应RK45算法积分状态方程，并处理截止事件和数值容差。
def _adaptive_rk45(
    function: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    y0: NDArray[np.float64],
    duration: float,
    max_step: float,
    event: Optional[Callable[[NDArray[np.float64]], float]],
    event_direction: float,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """不依赖 SciPy 的 Dormand–Prince 5(4) 自适应积分器。"""
    if not np.all(np.isfinite(y0)):
        raise ValueError("初始状态含非有限值。")
    upper_step = duration if not np.isfinite(max_step) else min(max_step, duration)
    if upper_step <= 0.0:
        raise ValueError("max_step 必须为正数。")
    h = min(upper_step, max(duration / 100.0, 1.0e-6))
    t = 0.0
    y = y0.copy()
    times = [t]
    states = [y.copy()]
    old_event = event(y) if event is not None else None
    terminated = False

    # Dormand–Prince 七级公式；五阶解用于推进，四阶解用于估计局部误差。
    for _ in range(1_000_000):
        if t >= duration:
            break
        h = min(h, upper_step, duration - t)
        k1 = function(t, y)
        k2 = function(t + h / 5, y + h * (k1 / 5))
        k3 = function(t + 3 * h / 10, y + h * (3 * k1 / 40 + 9 * k2 / 40))
        k4 = function(t + 4 * h / 5, y + h * (44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9))
        k5 = function(
            t + 8 * h / 9,
            y + h * (19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729),
        )
        k6 = function(
            t + h,
            y + h * (9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656),
        )
        y_fifth = y + h * (35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84)
        k7 = function(t + h, y_fifth)
        y_fourth = y + h * (
            5179 * k1 / 57600
            + 7571 * k3 / 16695
            + 393 * k4 / 640
            - 92097 * k5 / 339200
            + 187 * k6 / 2100
            + k7 / 40
        )
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_fifth))
        error = float(np.sqrt(np.mean(((y_fifth - y_fourth) / scale) ** 2)))
        if not np.isfinite(error):
            raise RuntimeError("积分误差变为非有限值；请检查参数和极限电流约束。")

        if error <= 1.0:
            old_t, old_y = t, y
            t, y = t + h, y_fifth
            if event is not None:
                new_event = event(y)
                crossed = (
                    (event_direction > 0 and old_event is not None and old_event < 0 <= new_event)
                    or (event_direction < 0 and old_event is not None and old_event > 0 >= new_event)
                )
                if crossed:
                    # 截止点在最后一步内线性插值，避免把状态推进到 SOC 界限之外。
                    denominator = abs(old_event) + abs(new_event)
                    fraction = abs(old_event) / denominator if denominator > 0 else 1.0
                    t = old_t + fraction * h
                    y = old_y + fraction * (y - old_y)
                    terminated = True
                old_event = new_event
            times.append(t)
            states.append(y.copy())
            if terminated:
                break

        factor = 5.0 if error == 0.0 else min(5.0, max(0.2, 0.9 * error ** -0.2))
        h *= factor
        if h < 1.0e-14 * max(1.0, duration):
            raise RuntimeError("积分步长过小；模型可能接近奇点或参数量级不合理。")
    else:
        raise RuntimeError("积分步数超过上限。")
    return np.asarray(times), np.column_stack(states), terminated


# 作用：依据正负极剩余活性物质库存中的较弱一侧计算当前理论容量。
def theoretical_capacity_ah(y: NDArray[np.float64], params: ModelParameters) -> float:
    """根据仍处于正确半电池一侧的活性物质库存计算理论容量。

    串联堆的同一端电流会同时通过 n 个单电池，因此总活性物质摩尔数换算为
    电池堆端容量时需要除以 n。跨到错误一侧的 T+_负 和 MV2+_正不再计入
    可用氧化还原对，这使该指标能反映扩散、交叉污染和降解造成的 SOH 下降。
    """
    tank_neg, tank_pos, cells = unpack_state(np.asarray(y, dtype=float), params.n_cells)
    half_cell_volume = params.cell_volume / 2.0
    usable_mv_moles = (
        params.tank_volume_neg * max(tank_neg[MV1_NEG] + tank_neg[MV2_NEG], 0.0)
        + half_cell_volume * np.maximum(cells[:, MV1_NEG] + cells[:, MV2_NEG], 0.0).sum()
    )
    usable_t_moles = (
        params.tank_volume_pos * max(tank_pos[0] + tank_pos[1], 0.0)
        + half_cell_volume * np.maximum(cells[:, T1_POS] + cells[:, T2_POS], 0.0).sum()
    )
    limiting_moles = min(usable_mv_moles, usable_t_moles)
    return params.electrons * FARADAY * limiting_moles / (params.n_cells * 3600.0)


# 作用：根据当前氧化还原状态估算仍可继续放出的电量。
def remaining_discharge_capacity_ah(y: NDArray[np.float64], params: ModelParameters) -> float:
    """由当前放电反应物 MV+_负 和 T2+_正库存估算剩余可放电容量。"""
    tank_neg, tank_pos, cells = unpack_state(np.asarray(y, dtype=float), params.n_cells)
    half_cell_volume = params.cell_volume / 2.0
    mv1_moles = (
        params.tank_volume_neg * max(tank_neg[MV1_NEG], 0.0)
        + half_cell_volume * np.maximum(cells[:, MV1_NEG], 0.0).sum()
    )
    t2_moles = (
        params.tank_volume_pos * max(tank_pos[1], 0.0)
        + half_cell_volume * np.maximum(cells[:, T2_POS], 0.0).sum()
    )
    return params.electrons * FARADAY * min(mv1_moles, t2_moles) / (params.n_cells * 3600.0)


# 作用：创建SOC、SOE、SOH、功率保持率和内阻健康度所需的新电池基准。
def make_performance_reference(
    healthy_state: NDArray[np.float64],
    params: ModelParameters,
    nominal_current: float,
    rated_capacity_ah: Optional[float] = None,
    rated_energy_wh: Optional[float] = None,
    rated_power_w: Optional[float] = None,
    rated_resistance_ohm: Optional[float] = None,
) -> PerformanceReference:
    """建立健康基准；未给实测额定值时使用模型理论值。

    推荐正式预测时传入电池出厂或首循环实测值。默认额定能量采用
    ``理论容量 × n_cells × 形式电势``，默认功率采用相同名义电压和电流。
    """
    capacity = theoretical_capacity_ah(healthy_state, params) if rated_capacity_ah is None else rated_capacity_ah
    nominal_voltage = params.n_cells * params.formal_potential
    energy = capacity * nominal_voltage if rated_energy_wh is None else rated_energy_wh
    power = abs(nominal_current) * nominal_voltage if rated_power_w is None else rated_power_w
    return PerformanceReference(capacity, energy, power, rated_resistance_ohm)


# 作用：把原始浓度轨迹转换为电压、SOC、SOE、SOH、容量、功率和内阻时间序列。
def analyze_simulation(
    result: SimulationResult,
    stack_current: float,
    params: ModelParameters,
    circuit: Optional[CircuitParameters] = None,
    reference: Optional[PerformanceReference] = None,
    cycle_index: float = 1.0,
) -> PerformanceSeries:
    """把浓度积分结果转换为电压、SOC、SOE、SOH、功率和等效内阻。"""
    if stack_current <= 0.0:
        raise ValueError("stack_current 必须为正数。")
    if result.state.shape[1] != result.time.size:
        raise ValueError("SimulationResult 的时间点数量与状态列数不一致。")
    if reference is None:
        reference = make_performance_reference(result.state[:, 0], params, stack_current)

    count = result.time.size
    cell_u = np.empty((count, params.n_cells), dtype=float)
    stack_u = np.empty(count, dtype=float)
    stack_ocv = np.empty(count, dtype=float)
    soc_neg = np.empty((count, params.n_cells), dtype=float)
    soc_pos = np.empty((count, params.n_cells), dtype=float)
    soc = np.empty(count, dtype=float)
    remaining_capacity = np.empty(count, dtype=float)
    theoretical_capacity = np.empty(count, dtype=float)

    for index in range(count):
        y = result.state[:, index]
        _, _, cells = unpack_state(y, params.n_cells)
        circuit_solution = linearized_circuit_currents(
            cells, stack_current, result.process, params, result.flow_rate, circuit, cycle_index
        )
        cell_u[index] = np.array(
            [
                cell_voltage(
                    cells[j], circuit_solution.cell[j], result.process, params,
                    result.flow_rate, cycle_index,
                )
                for j in range(params.n_cells)
            ]
        )
        stack_u[index] = cell_u[index].sum()
        stack_ocv[index] = sum(open_circuit_voltage(cells[j], params) for j in range(params.n_cells))
        soc_neg[index], soc_pos[index] = soc_from_concentration(cells)
        soc[index] = process_soc(cells, result.process)
        remaining_capacity[index] = remaining_discharge_capacity_ah(y, params)
        theoretical_capacity[index] = theoretical_capacity_ah(y, params)

    # SOE 采用“剩余可放电容量 × 当前 OCV”估算；完整可用能量仍建议用继续放电仿真求积分。
    remaining_energy = remaining_capacity * np.maximum(stack_ocv, 0.0)
    soe = np.clip(remaining_energy / reference.rated_energy_wh, 0.0, 1.0)
    soh = np.clip(theoretical_capacity / reference.rated_capacity_ah, 0.0, 1.0)
    capacity_fade = 1.0 - soh
    power = np.maximum(stack_u, 0.0) * abs(stack_current)
    power_retention = power / reference.rated_power_w
    power_fade = 1.0 - power_retention

    if result.process is Process.CHARGE:
        resistance = (stack_u - stack_ocv) / abs(stack_current)
    else:
        resistance = (stack_ocv - stack_u) / abs(stack_current)
    resistance = np.maximum(resistance, 0.0)
    if reference.rated_resistance_ohm is None:
        resistance_soh = np.full(count, np.nan)
    else:
        resistance_soh = np.clip(reference.rated_resistance_ohm / np.maximum(resistance, EPS), 0.0, 1.0)

    return PerformanceSeries(
        time=result.time.copy(),
        cell_voltage=cell_u,
        stack_voltage=stack_u,
        stack_ocv=stack_ocv,
        soc_neg=soc_neg,
        soc_pos=soc_pos,
        soc=soc,
        remaining_capacity_ah=remaining_capacity,
        remaining_energy_wh=remaining_energy,
        soe=soe,
        soh=soh,
        capacity_fade=capacity_fade,
        power_w=power,
        power_retention=power_retention,
        power_fade=power_fade,
        equivalent_resistance_ohm=resistance,
        resistance_soh=resistance_soh,
    )


# 作用：使用梯形法对时间序列进行积分，并兼容不同NumPy版本。
def _integral(values: NDArray[np.float64], time: NDArray[np.float64]) -> float:
    """兼容不同 NumPy 版本的梯形积分。"""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, time))


# 作用：连续执行多次恒流充放电循环并汇总容量、能量、效率和衰减指标。
def simulate_cycles(
    y0: NDArray[np.float64],
    n_cycles: int,
    charge_current: float,
    discharge_current: float,
    flow_rate: float,
    params: ModelParameters,
    soc_lower: float,
    soc_upper: float,
    max_charge_duration: float,
    max_discharge_duration: float,
    circuit: Optional[CircuitParameters] = None,
    reference: Optional[PerformanceReference] = None,
    max_step: float = np.inf,
) -> MultiCycleResult:
    """执行多次恒流充放电，并计算容量/功率衰减及循环 SOH。

    容量保持率和峰值功率保持率均以第 1 个完整循环为 1。循环 SOH 采用
    放电容量保持率，这是电池循环寿命分析中最常用的容量型 SOH 定义。
    """
    if n_cycles < 1:
        raise ValueError("n_cycles 必须至少为 1。")
    if not 0.0 < soc_lower < soc_upper < 1.0:
        raise ValueError("应满足 0 < soc_lower < soc_upper < 1。")
    if reference is None:
        reference = make_performance_reference(y0, params, discharge_current)

    current_state = np.asarray(y0, dtype=float).copy()
    cycles: list[CycleResult] = []
    first_capacity: Optional[float] = None
    first_power: Optional[float] = None

    for cycle_number in range(1, n_cycles + 1):
        charge = simulate_constant_current(
            current_state,
            Process.CHARGE,
            charge_current,
            flow_rate,
            params,
            max_charge_duration,
            soc_upper,
            circuit,
            max_step,
            float(cycle_number),
        )
        if not charge.terminated_by_cutoff:
            raise RuntimeError(f"第 {cycle_number} 次充电未达到 SOC 上限；请增大 max_charge_duration。")
        charge_output = analyze_simulation(
            charge, charge_current, params, circuit, reference, float(cycle_number)
        )

        discharge = simulate_constant_current(
            charge.state[:, -1],
            Process.DISCHARGE,
            discharge_current,
            flow_rate,
            params,
            max_discharge_duration,
            soc_lower,
            circuit,
            max_step,
            float(cycle_number),
        )
        if not discharge.terminated_by_cutoff:
            raise RuntimeError(f"第 {cycle_number} 次放电未达到 SOC 下限；请增大 max_discharge_duration。")
        discharge_output = analyze_simulation(
            discharge, discharge_current, params, circuit, reference, float(cycle_number)
        )

        charge_capacity = charge_current * (charge.time[-1] - charge.time[0]) / 3600.0
        discharge_capacity = discharge_current * (discharge.time[-1] - discharge.time[0]) / 3600.0
        charge_energy = _integral(charge_output.stack_voltage * charge_current, charge.time) / 3600.0
        discharge_energy = _integral(discharge_output.stack_voltage * discharge_current, discharge.time) / 3600.0
        coulombic = discharge_capacity / charge_capacity
        energy_efficiency = discharge_energy / charge_energy
        voltage_efficiency = energy_efficiency / coulombic
        peak_power = float(np.max(discharge_output.power_w))
        duration = discharge.time[-1] - discharge.time[0]
        average_power = discharge_energy * 3600.0 / duration
        mean_resistance = float(np.mean(discharge_output.equivalent_resistance_ohm))

        if first_capacity is None:
            first_capacity = discharge_capacity
            first_power = peak_power
        capacity_retention = discharge_capacity / first_capacity
        power_retention = peak_power / float(first_power)
        metrics = CycleMetrics(
            cycle=cycle_number,
            charge_capacity_ah=charge_capacity,
            discharge_capacity_ah=discharge_capacity,
            charge_energy_wh=charge_energy,
            discharge_energy_wh=discharge_energy,
            coulombic_efficiency=coulombic,
            energy_efficiency=energy_efficiency,
            voltage_efficiency=voltage_efficiency,
            peak_discharge_power_w=peak_power,
            average_discharge_power_w=average_power,
            mean_discharge_resistance_ohm=mean_resistance,
            capacity_retention=capacity_retention,
            capacity_fade=1.0 - capacity_retention,
            power_retention=power_retention,
            power_fade=1.0 - power_retention,
            soh=capacity_retention,
        )
        cycles.append(CycleResult(charge, discharge, charge_output, discharge_output, metrics))
        current_state = discharge.state[:, -1]
    return MultiCycleResult(cycles, reference)


# 作用：把逐时间点性能结果写入CSV文件。
def export_performance_csv(output: PerformanceSeries, path: str | Path) -> None:
    """将一次过程的七类预测量导出为 UTF-8 CSV。"""
    destination = Path(path)
    with destination.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        cell_headers = [f"cell_voltage_{j + 1}_V" for j in range(output.cell_voltage.shape[1])]
        writer.writerow(
            [
                "time_s",
                *cell_headers,
                "stack_voltage_V",
                "stack_ocv_V",
                "soc",
                "soe",
                "soh",
                "remaining_capacity_Ah",
                "capacity_fade",
                "power_W",
                "power_retention",
                "power_fade",
                "equivalent_resistance_ohm",
            ]
        )
        for i in range(output.time.size):
            writer.writerow(
                [
                    output.time[i],
                    *output.cell_voltage[i],
                    output.stack_voltage[i],
                    output.stack_ocv[i],
                    output.soc[i],
                    output.soe[i],
                    output.soh[i],
                    output.remaining_capacity_ah[i],
                    output.capacity_fade[i],
                    output.power_w[i],
                    output.power_retention[i],
                    output.power_fade[i],
                    output.equivalent_resistance_ohm[i],
                ]
            )


# 作用：把逐循环汇总指标写入CSV文件。
def export_cycle_metrics_csv(result: MultiCycleResult, path: str | Path) -> None:
    """将逐循环容量、功率、SOH 和效率指标导出为 UTF-8 CSV。"""
    destination = Path(path)
    fields = list(CycleMetrics.__dataclass_fields__.keys())
    with destination.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for cycle in result.cycles:
            writer.writerow({name: getattr(cycle.metrics, name) for name in fields})


# 作用：由一个完整充放电循环的容量与能量计算库仑、能量和电压效率。
def efficiencies(
    charge_time: NDArray[np.float64],
    charge_current: NDArray[np.float64],
    charge_voltage: NDArray[np.float64],
    discharge_time: NDArray[np.float64],
    discharge_current: NDArray[np.float64],
    discharge_voltage: NDArray[np.float64],
) -> tuple[float, float, float]:
    """式 (51)–(53)：库仑效率、能量效率和电压效率。"""
    q_charge = _integral(charge_current, charge_time)
    q_discharge = _integral(discharge_current, discharge_time)
    e_charge = _integral(charge_voltage * charge_current, charge_time)
    e_discharge = _integral(discharge_voltage * discharge_current, discharge_time)
    if q_charge <= 0.0 or e_charge <= 0.0 or q_discharge <= 0.0:
        raise ValueError("积分得到的充/放电电量和充电能量必须为正。")
    coulombic = q_discharge / q_charge
    energy = e_discharge / e_charge
    voltage = energy / coulombic
    return coulombic, energy, voltage


