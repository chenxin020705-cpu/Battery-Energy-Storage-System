# -*- coding: utf-8 -*-
"""
面向系统集成的水系有机液流电池设备适配器。

本文件直接封装 ``aofb_model.py`` 中的液流电池物理模型，并提供一个
小型、稳定的两端口接口后面，供系统级调度器与锂电池和超级电容统一使用。

电流符号约定：
    current_A > 0：放电，设备向外输出功率；
    current_A < 0：充电，设备从外部吸收功率。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping, Optional

import numpy as np

from aofb_model import (
    CircuitParameters,
    ModelParameters,
    Process,
    required_constant_flow_rate,
    cell_voltages,
    cell_voltages_and_slopes,
    initial_state,
    linearized_circuit_currents,
    open_circuit_voltages,
    remaining_discharge_capacity_ah,
    rhs_mass_balance,
    soc_from_concentration,
    theoretical_capacity_ah,
    unpack_state,
)

@dataclass
class FlowBatteryHydraulicInputs:
    """正、负极电解液循环回路的实时测量数据。"""

    flow_pos_L_min: Optional[float] = None
    flow_neg_L_min: Optional[float] = None
    pressure_drop_pos_kPa: Optional[float] = None
    pressure_drop_neg_kPa: Optional[float] = None
    tank_level_pos_pct: Optional[float] = None
    tank_level_neg_pct: Optional[float] = None
    pump_pos_running: Optional[bool] = None
    pump_neg_running: Optional[bool] = None
    data_age_s: Optional[float] = None

    # 作用：校验输入或配置中的数值范围，发现非法数据时立即抛出异常。
    def validate(self) -> None:
        """检查水力测量值的基本数值范围。"""
        for name in (
            "flow_pos_L_min",
            "flow_neg_L_min",
            "pressure_drop_pos_kPa",
            "pressure_drop_neg_kPa",
        ):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(value) or float(value) < 0.0
            ):
                raise ValueError(f"{name} 必须是有限的非负数。")
        for name in ("tank_level_pos_pct", "tank_level_neg_pct"):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(value) or not 0.0 <= float(value) <= 100.0
            ):
                raise ValueError(f"{name} 必须位于 [0, 100]。")
        if self.data_age_s is not None and (
            not np.isfinite(self.data_age_s) or self.data_age_s < 0.0
        ):
            raise ValueError("data_age_s 必须是有限的非负数。")


# 旧名称仅作为输入数据类型的兼容别名，不再依赖或加载旧 aorfb_model.py。
AORFBHydraulicInputs = FlowBatteryHydraulicInputs


class AORFBDevice:
    """已移除旧实现；新工程统一使用 FlowBatteryDevice。"""

    # 作用：初始化对象；旧AORFBDevice入口只负责提示改用新版FlowBatteryDevice。
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "旧版 AORFBDevice 已移除；请改用 bms.FlowBatteryDevice。"
        )


# 作用：调用底层单体与旁路模型计算电堆总端电压，且不推进内部状态。
def _stack_voltage(
    y: np.ndarray,
    current: float,
    process: Process,
    params: ModelParameters,
    flow_rate: float,
    cycle_index: float = 1.0,
    circuit: Optional[CircuitParameters] = None,
) -> float:
    """计算各单体端电压之和；超出传质极限时返回对应方向的无穷值。"""
    _, _, cells = unpack_state(y, params.n_cells)
    try:
        circuit_solution = linearized_circuit_currents(
            cells, current, process, params, flow_rate, circuit, cycle_index
        )
        voltages = cell_voltages(
            cells,
            circuit_solution.cell,
            process,
            params,
            flow_rate,
            cycle_index,
        )
        return float(np.sum(voltages))
    except ValueError:
        return np.inf if process is Process.CHARGE else -np.inf


# 作用：计算电堆端电压及其对电流的斜率，供功率反求和边界搜索使用。
def stack_voltage_and_slope(
    y: np.ndarray,
    current: float,
    process: Process,
    params: ModelParameters,
    flow_rate: float,
    cycle_index: float = 1.0,
    circuit: Optional[CircuitParameters] = None,
) -> tuple[float, float]:
    """返回电堆端电压以及端电压对电流幅值的斜率。"""
    if circuit is not None:
        voltage = _stack_voltage(
            y, current, process, params, flow_rate, cycle_index, circuit
        )
        delta = max(1.0e-3 * max(abs(current), 1.0), 1.0e-3)
        current_low = max(0.0, current - delta)
        current_high = current + delta
        voltage_low = _stack_voltage(
            y, current_low, process, params, flow_rate, cycle_index, circuit
        )
        voltage_high = _stack_voltage(
            y, current_high, process, params, flow_rate, cycle_index, circuit
        )
        slope = (voltage_high - voltage_low) / (current_high - current_low)
        return voltage, float(slope)
    _, _, cells = unpack_state(y, params.n_cells)
    try:
        voltages, slopes = cell_voltages_and_slopes(
            cells, current, process, params, flow_rate, cycle_index
        )
        return float(np.sum(voltages)), float(np.sum(slopes))
    except ValueError:
        voltage = np.inf if process is Process.CHARGE else -np.inf
        slope = np.inf if process is Process.CHARGE else -np.inf
        return voltage, slope


# 作用：按储液罐和全部半电池的物质库存分别计算系统正极SOC与负极SOC。
def system_soc_sides(
    y: np.ndarray, params: ModelParameters
) -> tuple[float, float]:
    """按储液罐和所有半电池的物质库存计算正、负极系统 SOC。"""
    tank_neg, tank_pos, cells = unpack_state(y, params.n_cells)
    half_cell_volume = params.cell_volume / 2.0
    neg_reduced = params.tank_volume_neg * tank_neg[0] + half_cell_volume * cells[:, 0].sum()
    neg_oxidized = params.tank_volume_neg * tank_neg[1] + half_cell_volume * cells[:, 1].sum()
    pos_reduced = params.tank_volume_pos * tank_pos[0] + half_cell_volume * cells[:, 3].sum()
    pos_oxidized = params.tank_volume_pos * tank_pos[1] + half_cell_volume * cells[:, 4].sum()
    soc_neg = neg_reduced / max(neg_reduced + neg_oxidized, 1.0e-12)
    soc_pos = pos_oxidized / max(pos_reduced + pos_oxidized, 1.0e-12)
    return float(soc_neg), float(soc_pos)


# 作用：用四阶Runge-Kutta方法把质量守恒方程推进一个时间步。
def rk4_step(
    y: np.ndarray,
    time: float,
    step: float,
    current_controller: Callable[[np.ndarray], float],
    process: Process,
    flow_rate: float,
    params: ModelParameters,
    circuit: Optional[CircuitParameters] = None,
    cycle_index: float = 1.0,
) -> np.ndarray:
    """对论文质量守恒方程执行一步四阶 Runge-Kutta 积分。"""
    # 作用：在RK4子步中根据控制器电流计算当前浓度状态导数。
    def derivative(local_time: float, local_state: np.ndarray) -> np.ndarray:
        current = max(float(current_controller(local_state)), 1.0e-10)
        return rhs_mass_balance(
            local_time, local_state, process, current, flow_rate,
            params, circuit, cycle_index,
        )
    k1 = derivative(time, y)
    k2 = derivative(time + step / 2.0, y + step * k1 / 2.0)
    k3 = derivative(time + step / 2.0, y + step * k2 / 2.0)
    k4 = derivative(time + step, y + step * k3)
    next_state = y + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if np.min(next_state) < -1.0e-6:
        raise RuntimeError("浓度出现明显负值，请减小时间步或检查控制限值。")
    return np.maximum(next_state, 0.0)


# 作用：对Nernst开路电压随SOC积分并结合SOH计算归一化SOE。
def soe_from_ocv_integral(
    soc: float,
    soh: float,
    params: ModelParameters,
    soc_lower: float,
    soc_upper: float,
) -> float:
    """对平衡态 Nernst OCV 关于容量积分，得到归一化 SOE。"""
    coefficient = 2.0 * 8.31 * params.temperature / (params.electrons * 96485.3)
    # 作用：计算SOE积分公式在指定SOC处的原函数值。
    def antiderivative(value: float) -> float:
        clipped = float(np.clip(value, 1.0e-9, 1.0 - 1.0e-9))
        mixing = clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
        return params.formal_potential * clipped + coefficient * mixing
    lower_energy = antiderivative(soc_lower)
    full_window_energy = antiderivative(soc_upper) - lower_energy
    current_energy = antiderivative(float(np.clip(soc, soc_lower, soc_upper))) - lower_energy
    if full_window_energy <= 0.0:
        raise ValueError("SOE 的 SOC 积分区间必须具有正能量。")
    return float(np.clip(current_energy / full_window_energy * soh, 0.0, 1.0))


# 作用：创建当前40节MV/TEMPTMA电堆的默认物理参数，供新版BMS直接使用。
def default_stack_parameters(degradation_rate: float = 0.0) -> ModelParameters:
    """BMS 使用的默认电池堆参数。

    参数数值目前仍来自 MV/TEMPTMA 论文，但函数定义直接放在 BMS 文件中，
    因而 BMS 不再从 CC-CV 实验脚本导入默认参数。以后更换电解液时，可在
    这里统一替换正负极浓度、电位、扩散系数、ASR 和降解率。
    """
    return ModelParameters(
        n_cells=40,
        tank_volume_neg=200.0e-3,  # 负极储液罐体积：200 L。
        tank_volume_pos=200.0e-3,  # 正极储液罐体积：200 L。
        cell_volume=2.0 * 0.9e-3,  # 单节两侧电解液总体积：1.8 L。
        membrane_area=0.30 * 0.50,  # 单节有效膜面积：0.15 m²。
        membrane_thickness=110.0e-6,  # 当前膜厚度：110 μm。
        electrode_area=0.30 * 0.50,  # 单节有效电极面积：0.15 m²。
        flow_cross_section=0.30 * 0.006,  # 电解液流动截面积，m²。
        total_negative_active=2.0e3,  # 当前负极活性物质总浓度：2 mol/L。
        total_positive_active=2.0e3,  # 当前正极活性物质总浓度：2 mol/L。
        formal_potential=1.225,  # 当前体系的单电池形式电位，V。
        temperature=25.0 + 273.15,  # 默认温度，K。
        cell_resistivity_charge=3.0e-4,  # 充电参考 ASR，Ω·m²。
        cell_resistivity_discharge=3.6e-4,  # 放电参考 ASR，Ω·m²。
        # 顺序见 aofb_model.STATE_NAMES：负极三状态、正极三状态。
        diffusion_electrolyte=np.array(
            [5.7e-10, 5.7e-10, 4.8e-10, 4.8e-10, 4.8e-10, 5.7e-10]
        ),
        diffusion_membrane=np.array(
            [0.129e-13, 0.129e-13, 0.845e-13, 0.845e-13, 0.845e-13, 0.129e-13]
        ),
        # 以下动态 ASR 修正系数仍属于待实验标定的模型参数。
        asr_reference_temperature=298.15,
        asr_activation_energy=15000.0,
        asr_soc_reference=0.50,
        asr_soc_coefficient=0.20,
        asr_aging_per_cycle=0.001,
        degradation=np.full(6, degradation_rate),
    )


@dataclass
class FlowBatteryBMSConfig:
    """水系有机液流电池堆的通用 BMS 边界配置。

    默认电压边界只是与现有论文仿真相匹配的保守仿真值，不是已经过硬件
    安全认证的保护值。接入真实电池前，应替换为实验确认的单体和电堆限值。
    """

    stack_voltage_min_V: float = 38.0
    stack_voltage_max_V: float = 58.0
    cell_voltage_min_V: float = 0.95
    cell_voltage_max_V: float = 1.45
    soc_min: float = 0.20
    soc_max: float = 0.80
    charge_current_max_A: float = 60.0
    discharge_current_max_A: float = 60.0
    flow_factor: float = 4.0
    nominal_flow_L_min: Optional[float] = None
    require_hydraulic_measurements: bool = False
    flow_pos_min_L_min: float = 0.0
    flow_neg_min_L_min: float = 0.0
    pressure_drop_pos_min_kPa: Optional[float] = None
    pressure_drop_pos_max_kPa: Optional[float] = None
    pressure_drop_neg_min_kPa: Optional[float] = None
    pressure_drop_neg_max_kPa: Optional[float] = None
    tank_level_pos_min_pct: Optional[float] = None
    tank_level_pos_max_pct: Optional[float] = None
    tank_level_neg_min_pct: Optional[float] = None
    tank_level_neg_max_pct: Optional[float] = None
    hydraulic_measurement_max_age_s: Optional[float] = None
    temperature_min_C: Optional[float] = None
    temperature_max_C: Optional[float] = None
    temperature_low_derate_start_C: Optional[float] = None
    temperature_high_derate_start_C: Optional[float] = None
    soh_derate_start: Optional[float] = None
    soh_min: Optional[float] = None
    current_ramp_rate_A_s: Optional[float] = None
    voltage_safety_margin_V: float = 1.0e-4
    current_solver_tolerance_A: float = 1.0e-4
    current_solver_max_iterations: int = 30
    enforce_limits: bool = True
    # False：快速一致单体模式；True：求解论文 5n-2 元旁路电流网络。
    enable_shunt_current: bool = False
    channel_pos_resistance_ohm: Optional[float] = None
    channel_neg_resistance_ohm: Optional[float] = None
    manifold_pos_resistance_ohm: Optional[float] = None
    manifold_neg_resistance_ohm: Optional[float] = None

    # 作用：校验输入或配置中的数值范围，发现非法数据时立即抛出异常。
    def validate(self) -> None:
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("SOC 边界必须满足 0 <= soc_min < soc_max <= 1。")
        if self.stack_voltage_min_V >= self.stack_voltage_max_V:
            raise ValueError("电堆最低电压必须小于最高电压。")
        if self.cell_voltage_min_V >= self.cell_voltage_max_V:
            raise ValueError("单体最低电压必须小于最高电压。")
        if min(self.charge_current_max_A, self.discharge_current_max_A) < 0.0:
            raise ValueError("充放电电流上限不能为负。")
        if self.flow_factor <= 0.0:
            raise ValueError("flow_factor 必须为正数。")
        if self.nominal_flow_L_min is not None and self.nominal_flow_L_min <= 0.0:
            raise ValueError("nominal_flow_L_min 必须为正数。")
        if min(self.flow_pos_min_L_min, self.flow_neg_min_L_min) < 0.0:
            raise ValueError("最低流量不能为负。")
        if self.temperature_min_C is not None and self.temperature_max_C is not None:
            if self.temperature_min_C >= self.temperature_max_C:
                raise ValueError("最低温度必须小于最高温度。")
        if self.soh_derate_start is not None and self.soh_min is not None:
            if not 0.0 <= self.soh_min < self.soh_derate_start <= 1.0:
                raise ValueError("SOH 降额参数必须满足 0 <= soh_min < soh_derate_start <= 1。")
        if self.current_ramp_rate_A_s is not None and self.current_ramp_rate_A_s <= 0.0:
            raise ValueError("电流爬坡速率必须为正数。")
        if self.current_solver_tolerance_A <= 0.0 or self.current_solver_max_iterations <= 0:
            raise ValueError("电流求解容差和最大迭代次数必须为正数。")
        circuit_values = (
            self.channel_pos_resistance_ohm,
            self.channel_neg_resistance_ohm,
            self.manifold_pos_resistance_ohm,
            self.manifold_neg_resistance_ohm,
        )
        if self.enable_shunt_current:
            if any(value is None for value in circuit_values):
                raise ValueError("开启旁路电流时必须提供正负极流道和歧管的四个电阻。")
            if any(float(value) <= 0.0 for value in circuit_values):
                raise ValueError("旁路电流模型的四个电阻必须都是正数。")


@dataclass
class FlowBatteryDeviceOutput:
    """BMS 推进一步后的主要状态、性能指标和限制信息。"""

    time_s: float
    dt_s: float
    requested_current_A: float
    current_A: float
    process: str
    stack_voltage_V: float
    stack_ocv_V: float
    cell_voltage_min_V: float
    cell_voltage_max_V: float
    power_W: float
    soc: float
    soc_neg: float
    soc_pos: float
    cell_soc_neg_min: float
    cell_soc_neg_max: float
    cell_soc_pos_min: float
    cell_soc_pos_max: float
    soe: float
    soh: float
    remaining_capacity_Ah: float
    capacity_fade: float
    equivalent_resistance_ohm: float
    temperature_C: float
    flow_rate_L_min: float
    i_charge_max_A: float
    i_discharge_max_A: float
    equivalent_full_cycles: float
    limited: bool
    limit_reason: str
    fault: bool
    violations: list[str] = field(default_factory=list)
    cell_voltages_V: list[float] = field(default_factory=list)
    cell_currents_A: list[float] = field(default_factory=list)
    channel_currents_pos_A: list[float] = field(default_factory=list)
    channel_currents_neg_A: list[float] = field(default_factory=list)
    manifold_currents_pos_A: list[float] = field(default_factory=list)
    manifold_currents_neg_A: list[float] = field(default_factory=list)
    max_abs_shunt_current_A: float = 0.0
    shunt_current_enabled: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    # 作用：把设备输出数据类递归转换成普通字典，便于保存和通信。
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FlowBatteryDevice:
    """把正负极电解液浓度模型封装成两端口 BMS 设备。

    每次调用 ``step`` 时先计算允许电流，再用实际电流执行且只执行一次
    RK4 状态推进。40 节单体保持独立浓度状态，不使用平均单体替代。
    """

    device_type = "AQUEOUS_ORGANIC_FLOW_BATTERY"
    current_sign_convention = "positive_discharge_negative_charge"

    # 作用：用物理参数、BMS配置和初始SOC/浓度状态创建新版液流电池设备。
    def __init__(
        self,
        params: Optional[ModelParameters] = None,
        state: Optional[np.ndarray] = None,
        config: Optional[FlowBatteryBMSConfig] = None,
        initial_soc: float = 0.50,
    ) -> None:
        # 默认参数目前仍来自 MV/TEMPTMA 论文；设备接口本身不绑定具体物质名。
        self.params = default_stack_parameters() if params is None else params
        self.config = FlowBatteryBMSConfig() if config is None else config
        self.config.validate()
        self.circuit = self._build_circuit_parameters()
        self.y = (
            initial_state(self.params, initial_soc)
            if state is None
            else np.asarray(state, dtype=float).copy()
        )
        # 用 unpack_state 同时检查输入状态长度是否与单体数一致。
        unpack_state(self.y, self.params.n_cells)
        self._initial_state = self.y.copy()
        self.time_s = 0.0
        self._last_current_A = 0.0
        self._last_process = Process.DISCHARGE
        self.equivalent_full_cycles = 0.0
        self.rated_capacity_Ah = theoretical_capacity_ah(self.y, self.params)
        if self.rated_capacity_Ah <= 0.0:
            raise ValueError("初始状态的理论容量必须为正数。")
        self._nominal_flow_m3_s = self._calculate_nominal_flow()

    # 作用：根据旁路电流开关创建流道与歧管电阻参数；关闭时返回空值。
    def _build_circuit_parameters(self) -> Optional[CircuitParameters]:
        """根据BMS开关构造旁路等效电路；关闭时返回None。"""
        if not self.config.enable_shunt_current:
            return None
        return CircuitParameters(
            channel_pos=float(self.config.channel_pos_resistance_ohm),
            channel_neg=float(self.config.channel_neg_resistance_ohm),
            manifold_pos=float(self.config.manifold_pos_resistance_ohm),
            manifold_neg=float(self.config.manifold_neg_resistance_ohm),
        )

    # 作用：优先使用配置流量，否则根据额定电流和SOC窗口计算标称公共流量。
    def _calculate_nominal_flow(self) -> float:
        if self.config.nominal_flow_L_min is not None:
            return self.config.nominal_flow_L_min / 60000.0
        return required_constant_flow_rate(
            self.config.charge_current_max_A,
            self.config.discharge_current_max_A,
            self.params.total_positive_active,
            self.params.total_negative_active,
            self.config.soc_min,
            self.config.soc_max,
            self.config.flow_factor,
            self.params.electrons,
        )

    # 作用：检查仿真时间步必须为有限正数。
    @staticmethod
    def _validate_dt(dt_s: float) -> None:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s 必须是有限正数。")

    # 作用：把设备恢复到初始或指定浓度状态，并清零时间、电流和等效循环计数。
    def reset(self, state: Optional[np.ndarray] = None) -> None:
        """恢复初始状态，或恢复到用户给定的完整浓度状态。"""
        candidate = self._initial_state if state is None else np.asarray(state, dtype=float)
        unpack_state(candidate, self.params.n_cells)
        self.y = candidate.copy()
        self.time_s = 0.0
        self._last_current_A = 0.0
        self._last_process = Process.DISCHARGE
        self.equivalent_full_cycles = 0.0

    # 作用：复制当前物理参数并把摄氏温度转换为模型使用的开尔文温度。
    def _params_at_temperature(self, temperature_C: float) -> ModelParameters:
        if not np.isfinite(temperature_C):
            raise ValueError("temperature_C 必须是有限数。")
        temperature_K = temperature_C + 273.15
        if temperature_K <= 0.0:
            raise ValueError("绝对温度必须大于 0 K。")
        return replace(self.params, temperature=temperature_K)

    # 作用：检查泵状态、流量、压差、液位和数据时效，返回全部水力越界原因。
    def _hydraulic_violations(
        self, hydraulic: Optional[FlowBatteryHydraulicInputs]
    ) -> list[str]:
        cfg = self.config
        if hydraulic is None:
            return ["缺少液压测量"] if cfg.require_hydraulic_measurements else []
        hydraulic.validate()
        violations: list[str] = []
        required = (
            hydraulic.flow_pos_L_min,
            hydraulic.flow_neg_L_min,
            hydraulic.pump_pos_running,
            hydraulic.pump_neg_running,
        )
        if cfg.require_hydraulic_measurements and any(value is None for value in required):
            violations.append("液压测量不完整")
        if hydraulic.pump_pos_running is False:
            violations.append("正极泵未运行")
        if hydraulic.pump_neg_running is False:
            violations.append("负极泵未运行")
        for side in ("pos", "neg"):
            flow = getattr(hydraulic, f"flow_{side}_L_min")
            flow_min = getattr(cfg, f"flow_{side}_min_L_min")
            if flow is not None and flow < flow_min:
                violations.append(f"{side}侧流量低于下限")
            pressure = getattr(hydraulic, f"pressure_drop_{side}_kPa")
            pressure_min = getattr(cfg, f"pressure_drop_{side}_min_kPa")
            pressure_max = getattr(cfg, f"pressure_drop_{side}_max_kPa")
            if pressure is not None and pressure_min is not None and pressure < pressure_min:
                violations.append(f"{side}侧压降低于下限")
            if pressure is not None and pressure_max is not None and pressure > pressure_max:
                violations.append(f"{side}侧压降高于上限")
            level = getattr(hydraulic, f"tank_level_{side}_pct")
            level_min = getattr(cfg, f"tank_level_{side}_min_pct")
            level_max = getattr(cfg, f"tank_level_{side}_max_pct")
            if level is not None and level_min is not None and level < level_min:
                violations.append(f"{side}侧液位低于下限")
            if level is not None and level_max is not None and level > level_max:
                violations.append(f"{side}侧液位高于上限")
        if (
            cfg.hydraulic_measurement_max_age_s is not None
            and hydraulic.data_age_s is not None
            and hydraulic.data_age_s > cfg.hydraulic_measurement_max_age_s
        ):
            violations.append("液压测量已过期")
        return violations

    # 作用：从正负极实测流量中取保守有效值；无测量时使用标称流量。
    def _resolve_flow_rate(
        self, hydraulic: Optional[FlowBatteryHydraulicInputs]
    ) -> float:
        if hydraulic is None:
            return self._nominal_flow_m3_s
        measured = [
            value
            for value in (hydraulic.flow_pos_L_min, hydraulic.flow_neg_L_min)
            if value is not None and value > 0.0
        ]
        # 电压模型目前只有一个公共流量输入，取两回路较小值作为保守有效流量。
        return min(measured) / 60000.0 if measured else self._nominal_flow_m3_s

    # 作用：根据温度停机和降额区间计算0到1之间的电流能力系数。
    def _temperature_factor(self, temperature_C: float) -> float:
        cfg = self.config
        if cfg.temperature_min_C is not None and temperature_C < cfg.temperature_min_C:
            return 0.0
        if cfg.temperature_max_C is not None and temperature_C > cfg.temperature_max_C:
            return 0.0
        factor = 1.0
        if (
            cfg.temperature_low_derate_start_C is not None
            and cfg.temperature_min_C is not None
            and temperature_C < cfg.temperature_low_derate_start_C
        ):
            span = cfg.temperature_low_derate_start_C - cfg.temperature_min_C
            factor = min(factor, (temperature_C - cfg.temperature_min_C) / max(span, 1e-12))
        if (
            cfg.temperature_high_derate_start_C is not None
            and cfg.temperature_max_C is not None
            and temperature_C > cfg.temperature_high_derate_start_C
        ):
            span = cfg.temperature_max_C - cfg.temperature_high_derate_start_C
            factor = min(factor, (cfg.temperature_max_C - temperature_C) / max(span, 1e-12))
        return float(np.clip(factor, 0.0, 1.0))

    # 作用：由当前理论容量与初始额定容量之比计算容量型SOH。
    def _soh(self, state: Optional[np.ndarray] = None) -> float:
        target = self.y if state is None else state
        return float(np.clip(theoretical_capacity_ah(target, self.params) / self.rated_capacity_Ah, 0.0, 1.0))

    # 作用：根据SOH降额起点和最低允许SOH计算0到1之间的能力系数。
    def _soh_factor(self) -> float:
        cfg = self.config
        if cfg.soh_derate_start is None or cfg.soh_min is None:
            return 1.0
        soh = self._soh()
        return float(np.clip((soh - cfg.soh_min) / (cfg.soh_derate_start - cfg.soh_min), 0.0, 1.0))

    # 作用：计算当前状态下各单体电压以及单体、流道和歧管电流。
    def _electrical_solution(
        self,
        cells: np.ndarray,
        stack_current_A: float,
        process: Process,
        params: ModelParameters,
        flow_rate: float,
    ) -> tuple[np.ndarray, Any]:
        """求各节实际电流与电压；关闭旁路时自动退化为完全一致模式。"""
        circuit_solution = linearized_circuit_currents(
            cells,
            stack_current_A,
            process,
            params,
            flow_rate,
            self.circuit,
            1.0 + self.equivalent_full_cycles,
        )
        voltages = cell_voltages(
            cells,
            circuit_solution.cell,
            process,
            params,
            flow_rate,
            1.0 + self.equivalent_full_cycles,
        )
        return voltages, circuit_solution

    # 作用：在不修改设备状态的前提下汇总一次电压、SOC、SOE、SOH和内阻快照。
    def _snapshot(
        self,
        state: np.ndarray,
        current_magnitude_A: float,
        process: Process,
        params: ModelParameters,
        flow_rate: float,
    ) -> dict[str, Any]:
        _, _, cells = unpack_state(state, params.n_cells)
        cell_u, circuit_solution = self._electrical_solution(
            cells, current_magnitude_A, process, params, flow_rate
        )
        cell_ocv = open_circuit_voltages(cells, params)
        cell_soc_neg, cell_soc_pos = soc_from_concentration(cells)
        soc_neg, soc_pos = system_soc_sides(state, params)
        process_soc = max(soc_neg, soc_pos) if process is Process.CHARGE else min(soc_neg, soc_pos)
        stack_u = float(np.sum(cell_u))
        if self.circuit is None:
            _, slope = stack_voltage_and_slope(
                state,
                current_magnitude_A,
                process,
                params,
                flow_rate,
                1.0 + self.equivalent_full_cycles,
            )
            equivalent_resistance = abs(float(slope))
        else:
            # 旁路网络会使 Icell 随端口电流一起变化，因此用端口电压的
            # 小信号差分计算整个电堆的等效内阻。
            delta_current = max(0.02 * max(current_magnitude_A, 1.0), 0.1)
            current_low = max(current_magnitude_A - delta_current, 1.0e-6)
            current_high = current_magnitude_A + delta_current
            voltage_low = float(
                np.sum(self._electrical_solution(cells, current_low, process, params, flow_rate)[0])
            )
            voltage_high = float(
                np.sum(self._electrical_solution(cells, current_high, process, params, flow_rate)[0])
            )
            equivalent_resistance = abs(voltage_high - voltage_low) / (
                current_high - current_low
            )
        reaction_sign = -1.0 if process is Process.CHARGE else 1.0
        shunt_arrays = (
            circuit_solution.channel_pos,
            circuit_solution.channel_neg,
            circuit_solution.manifold_pos,
            circuit_solution.manifold_neg,
        )
        max_abs_shunt = max(
            (float(np.max(np.abs(values))) for values in shunt_arrays if values.size),
            default=0.0,
        )
        soh = self._soh(state)
        energy_soc = min(soc_neg, soc_pos)
        return {
            "stack_voltage_V": float(stack_u),
            "stack_ocv_V": float(np.sum(cell_ocv)),
            "cell_voltages_V": np.asarray(cell_u, dtype=float),
            "cell_currents_A": reaction_sign * np.asarray(circuit_solution.cell, dtype=float),
            "channel_currents_pos_A": np.asarray(circuit_solution.channel_pos, dtype=float),
            "channel_currents_neg_A": np.asarray(circuit_solution.channel_neg, dtype=float),
            "manifold_currents_pos_A": np.asarray(circuit_solution.manifold_pos, dtype=float),
            "manifold_currents_neg_A": np.asarray(circuit_solution.manifold_neg, dtype=float),
            "max_abs_shunt_current_A": max_abs_shunt,
            "soc": float(process_soc),
            "soc_neg": soc_neg,
            "soc_pos": soc_pos,
            "cell_soc_neg": np.asarray(cell_soc_neg, dtype=float),
            "cell_soc_pos": np.asarray(cell_soc_pos, dtype=float),
            "soh": soh,
            "soe": soe_from_ocv_integral(
                energy_soc, soh, params, self.config.soc_min, self.config.soc_max
            ),
            "remaining_capacity_Ah": remaining_discharge_capacity_ah(state, params),
            "equivalent_resistance_ohm": equivalent_resistance,
        }

    # 作用：预测给定电流执行一个时间步后的状态和输出，但不提交到设备内部。
    def predict_step(
        self,
        current_A: float,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Optional[FlowBatteryHydraulicInputs] = None,
        process_hint: Optional[Process] = None,
    ) -> dict[str, Any]:
        """预测一步但不改变设备状态；正电流放电，负电流充电。"""
        self._validate_dt(dt_s)
        if not np.isfinite(current_A):
            raise ValueError("current_A 必须是有限数。")
        process = process_hint or (
            Process.DISCHARGE if current_A > 0.0
            else Process.CHARGE if current_A < 0.0
            else self._last_process
        )
        magnitude = abs(float(current_A))
        params = self._params_at_temperature(temperature_C)
        flow_rate = self._resolve_flow_rate(hydraulic)
        next_state = rk4_step(
            self.y,
            self.time_s,
            dt_s,
            lambda _state: magnitude,
            process,
            flow_rate,
            params,
            self.circuit,
            1.0 + self.equivalent_full_cycles,
        )
        result = self._snapshot(next_state, magnitude, process, params, flow_rate)
        result.update(
            {
                "state": next_state,
                "process": process,
                "flow_rate_m3_s": flow_rate,
                "temperature_C": temperature_C,
            }
        )
        return result

    # 作用：检查预测结果在指定充放电方向是否违反SOC及单体、电堆电压边界。
    def _directional_violations(
        self, prediction: dict[str, Any], process: Process
    ) -> list[str]:
        cfg = self.config
        violations: list[str] = []
        stack_u = prediction["stack_voltage_V"]
        cell_u = prediction["cell_voltages_V"]
        if not np.isfinite(stack_u) or not np.all(np.isfinite(cell_u)):
            return ["电压计算超出传质极限"]
        if process is Process.CHARGE:
            if stack_u > cfg.stack_voltage_max_V - cfg.voltage_safety_margin_V:
                violations.append("电堆达到充电电压上限")
            if np.max(cell_u) > cfg.cell_voltage_max_V - cfg.voltage_safety_margin_V:
                violations.append("单体达到充电电压上限")
            if max(prediction["soc_neg"], prediction["soc_pos"]) > cfg.soc_max:
                violations.append("系统达到 SOC 上限")
            if max(np.max(prediction["cell_soc_neg"]), np.max(prediction["cell_soc_pos"])) > cfg.soc_max:
                violations.append("单体达到 SOC 上限")
        else:
            if stack_u < cfg.stack_voltage_min_V + cfg.voltage_safety_margin_V:
                violations.append("电堆达到放电电压下限")
            if np.min(cell_u) < cfg.cell_voltage_min_V + cfg.voltage_safety_margin_V:
                violations.append("单体达到放电电压下限")
            if min(prediction["soc_neg"], prediction["soc_pos"]) < cfg.soc_min:
                violations.append("系统达到 SOC 下限")
            if min(np.min(prediction["cell_soc_neg"]), np.min(prediction["cell_soc_pos"])) < cfg.soc_min:
                violations.append("单体达到 SOC 下限")
        return violations

    # 作用：判断当前系统SOC是否已经到达相应充电或放电边界。
    def _already_at_soc_boundary(self, process: Process) -> bool:
        _, _, cells = unpack_state(self.y, self.params.n_cells)
        local_neg, local_pos = soc_from_concentration(cells)
        soc_neg, soc_pos = system_soc_sides(self.y, self.params)
        tolerance = 1.0e-10
        if process is Process.CHARGE:
            return max(soc_neg, soc_pos, float(np.max(local_neg)), float(np.max(local_pos))) >= self.config.soc_max - tolerance
        return min(soc_neg, soc_pos, float(np.min(local_neg)), float(np.min(local_pos))) <= self.config.soc_min + tolerance

    # 作用：通过边界搜索求下一时间步不越界的最大允许充电或放电电流。
    def _max_feasible_current(
        self,
        process: Process,
        upper_A: float,
        dt_s: float,
        temperature_C: float,
        hydraulic: Optional[FlowBatteryHydraulicInputs],
    ) -> tuple[float, str]:
        if upper_A <= 0.0 or self._already_at_soc_boundary(process):
            return 0.0, "SOC边界"

        # 作用：预测候选电流并判断其是否满足全部方向性边界。
        def feasible(magnitude: float) -> tuple[bool, list[str]]:
            signed = magnitude if process is Process.DISCHARGE else -magnitude
            try:
                prediction = self.predict_step(signed, dt_s, temperature_C, hydraulic)
            except (ValueError, RuntimeError, FloatingPointError):
                return False, ["传质或积分边界"]
            issues = self._directional_violations(prediction, process)
            return not issues, issues

        upper_ok, upper_issues = feasible(upper_A)
        if upper_ok:
            return upper_A, "额定电流上限"
        low, high = 0.0, upper_A
        for _ in range(self.config.current_solver_max_iterations):
            if high - low <= self.config.current_solver_tolerance_A:
                break
            midpoint = 0.5 * (low + high)
            midpoint_ok, _ = feasible(midpoint)
            if midpoint_ok:
                low = midpoint
            else:
                high = midpoint
        return low, upper_issues[0] if upper_issues else "模型边界"

    # 作用：综合额定值、温度、SOH、水力、爬坡率和预测边界返回最大允许电流。
    def get_current_limits(
        self,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Optional[FlowBatteryHydraulicInputs] = None,
    ) -> dict[str, Any]:
        """返回下一时间步允许的最大充电/放电电流幅值。"""
        self._validate_dt(dt_s)
        hydraulic_issues = self._hydraulic_violations(hydraulic)
        temperature_factor = self._temperature_factor(temperature_C)
        common_factor = temperature_factor * self._soh_factor()
        if hydraulic_issues:
            common_factor = 0.0
        charge_upper = self.config.charge_current_max_A * common_factor
        discharge_upper = self.config.discharge_current_max_A * common_factor
        if self.config.current_ramp_rate_A_s is not None:
            delta = self.config.current_ramp_rate_A_s * dt_s
            discharge_upper = min(discharge_upper, max(0.0, self._last_current_A + delta))
            charge_upper = min(charge_upper, max(0.0, -self._last_current_A + delta))
        charge_limit, charge_reason = self._max_feasible_current(
            Process.CHARGE, charge_upper, dt_s, temperature_C, hydraulic
        )
        discharge_limit, discharge_reason = self._max_feasible_current(
            Process.DISCHARGE, discharge_upper, dt_s, temperature_C, hydraulic
        )
        return {
            "charge_A": charge_limit,
            "discharge_A": discharge_limit,
            "charge_reason": charge_reason,
            "discharge_reason": discharge_reason,
            "hydraulic_violations": hydraulic_issues,
            "temperature_factor": temperature_factor,
            "soh_factor": self._soh_factor(),
        }

    # 作用：把最大允许电流及对应预测电压换算成电堆直流端充放电功率边界。
    def get_power_limits(
        self,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Optional[FlowBatteryHydraulicInputs] = None,
    ) -> dict[str, Any]:
        """返回下一时间步电堆直流端允许的充放电功率边界。

        所有功率均位于电堆直流端，不包含循环泵和 PCS 损耗。放电功率为
        正值，充电功率 ``charge_W`` 为负值，同时提供非负幅值
        ``charge_magnitude_W``，便于 EMS 使用。
        """
        current_limits = self.get_current_limits(dt_s, temperature_C, hydraulic)
        charge_A = float(current_limits["charge_A"])
        discharge_A = float(current_limits["discharge_A"])
        charge_prediction = self.predict_step(
            -charge_A, dt_s, temperature_C, hydraulic, process_hint=Process.CHARGE
        )
        discharge_prediction = self.predict_step(
            discharge_A,
            dt_s,
            temperature_C,
            hydraulic,
            process_hint=Process.DISCHARGE,
        )
        charge_W = -charge_A * float(charge_prediction["stack_voltage_V"])
        discharge_W = discharge_A * float(discharge_prediction["stack_voltage_V"])
        return {
            **current_limits,
            "charge_W": min(0.0, float(charge_W)),
            "charge_magnitude_W": max(0.0, -float(charge_W)),
            "discharge_W": max(0.0, float(discharge_W)),
            "charge_voltage_V": float(charge_prediction["stack_voltage_V"]),
            "discharge_voltage_V": float(discharge_prediction["stack_voltage_V"]),
            "power_reference": "stack_dc_terminal",
        }

    # 作用：迭代求解满足目标功率P等于U乘I且位于BMS边界内的端口电流。
    def _current_for_power(
        self,
        power_W: float,
        dt_s: float,
        temperature_C: float,
        hydraulic: Optional[FlowBatteryHydraulicInputs],
        power_limits: Mapping[str, Any],
    ) -> float:
        """在当前状态下求解满足 ``P=U(I)I`` 的端口电流。"""
        if abs(power_W) <= 1.0e-12:
            return 0.0
        discharge = power_W > 0.0
        upper_A = float(
            power_limits["discharge_A"] if discharge else power_limits["charge_A"]
        )
        process = Process.DISCHARGE if discharge else Process.CHARGE
        target_magnitude_W = abs(float(power_W))
        low_A, high_A = 0.0, max(0.0, upper_A)
        low_power_W = 0.0
        high_power_W = float(
            power_limits["discharge_W"]
            if discharge
            else power_limits["charge_magnitude_W"]
        )
        if target_magnitude_W >= high_power_W or high_A <= 0.0:
            return high_A if discharge else -high_A
        for _ in range(self.config.current_solver_max_iterations):
            if high_A - low_A <= self.config.current_solver_tolerance_A:
                break
            power_span = high_power_W - low_power_W
            if power_span <= 1.0e-12:
                candidate_A = 0.5 * (low_A + high_A)
            else:
                candidate_A = low_A + (high_A - low_A) * (
                    target_magnitude_W - low_power_W
                ) / power_span
                # 防止割线点长期贴住边界，保留一个小的二分收缩量。
                margin_A = 0.05 * (high_A - low_A)
                candidate_A = float(
                    np.clip(candidate_A, low_A + margin_A, high_A - margin_A)
                )
            signed_A = candidate_A if discharge else -candidate_A
            prediction = self.predict_step(
                signed_A,
                dt_s,
                temperature_C,
                hydraulic,
                process_hint=process,
            )
            predicted_magnitude_W = abs(
                signed_A * float(prediction["stack_voltage_V"])
            )
            if abs(predicted_magnitude_W - target_magnitude_W) <= max(
                0.01, 1.0e-6 * target_magnitude_W
            ):
                return signed_A
            if predicted_magnitude_W < target_magnitude_W:
                low_A = candidate_A
                low_power_W = predicted_magnitude_W
            else:
                high_A = candidate_A
                high_power_W = predicted_magnitude_W
        magnitude_A = 0.5 * (low_A + high_A)
        return magnitude_A if discharge else -magnitude_A

    # 作用：接收功率指令，执行功率限幅、反求电流并推进一次设备状态。
    def step_power(
        self,
        power_W: float,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Optional[FlowBatteryHydraulicInputs] = None,
        power_limits: Optional[Mapping[str, Any]] = None,
    ) -> FlowBatteryDeviceOutput:
        """按电堆直流端功率指令推进一步，内部自动求解端口电流。

        ``power_limits`` 可传入同一调度周期刚读取的边界，避免重复执行昂贵的
        边界搜索。调用者必须保证该边界与设备当前状态和本次工况相对应。
        """
        self._validate_dt(dt_s)
        if not np.isfinite(power_W):
            raise ValueError("power_W 必须是有限数。")
        limits = (
            self.get_power_limits(dt_s, temperature_C, hydraulic)
            if power_limits is None
            else dict(power_limits)
        )
        requested_power_W = float(power_W)
        clipped_power_W = float(
            np.clip(
                requested_power_W,
                -float(limits["charge_magnitude_W"]),
                float(limits["discharge_W"]),
            )
        )
        current_request_A = self._current_for_power(
            clipped_power_W,
            dt_s,
            temperature_C,
            hydraulic,
            limits,
        )
        current_limits = {
            key: limits[key]
            for key in (
                "charge_A",
                "discharge_A",
                "charge_reason",
                "discharge_reason",
                "hydraulic_violations",
                "temperature_factor",
                "soh_factor",
            )
        }
        output = self._commit_current(
            current_request_A,
            dt_s,
            temperature_C,
            hydraulic,
            current_limits,
        )
        power_limited = not np.isclose(
            requested_power_W, clipped_power_W, atol=1.0e-6, rtol=0.0
        )
        if power_limited:
            output.limited = True
            output.limit_reason = (
                limits["discharge_reason"]
                if requested_power_W > 0.0
                else limits["charge_reason"]
            )
        output.raw.update(
            {
                "requested_power_W": requested_power_W,
                "clipped_power_W": clipped_power_W,
                "power_reference": "stack_dc_terminal",
            }
        )
        return output

    # 作用：接收电流指令，执行BMS限流并推进一次设备状态。
    def step(
        self,
        current_A: float,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Optional[FlowBatteryHydraulicInputs] = None,
    ) -> FlowBatteryDeviceOutput:
        """应用 BMS 约束并将物理模型推进一个时间步。"""
        if not np.isfinite(current_A):
            raise ValueError("current_A 必须是有限数。")
        limits = self.get_current_limits(dt_s, temperature_C, hydraulic)
        return self._commit_current(current_A, dt_s, temperature_C, hydraulic, limits)

    # 作用：使用已计算的电流边界提交一次状态更新，并生成完整执行反馈。
    def _commit_current(
        self,
        current_A: float,
        dt_s: float,
        temperature_C: float,
        hydraulic: Optional[FlowBatteryHydraulicInputs],
        limits: Mapping[str, Any],
    ) -> FlowBatteryDeviceOutput:
        """使用已计算的边界提交一次电流和状态，避免重复推进模型。"""
        clipped = float(np.clip(current_A, -limits["charge_A"], limits["discharge_A"]))
        applied = clipped if self.config.enforce_limits else float(current_A)
        requested_process = (
            Process.DISCHARGE if current_A > 0.0
            else Process.CHARGE if current_A < 0.0
            else self._last_process
        )
        prediction = self.predict_step(
            applied,
            dt_s,
            temperature_C,
            hydraulic,
            process_hint=requested_process,
        )
        # 状态只在此处提交一次；get_current_limits 和 predict_step 均不修改 self.y。
        self.y = prediction["state"]
        self.time_s += dt_s
        if applied != 0.0:
            self._last_process = prediction["process"]
        self._last_current_A = applied
        usable_window_capacity = self.rated_capacity_Ah * (self.config.soc_max - self.config.soc_min)
        self.equivalent_full_cycles += abs(applied) * dt_s / max(2.0 * usable_window_capacity * 3600.0, 1e-12)
        process = prediction["process"]
        violations = self._hydraulic_violations(hydraulic) + self._directional_violations(prediction, process)
        limited = not np.isclose(applied, current_A, atol=1e-9, rtol=0.0)
        if limited:
            reason = limits["discharge_reason"] if current_A > 0.0 else limits["charge_reason"]
            if limits["hydraulic_violations"]:
                reason = limits["hydraulic_violations"][0]
        else:
            reason = "未限流"
        cell_u = prediction["cell_voltages_V"]
        local_neg = prediction["cell_soc_neg"]
        local_pos = prediction["cell_soc_pos"]
        power = prediction["stack_voltage_V"] * applied
        return FlowBatteryDeviceOutput(
            time_s=self.time_s,
            dt_s=dt_s,
            requested_current_A=float(current_A),
            current_A=applied,
            process=process.value,
            stack_voltage_V=prediction["stack_voltage_V"],
            stack_ocv_V=prediction["stack_ocv_V"],
            cell_voltage_min_V=float(np.min(cell_u)),
            cell_voltage_max_V=float(np.max(cell_u)),
            power_W=float(power),
            soc=prediction["soc"],
            soc_neg=prediction["soc_neg"],
            soc_pos=prediction["soc_pos"],
            cell_soc_neg_min=float(np.min(local_neg)),
            cell_soc_neg_max=float(np.max(local_neg)),
            cell_soc_pos_min=float(np.min(local_pos)),
            cell_soc_pos_max=float(np.max(local_pos)),
            soe=prediction["soe"],
            soh=prediction["soh"],
            remaining_capacity_Ah=prediction["remaining_capacity_Ah"],
            capacity_fade=1.0 - prediction["soh"],
            equivalent_resistance_ohm=prediction["equivalent_resistance_ohm"],
            temperature_C=temperature_C,
            flow_rate_L_min=prediction["flow_rate_m3_s"] * 60000.0,
            i_charge_max_A=limits["charge_A"],
            i_discharge_max_A=limits["discharge_A"],
            equivalent_full_cycles=self.equivalent_full_cycles,
            limited=limited,
            limit_reason=reason,
            fault=bool(violations),
            violations=violations,
            cell_voltages_V=cell_u.tolist(),
            cell_currents_A=prediction["cell_currents_A"].tolist(),
            channel_currents_pos_A=prediction["channel_currents_pos_A"].tolist(),
            channel_currents_neg_A=prediction["channel_currents_neg_A"].tolist(),
            manifold_currents_pos_A=prediction["manifold_currents_pos_A"].tolist(),
            manifold_currents_neg_A=prediction["manifold_currents_neg_A"].tolist(),
            max_abs_shunt_current_A=prediction["max_abs_shunt_current_A"],
            shunt_current_enabled=self.config.enable_shunt_current,
            raw={
                "temperature_factor": limits["temperature_factor"],
                "soh_factor": limits["soh_factor"],
                "state_vector": self.y.copy(),
            },
        )

    # 作用：读取当前SOC、SOE、SOH、容量、OCV及内部状态，不推进仿真时间。
    def get_state(self, temperature_C: float = 25.0) -> dict[str, Any]:
        """读取当前状态，不推进时间。"""
        params = self._params_at_temperature(temperature_C)
        snapshot = self._snapshot(
            self.y, 0.0, self._last_process, params, self._nominal_flow_m3_s
        )
        return {
            "device_type": self.device_type,
            "time_s": self.time_s,
            "n_cells": self.params.n_cells,
            "soc": snapshot["soc"],
            "soc_neg": snapshot["soc_neg"],
            "soc_pos": snapshot["soc_pos"],
            "soe": snapshot["soe"],
            "soh": snapshot["soh"],
            "capacity_fade": 1.0 - snapshot["soh"],
            "remaining_capacity_Ah": snapshot["remaining_capacity_Ah"],
            "stack_ocv_V": snapshot["stack_ocv_V"],
            "shunt_current_enabled": self.config.enable_shunt_current,
            "cell_currents_A": snapshot["cell_currents_A"].copy(),
            "max_abs_shunt_current_A": snapshot["max_abs_shunt_current_A"],
            "equivalent_full_cycles": self.equivalent_full_cycles,
            "last_current_A": self._last_current_A,
            "state_vector": self.y.copy(),
        }


# 兼容旧脚本：旧类名仍可导入，但新代码统一使用 FlowBattery* 名称。
MVTEMPTMABMSConfig = FlowBatteryBMSConfig
MVTEMPTMADeviceOutput = FlowBatteryDeviceOutput
MVTEMPTMADevice = FlowBatteryDevice


# 作用：运行新版液流电池BMS的最小电流控制演示。
def demo() -> None:
    """通用液流电池 BMS 接口的最小运行示例。"""
    device = FlowBatteryDevice()
    for _ in range(5):
        out = device.step(current_A=20.0, dt_s=1.0, temperature_C=25.0)
        print(
            f"t={out.time_s:.0f}s, I={out.current_A:.2f}A, "
            f"U={out.stack_voltage_V:.3f}V, SOC={out.soc:.5f}, "
            f"limit={out.limit_reason}"
        )


if __name__ == "__main__":
    demo()




