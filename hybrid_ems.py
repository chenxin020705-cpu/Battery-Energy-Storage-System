# -*- coding: utf-8 -*-
"""水系有机液流电池、锂电池、双电层超容与混合超容储能 EMS。

功率符号约定（与 ``bms.FlowBatteryDevice`` 一致）：
    功率 > 0：储能向电网放电；
    功率 < 0：电网向储能充电。

单个调度周期依次执行：
    1. 读取场站 AGC 目标、火电实发功率和四套 BMS 边界；
    2. 生成慢速 SOC/SOE 恢复计划，并据此形成火电基线参考；
    3. 计算火电相对参考值的跟踪偏差，储能只补偿该偏差；
    4. 一级分频把跟踪偏差分成电池低频分量和超容高频分量；
    5. 低频分量再次分频，液流电池承担慢分量、锂电池承担中频残差；
    6. 使用 BMS 实时边界进行有界分配和按时间尺度排序的缺口转移；
    7. 汇总场站跟踪残差、SOC恢复结果和重分配原因；
    8. 向四台 PCS 下发最终功率指令。

本文件不依赖 NumPy 等第三方包，可以直接运行演示。工程接入时，只需实现
AGCReader、ThermalPowerReader、BMSReader 和 PCSInterface 四个小接口。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


FLOW_BATTERY = "flow_battery"
# 兼容旧脚本的导入名称；新代码统一使用 FLOW_BATTERY。
AORFB = FLOW_BATTERY
LITHIUM = "lithium"
HYBRID_SUPERCAPACITOR = "hybrid_supercapacitor"
DOUBLE_LAYER_SUPERCAPACITOR = "double_layer_supercapacitor"
# Backward-compatible symbol: the former generic supercapacitor is now
# explicitly treated as the electric double-layer supercapacitor (EDLC).
SUPERCAPACITOR = DOUBLE_LAYER_SUPERCAPACITOR
DEVICE_NAMES: Tuple[str, ...] = (
    FLOW_BATTERY,
    LITHIUM,
    HYBRID_SUPERCAPACITOR,
    DOUBLE_LAYER_SUPERCAPACITOR,
)
LOW_FREQUENCY_GROUP: Tuple[str, ...] = (FLOW_BATTERY, LITHIUM)
HIGH_FREQUENCY_GROUP: Tuple[str, ...] = (
    HYBRID_SUPERCAPACITOR,
    DOUBLE_LAYER_SUPERCAPACITOR,
)
LOW_FREQUENCY = "low_frequency"
HIGH_FREQUENCY = "high_frequency"
AORFB_SLOW = "flow_battery_slow"
LITHIUM_MEDIUM = "lithium_medium"


def _finite(value: float, name: str) -> float:
    """把输入转换为有限浮点数，并拒绝 NaN 和正负无穷。

    参数：
        value: 需要检查的数值；整数、浮点数或可转换成浮点数的对象均可。
        name: 字段名称，只用于在异常信息中指出是哪一个输入有问题。

    返回：
        转换后的 ``float``。

    异常：
        ``ValueError``：输入不能转换为浮点数，或结果为 NaN/正负无穷。

    EMS 的滤波、求和和比例分配都要求输入为有限数。如果 NaN 进入权重计算，
    它会在后续计算中扩散，因此在接口入口处立即拒绝比事后排查更安全。
    """

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite." % name)
    return value


def _clip(value: float, lower: float, upper: float) -> float:
    """把 ``value`` 限制在闭区间 ``[lower, upper]`` 内。

    返回规则为：低于下限时返回下限，高于上限时返回上限，否则返回原值。
    本函数用于归一化 SOC 裕度和最终功率限幅；调用者负责保证下限不大于上限。
    """

    return min(max(value, lower), upper)


@dataclass
class AGCCommand:
    """场站并网点AGC有功功率目标。

    新的火储联合模式中，``power_W`` 是场站总有功目标；EMS减去火电实发和
    其他电源实发，只让储能承担剩余偏差。为兼容旧测试，未传火电状态时仍可
    把它解释为储能净功率目标。
    """

    power_W: float
    timestamp_s: float

    def validate(self) -> None:
        """校验 AGC 功率和时间戳均为有限数。

        ``power_W`` 可以为正、负或零：正值表示要求储能放电，负值表示要求
        储能充电。这里不限制AGC幅值，因为“能否满足”应由后面的BMS边界和
        功率重分配判断。该函数只做数据质量检查，不修改指令。
        """

        _finite(self.power_W, "AGC power_W")
        _finite(self.timestamp_s, "AGC timestamp_s")


@dataclass
class ThermalPowerState:
    """火电及其他非储能电源在并网计量点的实时功率状态。"""

    actual_power_W: float
    timestamp_s: float
    other_power_W: float = 0.0
    available: bool = True
    data_age_s: float = 0.0

    def validate(self, max_data_age_s: float) -> None:
        """校验火电测量值、时间戳、可用状态和数据新鲜度。"""
        _finite(self.actual_power_W, "thermal actual_power_W")
        _finite(self.other_power_W, "other power_W")
        _finite(self.timestamp_s, "thermal timestamp_s")
        _finite(self.data_age_s, "thermal data_age_s")
        if self.data_age_s < 0.0:
            raise ValueError("thermal data_age_s cannot be negative.")
        if not self.available:
            raise ValueError("Thermal power measurement is unavailable.")
        if self.data_age_s > max_data_age_s:
            raise ValueError("Thermal power measurement is stale.")


@dataclass
class DeviceBMSState:
    """EMS 所需的最小 BMS 数据集。

    ``max_charge_power_W`` 使用正的功率幅值；例如最大允许充电功率为
    -2000 W 时，本字段填写 2000。
    """

    name: str
    soc: float
    max_discharge_power_W: float
    max_charge_power_W: float
    available: bool = True
    fault: bool = False
    data_age_s: float = 0.0
    soc_min: Optional[float] = None
    soc_max: Optional[float] = None
    operating_state: str = "READY"
    soh: float = 1.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    # BMS给出的归一化温度裕度：1表示温度不降额，0表示该设备不应参与分配。
    temperature_factor: float = 1.0
    # BMS给出的无量纲边际寿命代价；越大表示本次使用造成的寿命损耗越高。
    marginal_life_cost: float = 0.0
    # 当前可用最大能量；缺失时禁用基于能量偏差的主动SOC/SOE恢复。
    energy_capacity_Wh: Optional[float] = None
    # 能量状态优先用于恢复计划；缺失时退回SOC。
    soe: Optional[float] = None

    def validate(self) -> None:
        """校验单个储能设备提供给 EMS 的 BMS 数据。

        检查内容包括：
        1. 设备名称必须属于系统定义的四种储能；
        2. SOC、SOH、温度裕度和效率必须位于规定区间；
        3. 充放电功率边界必须以非负“幅值”提供；
        4. 数据年龄与边际寿命代价不能为负；
        5. 所有参与计算的数值必须为有限数。

        该函数不判断数据是否过期，也不根据SOC切断充放电能力。数据过期和
        运行边界属于EMS策略，由 ``_effective_limits`` 统一处理。

        异常：
            任一字段不符合约定时抛出 ``ValueError``。
        """

        if self.name not in DEVICE_NAMES:
            raise ValueError("Unknown device name: %s" % self.name)
        _finite(self.soc, "%s soc" % self.name)
        _finite(self.max_discharge_power_W, "%s max discharge power" % self.name)
        _finite(self.max_charge_power_W, "%s max charge power" % self.name)
        _finite(self.data_age_s, "%s data age" % self.name)
        _finite(self.soh, "%s soh" % self.name)
        _finite(self.charge_efficiency, "%s charge efficiency" % self.name)
        _finite(self.discharge_efficiency, "%s discharge efficiency" % self.name)
        _finite(self.temperature_factor, "%s temperature factor" % self.name)
        _finite(self.marginal_life_cost, "%s marginal life cost" % self.name)
        if self.energy_capacity_Wh is not None:
            _finite(self.energy_capacity_Wh, "%s energy capacity" % self.name)
        if self.soe is not None:
            _finite(self.soe, "%s soe" % self.name)
        if not 0.0 <= self.soc <= 1.0:
            raise ValueError("%s SOC must be in [0, 1]." % self.name)
        if self.max_discharge_power_W < 0.0 or self.max_charge_power_W < 0.0:
            raise ValueError("BMS power limits must be non-negative magnitudes.")
        if self.data_age_s < 0.0:
            raise ValueError("BMS data_age_s cannot be negative.")
        if not 0.0 <= self.soh <= 1.0:
            raise ValueError("%s SOH must be in [0, 1]." % self.name)
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must be in (0, 1].")
        if not 0.0 < self.discharge_efficiency <= 1.0:
            raise ValueError("discharge_efficiency must be in (0, 1].")
        if not 0.0 <= self.temperature_factor <= 1.0:
            raise ValueError("temperature_factor must be in [0, 1].")
        if self.marginal_life_cost < 0.0:
            raise ValueError("marginal_life_cost cannot be negative.")
        if self.energy_capacity_Wh is not None and self.energy_capacity_Wh <= 0.0:
            raise ValueError("energy_capacity_Wh must be positive when provided.")
        if self.soe is not None and not 0.0 <= self.soe <= 1.0:
            raise ValueError("soe must be in [0, 1].")
        if self.soc_min is not None:
            _finite(self.soc_min, "%s soc_min" % self.name)
        if self.soc_max is not None:
            _finite(self.soc_max, "%s soc_max" % self.name)
        if self.soc_min is not None and self.soc_max is not None:
            if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
                raise ValueError("BMS SOC limits must satisfy 0 <= min < max <= 1.")

    @classmethod
    def from_flow_battery(
        cls,
        power_limits: Mapping[str, Any],
        state: Mapping[str, Any],
        soc_min: float,
        soc_max: float,
        data_age_s: float = 0.0,
    ) -> "DeviceBMSState":
        """把通用 ``FlowBatteryDevice`` 状态和直流功率边界转换为EMS输入。"""
        hydraulic_issues = list(power_limits.get("hydraulic_violations", ()))
        temperature_factor = float(power_limits.get("temperature_factor", 1.0))
        available = not hydraulic_issues and temperature_factor > 0.0
        discharge = max(0.0, float(power_limits["discharge_W"]))
        charge = max(0.0, float(power_limits["charge_magnitude_W"]))
        if not available:
            operating_state = "FAULT"
        elif discharge <= 0.0 and charge <= 0.0:
            operating_state = "BLOCKED"
        elif temperature_factor < 1.0 or float(power_limits.get("soh_factor", 1.0)) < 1.0:
            operating_state = "DERATED"
        else:
            operating_state = "READY"
        return cls(
            name=FLOW_BATTERY,
            soc=float(state["soc"]),
            max_discharge_power_W=discharge,
            max_charge_power_W=charge,
            available=available,
            fault=bool(hydraulic_issues),
            data_age_s=float(data_age_s),
            soc_min=float(soc_min),
            soc_max=float(soc_max),
            operating_state=operating_state,
            soh=float(state.get("soh", 1.0)),
            temperature_factor=temperature_factor,
            soe=(float(state["soe"]) if state.get("soe") is not None else None),
        )

    @classmethod
    def from_aorfb(
        cls,
        constraints: Mapping[str, Any],
        state: Mapping[str, Any],
        data_age_s: float = 0.0,
    ) -> "DeviceBMSState":
        """把现有 ``AORFBDevice`` 的状态和边界转换成 EMS 输入。

        这里采用净功率边界，即已经扣除了液流电池循环泵功耗。所有 PCS
        指令也应定义在相同的并网计量点；若现场 PCS 使用电堆侧功率，需由
        PCS 适配层补偿泵功率。

        参数：
            constraints: ``AORFBDevice.extract_constraints`` 的返回结果。
                其中放电净功率上限应为正值；充电净功率沿用系统符号约定，
                通常为负值。
            state: ``AORFBDevice.get_state`` 返回的当前状态，至少包含 ``soc``。
            data_age_s: 这组状态/边界距离当前调度时刻的时间，单位为秒。

        转换规则：
            - ``p_discharge_net_max_W`` 直接转换为非负放电上限；
            - 对 ``p_charge_net_max_W`` 取负号，得到非负充电功率幅值；
            - FAULT、LOCKED或已有约束违规会使设备标记为不可用；
            - SOH缺失时暂按1处理，温度降额因子缺失时也暂按1处理。

        返回：
            可直接放入 ``HybridStorageEMS.dispatch`` 的 ``DeviceBMSState``。

        注意：
            本适配器不会再次计算泵耗，也不会把净功率换回电堆功率，避免在
            BMS和EMS两层重复扣除辅助功耗。
        """

        discharge = max(0.0, float(constraints["p_discharge_net_max_W"]))
        charge_signed = float(constraints["p_charge_net_max_W"])
        charge = max(0.0, -charge_signed)
        operating_state = str(constraints.get("operating_state", "READY"))
        no_fault = bool(constraints.get("state_within_constraints", True))
        no_fault = no_fault and operating_state not in ("FAULT", "LOCKED")
        derating_factors = constraints.get("derating_factors", {})
        if not isinstance(derating_factors, Mapping):
            derating_factors = {}
        return cls(
            name=FLOW_BATTERY,
            soc=float(state["soc"]),
            max_discharge_power_W=discharge,
            max_charge_power_W=charge,
            available=no_fault,
            fault=not no_fault,
            data_age_s=float(data_age_s),
            soc_min=float(constraints.get("soc_min", 0.10)),
            soc_max=float(constraints.get("soc_max", 0.90)),
            operating_state=operating_state,
            soh=float(state.get("soh", 1.0)),
            temperature_factor=float(derating_factors.get("temperature", 1.0)),
            energy_capacity_Wh=(
                float(state["energy_current_max_Wh"])
                if state.get("energy_current_max_Wh") is not None
                else None
            ),
            soe=float(state.get("soe", state["soc"])),
        )


@dataclass
class DevicePolicy:
    """某类储能在 EMS 中的调度参数。

    ``soc_min/soc_max`` 是EMS默认安全工作区间；若BMS状态中提供设备实时
    SOC边界，则优先使用BMS边界。各 ``*_weight_exponent`` 控制对应状态量
    对动态分配权重的影响：0表示忽略，1表示线性影响，大于1表示加强惩罚。
    ``redistribution_weight`` 是设备的基础偏好，``life_cost_weight`` 用于
    调整边际寿命代价的敏感程度。

    ``soc_target`` 同时用于方向偏好和主动SOC/SOE恢复。恢复功率依据设备可用
    能量、目标偏差和恢复时间计算，并受恢复功率比例及BMS边界共同限制。
    """

    rated_power_W: float
    soc_target: float
    soc_min: float
    soc_max: float
    # Base preference and exponents in the group-allocation weight formula.
    redistribution_weight: float = 1.0
    soc_weight_exponent: float = 1.0
    efficiency_weight_exponent: float = 1.0
    soh_weight_exponent: float = 1.0
    temperature_weight_exponent: float = 1.0
    life_cost_weight: float = 1.0
    soc_target_weight_gain: float = 1.0
    soc_target_factor_min: float = 0.25
    soc_target_factor_max: float = 2.0
    soc_recovery_deadband: float = 0.03
    soc_recovery_time_s: float = 1800.0
    soc_recovery_power_fraction: float = 0.10

    def validate(self, name: str) -> None:
        """校验某一设备的EMS策略参数。

        参数：
            name: 设备名称，仅用于生成清晰的错误信息。

        约束：
            额定功率必须为正；SOC必须满足
            ``0 <= soc_min < soc_target < soc_max <= 1``；基础权重、各状态
            指数及寿命权重均不得为负。

        异常：
            任一配置不合理时抛出 ``ValueError``，防止EMS带错误参数启动。
        """

        if self.rated_power_W <= 0.0:
            raise ValueError("%s rated_power_W must be positive." % name)
        if not 0.0 <= self.soc_min < self.soc_target < self.soc_max <= 1.0:
            raise ValueError(
                "%s SOC values must satisfy min < target < max in [0, 1]." % name
            )
        if self.redistribution_weight < 0.0:
            raise ValueError("redistribution_weight cannot be negative.")
        for field_name in (
            "soc_weight_exponent",
            "efficiency_weight_exponent",
            "soh_weight_exponent",
            "temperature_weight_exponent",
            "life_cost_weight",
            "soc_target_weight_gain",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError("%s cannot be negative." % field_name)
        if not 0.0 < self.soc_target_factor_min <= self.soc_target_factor_max:
            raise ValueError("SOC target factor limits must be positive and ordered.")
        if not 0.0 <= self.soc_recovery_deadband < 1.0:
            raise ValueError("soc_recovery_deadband must be in [0, 1).")
        if self.soc_recovery_time_s <= 0.0:
            raise ValueError("soc_recovery_time_s must be positive.")
        if not 0.0 <= self.soc_recovery_power_fraction <= 1.0:
            raise ValueError("soc_recovery_power_fraction must be in [0, 1].")


def default_device_policies() -> Dict[str, DevicePolicy]:
    """建立四类储能的默认EMS策略参数。

    返回字典的键与 ``DEVICE_NAMES`` 完全一致，分别包含水系有机液流电池、
    锂电池、混合超级电容和双电层超级电容。默认值只用于演示和代码联调，
    其中额定功率、SOC边界、目标SOC以及基础优先级均需要根据现场设备资料、
    BMS允许边界和AGC回放试验重新标定。

    返回：
        ``{设备名称: DevicePolicy}`` 映射。每次调用都会创建一套新的对象，
        避免不同EMS实例意外共享和修改同一个配置字典。
    """

    return {
        FLOW_BATTERY: DevicePolicy(
            rated_power_W=5000.0,
            soc_target=0.50,
            soc_min=0.10,
            soc_max=0.90,
            redistribution_weight=1.0,
            soc_recovery_time_s=3600.0,
            soc_recovery_power_fraction=0.10,
        ),
        LITHIUM: DevicePolicy(
            rated_power_W=3000.0,
            soc_target=0.55,
            soc_min=0.15,
            soc_max=0.90,
            redistribution_weight=0.8,
            soc_recovery_time_s=1800.0,
            soc_recovery_power_fraction=0.10,
        ),
        HYBRID_SUPERCAPACITOR: DevicePolicy(
            rated_power_W=2500.0,
            soc_target=0.50,
            soc_min=0.15,
            soc_max=0.85,
            redistribution_weight=1.0,
            soc_recovery_time_s=120.0,
            soc_recovery_power_fraction=0.20,
        ),
        DOUBLE_LAYER_SUPERCAPACITOR: DevicePolicy(
            rated_power_W=2000.0,
            soc_target=0.50,
            soc_min=0.20,
            soc_max=0.80,
            redistribution_weight=1.2,
            soc_recovery_time_s=60.0,
            soc_recovery_power_fraction=0.20,
        ),
    }


@dataclass
class EMSConfig:
    """混合储能EMS的系统级配置。

    ``frequency_split_tau_s`` 决定电池组与超容组之间的一级分频；
    ``low_group_split_tau_s`` 决定液流电池与锂电池之间的二级分频，且必须更大。
    ``max_bms_data_age_s`` 是BMS数据允许的最大年龄，超过后设备本周期被视为
    不可用。``tracking_tolerance_W`` 用于浮点比较、迭代停止和AGC完成判断。
    ``bumpless_filter_initialization`` 决定首个样本是否无扰初始化。
    """

    # One tunable boundary between the low-frequency battery group and the
    # high-frequency supercapacitor group.  This is an initial value only.
    frequency_split_tau_s: float = 15.0
    low_group_split_tau_s: float = 60.0
    thermal_tracking_deadband_W: float = 0.0
    max_thermal_data_age_s: float = 5.0
    allow_aorfb_fast_emergency_support: bool = False
    max_bms_data_age_s: float = 5.0
    tracking_tolerance_W: float = 1.0e-6
    # 以下旧参数只为兼容已有调用；新的火储联合模式使用主动SOC/SOE恢复计划。
    soc_balancing_gain_W_per_soc: float = 2000.0
    soc_balancing_max_power_W: float = 250.0
    soc_balancing_deadband: float = 0.02
    # 首周期使用调用者传入的dt，后续周期优先使用连续AGC时间戳之差。
    use_agc_timestamp_delta: bool = True
    # 默认从零滤波状态启动，使初始快速变化先由快速设备承担。
    bumpless_filter_initialization: bool = False
    device_policies: Dict[str, DevicePolicy] = field(
        default_factory=default_device_policies
    )

    def validate(self) -> None:
        """校验系统级配置以及四种储能的全部设备策略。

        时间常数必须为正，BMS数据年龄阈值和跟踪容差不得为负；策略字典必须
        恰好包含四个标准设备名称，不能缺少或混入未知设备。校验成功时无返回值，
        失败时抛出 ``ValueError``。构造 ``HybridStorageEMS`` 时会自动调用。
        """

        if self.frequency_split_tau_s <= 0.0:
            raise ValueError("frequency_split_tau_s must be positive.")
        if self.low_group_split_tau_s <= self.frequency_split_tau_s:
            raise ValueError(
                "low_group_split_tau_s must be greater than frequency_split_tau_s."
            )
        if self.thermal_tracking_deadband_W < 0.0:
            raise ValueError("thermal_tracking_deadband_W cannot be negative.")
        if self.max_thermal_data_age_s < 0.0:
            raise ValueError("max_thermal_data_age_s cannot be negative.")
        if self.max_bms_data_age_s < 0.0:
            raise ValueError("max_bms_data_age_s cannot be negative.")
        if self.tracking_tolerance_W < 0.0:
            raise ValueError("tracking_tolerance_W cannot be negative.")
        if self.soc_balancing_gain_W_per_soc < 0.0:
            raise ValueError("soc_balancing_gain_W_per_soc cannot be negative.")
        if self.soc_balancing_max_power_W < 0.0:
            raise ValueError("soc_balancing_max_power_W cannot be negative.")
        if not 0.0 <= self.soc_balancing_deadband < 1.0:
            raise ValueError("soc_balancing_deadband must be in [0, 1).")
        if set(self.device_policies) != set(DEVICE_NAMES):
            raise ValueError("Policies are required for all four storage devices.")
        for name, policy in self.device_policies.items():
            policy.validate(name)


class FirstOrderLowPass:
    """按实际采样周期递推的一阶低通滤波器。

    离散递推公式为：
    ``y(k) = alpha*y(k-1) + (1-alpha)*u(k)``，其中
    ``alpha = exp(-dt/tau)``。输出 ``y`` 是低频分量，输入减去输出则得到
    高频分量。对象内部保存上一次输出，因此必须按时间顺序调用 ``update``。
    """

    def __init__(self, tau_s: float) -> None:
        """创建滤波器并把内部状态初始化为“尚未收到首个样本”。

        参数：
            tau_s: 滤波时间常数，单位秒，必须大于0。时间常数越大，输出对
                输入变化越迟缓；时间常数越小，输出越接近原始输入。

        初始 ``value=0`` 只是占位值，首个样本如何处理由 ``update`` 的
        ``bumpless`` 参数决定。
        """

        if tau_s <= 0.0:
            raise ValueError("tau_s must be positive.")
        self.tau_s = float(tau_s)
        self.value = 0.0
        self.initialized = False

    def reset(self, value: Optional[float] = None) -> None:
        """重置低通滤波器的记忆状态。

        参数：
            value: 若为 ``None``，清零并标记为未初始化，下一次 ``update``
                将重新执行首样本逻辑；若给定数值，则把它作为上一时刻低频
                功率并标记为已初始化，可用于系统恢复或无扰切换。

        该函数只改变滤波器内部状态，不产生功率输出。
        """

        self.value = 0.0 if value is None else float(value)
        self.initialized = value is not None

    def update(self, input_value: float, dt_s: float, bumpless: bool) -> float:
        """输入一个新样本并返回本周期的一阶低通输出。

        参数：
            input_value: 当前净功率缺口，单位W，正值放电、负值充电。
            dt_s: 当前样本与上一样本的实际时间间隔，单位秒，必须大于0。
            bumpless: 首次调用是否采用无扰初始化。为True时首个输出直接等于
                首个输入，因此不会人为产生高频冲击；为False时从零状态按
                递推公式启动，首个样本也会被分解成低频和高频。

        返回：
            更新后的低频分量 ``value``，单位W。

        状态变化：
            保存本次输出，供下一周期作为 ``y(k-1)`` 使用。
        """

        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        input_value = _finite(input_value, "filter input")
        if not self.initialized:
            self.value = input_value if bumpless else 0.0
            self.initialized = True
            if bumpless:
                return self.value
        alpha = math.exp(-float(dt_s) / self.tau_s)
        self.value = alpha * self.value + (1.0 - alpha) * input_value
        return self.value


@dataclass
class DispatchResult:
    """一次EMS调度的完整结果和审计信息。

    除最终PCS指令外，还保存分频结果、组内初次分配、动态权重、BMS有效边界、
    组内缺口、跨组补偿量和未满足AGC功率。这样现场可以追溯“为什么某台设备
    分到这个功率”。所有功率字段均遵循正值放电、负值充电的统一约定；只有
    ``max_charge_magnitude_W`` 使用非负充电幅值。
    """

    timestamp_s: float
    dt_s: float
    agc_requested_W: float
    thermal_actual_W: float
    other_power_W: float
    thermal_reference_W: float
    raw_thermal_tracking_error_W: float
    storage_tracking_request_W: float
    storage_reference_W: float
    soc_recovery_requested_W: Dict[str, float]
    soc_recovery_total_W: float
    frequency_components_W: Dict[str, float]
    low_group_components_W: Dict[str, float]
    group_setpoints_W: Dict[str, Dict[str, float]]
    allocation_weights: Dict[str, Dict[str, float]]
    group_residuals_W: Dict[str, float]
    decomposed_power_W: Dict[str, float]
    # 兼容旧接口；新模式中记录主动SOC/SOE恢复计划。
    soc_correction_W: Dict[str, float]
    soc_corrected_power_W: Dict[str, float]
    bms_limited_power_W: Dict[str, float]
    bms_limits_W: Dict[str, Dict[str, float]]
    limit_reasons: Dict[str, List[str]]
    gap_before_redistribution_W: float
    redistribution_W: Dict[str, float]
    final_setpoints_W: Dict[str, float]
    tracking_residual_W: float
    agc_fully_met: bool
    redistribution_reasons: List[str]
    pcs_accepted: Optional[bool] = None
    actual_power_W: Optional[Dict[str, float]] = None
    actual_tracking_residual_W: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """递归地把调度结果转换成普通字典。"""
        return asdict(self)


@dataclass
class PCSDispatchFeedback:
    """PCS执行反馈；实际功率与EMS一样采用正放电、负充电。"""

    accepted: bool
    actual_power_W: Optional[Dict[str, float]] = None


@dataclass
class DevicePowerFeedback:
    """任意储能设备在统一并网计量点上的单次功率执行反馈。"""

    name: str
    requested_power_W: float
    actual_power_W: float
    accepted: bool
    raw: Any = None


class StorageDeviceAdapter(ABC):
    """所有储能设备接入EMS时必须实现的统一适配接口。

    ``read_state`` 只读取状态和下一步功率边界，不能推进设备；
    ``apply_power`` 每个调度周期只执行一次功率并推进一次设备状态。
    两个接口中的功率均位于统一并网计量点，正值放电、负值充电。
    """

    name: str

    @abstractmethod
    def read_state(
        self, dt_s: float, timestamp_s: Optional[float] = None
    ) -> DeviceBMSState:
        """读取设备状态及下一时间步的动态充放电功率边界。"""

    @abstractmethod
    def apply_power(
        self,
        power_W: float,
        dt_s: float,
        timestamp_s: Optional[float] = None,
    ) -> DevicePowerFeedback:
        """执行并网点功率指令并返回实际达到的并网点功率。"""


class AGCReader(ABC):
    """AGC数据源的抽象接口。

    Python不会因为声明了 ``read`` 就自动读取指令。现场需要编写子类并实现
    通信、数据库或文件读取逻辑；抽象基类只规定EMS期望的统一调用形式。
    """

    @abstractmethod
    def read(self) -> AGCCommand:
        """读取并返回最新的一条 ``AGCCommand``。

        实现类应同时提供功率值和对应时间戳，并自行处理通信超时、报文校验、
        重复帧等现场问题。抽象方法没有默认读取行为。
        """


class BMSReader(ABC):
    """四套设备BMS数据源的抽象接口。"""

    @abstractmethod
    def read(self) -> Mapping[str, DeviceBMSState]:
        """读取四个设备的BMS状态和实时功率边界。

        返回映射必须以标准设备常量为键，并为每个设备提供一个
        ``DeviceBMSState``。实际实现应保证四台设备数据尽量来自同一调度时刻，
        并正确填写 ``data_age_s``，由EMS决定过期数据是否可用。
        """


class ThermalPowerReader(ABC):
    """火电及其他非储能电源实时功率的抽象读取接口。"""

    @abstractmethod
    def read(self) -> ThermalPowerState:
        """读取与AGC同一计量点、同一符号方向的实时发电功率。"""


class PCSInterface(ABC):
    """EMS向四台PCS下发有功功率指令的抽象接口。"""

    @abstractmethod
    def send_power_setpoints(
        self, setpoints_W: Mapping[str, float], timestamp_s: float
    ) -> bool | PCSDispatchFeedback:
        """向四台PCS下发同一调度周期的有功功率指令。

        参数：
            setpoints_W: 设备名称到有功功率指令的映射，正值放电、负值充电。
            timestamp_s: 指令对应的AGC时间戳，可用于顺序检查和回执匹配。

        返回：
            四台PCS均成功接收并确认时返回True，否则返回False。

        现场实现通常还应包含通信超时、序号、防重放、回读校验和失败置零，
        这些硬件通信细节不属于当前EMS核心算法。
        """


@dataclass
class FlowBatteryPowerFeedback(DevicePowerFeedback):
    """液流电池一次功率执行结果及其统一并网计量点实际功率。"""

    requested_stack_power_W: float = 0.0
    device_output: Any = None

    @property
    def requested_grid_power_W(self) -> float:
        """兼容原接口的并网点请求功率字段。"""
        return self.requested_power_W

    @property
    def actual_grid_power_W(self) -> float:
        """兼容原接口的并网点实际功率字段。"""
        return self.actual_power_W


class FlowBatteryEMSAdapter(StorageDeviceAdapter):
    """在EMS并网功率接口和 ``FlowBatteryDevice`` 直流电堆接口之间转换。"""

    name = FLOW_BATTERY

    def __init__(
        self,
        device: Any,
        pcs_charge_efficiency: float = 1.0,
        pcs_discharge_efficiency: float = 1.0,
        auxiliary_power_W: float = 0.0,
    ) -> None:
        self.device = device
        self.pcs_charge_efficiency = float(pcs_charge_efficiency)
        self.pcs_discharge_efficiency = float(pcs_discharge_efficiency)
        self.auxiliary_power_W = float(auxiliary_power_W)
        if not 0.0 < self.pcs_charge_efficiency <= 1.0:
            raise ValueError("pcs_charge_efficiency must be in (0, 1].")
        if not 0.0 < self.pcs_discharge_efficiency <= 1.0:
            raise ValueError("pcs_discharge_efficiency must be in (0, 1].")
        if self.auxiliary_power_W < 0.0:
            raise ValueError("auxiliary_power_W cannot be negative.")
        self._cached_limits: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[Tuple[float, float, float]] = None

    def _key(self, dt_s: float, temperature_C: float) -> Tuple[float, float, float]:
        return (
            float(self.device.time_s),
            float(dt_s),
            float(temperature_C),
        )

    def read_state(
        self,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Any = None,
        data_age_s: float = 0.0,
        timestamp_s: Optional[float] = None,
    ) -> DeviceBMSState:
        """读取实时状态及边界，并统一换算为交流并网点功率。"""
        state = self.device.get_state(temperature_C)
        limits = self.device.get_power_limits(dt_s, temperature_C, hydraulic)
        self._cached_limits = dict(limits)
        self._cache_key = self._key(dt_s, temperature_C)
        result = DeviceBMSState.from_flow_battery(
            limits,
            state,
            self.device.config.soc_min,
            self.device.config.soc_max,
            data_age_s,
        )
        result.energy_capacity_Wh = max(
            1.0e-12,
            float(self.device.rated_capacity_Ah)
            * float(self.device.params.n_cells)
            * float(self.device.params.formal_potential)
            * float(self.device.config.soc_max - self.device.config.soc_min)
            * float(state.get("soh", 1.0)),
        )
        result.max_discharge_power_W = max(
            0.0,
            self.pcs_discharge_efficiency * float(limits["discharge_W"])
            - self.auxiliary_power_W,
        )
        charge_stack_magnitude_W = float(limits["charge_magnitude_W"])
        result.max_charge_power_W = (
            charge_stack_magnitude_W / self.pcs_charge_efficiency
            + self.auxiliary_power_W
            if charge_stack_magnitude_W > 0.0
            else 0.0
        )
        result.charge_efficiency = self.pcs_charge_efficiency
        result.discharge_efficiency = self.pcs_discharge_efficiency
        result.validate()
        return result

    def apply_power(
        self,
        grid_power_W: float,
        dt_s: float,
        temperature_C: float = 25.0,
        hydraulic: Any = None,
        timestamp_s: Optional[float] = None,
    ) -> FlowBatteryPowerFeedback:
        """执行并网点功率指令，并返回模型实际达到的并网点功率。"""
        grid_power_W = _finite(grid_power_W, "flow battery grid power")
        if grid_power_W > 0.0:
            stack_power_W = (
                grid_power_W + self.auxiliary_power_W
            ) / self.pcs_discharge_efficiency
        elif grid_power_W < 0.0:
            stack_power_W = min(
                0.0,
                (grid_power_W + self.auxiliary_power_W)
                * self.pcs_charge_efficiency,
            )
        else:
            stack_power_W = 0.0

        cached = (
            self._cached_limits
            if self._cache_key == self._key(dt_s, temperature_C)
            else None
        )
        output = self.device.step_power(
            stack_power_W,
            dt_s,
            temperature_C,
            hydraulic,
            power_limits=cached,
        )
        self._cached_limits = None
        self._cache_key = None
        if output.power_W > 0.0:
            actual_grid_W = (
                output.power_W * self.pcs_discharge_efficiency
                - self.auxiliary_power_W
            )
        elif output.power_W < 0.0:
            actual_grid_W = (
                output.power_W / self.pcs_charge_efficiency
                - self.auxiliary_power_W
            )
        else:
            actual_grid_W = 0.0
        return FlowBatteryPowerFeedback(
            name=self.name,
            requested_power_W=grid_power_W,
            actual_power_W=float(actual_grid_W),
            accepted=not bool(output.fault),
            requested_stack_power_W=stack_power_W,
            device_output=output,
            raw=output,
        )


class LithiumBatteryEMSAdapter(StorageDeviceAdapter):
    """把 ``LithiumBatteryDevice`` 接入统一EMS并网功率接口。"""

    name = LITHIUM

    def __init__(
        self,
        device: Any,
        temperature_C: float = 25.0,
        pcs_charge_efficiency: float = 1.0,
        pcs_discharge_efficiency: float = 1.0,
        auxiliary_power_W: float = 0.0,
    ) -> None:
        self.device = device
        self.temperature_C = float(temperature_C)
        self.pcs_charge_efficiency = float(pcs_charge_efficiency)
        self.pcs_discharge_efficiency = float(pcs_discharge_efficiency)
        self.auxiliary_power_W = float(auxiliary_power_W)
        if not 0.0 < self.pcs_charge_efficiency <= 1.0:
            raise ValueError("pcs_charge_efficiency must be in (0, 1].")
        if not 0.0 < self.pcs_discharge_efficiency <= 1.0:
            raise ValueError("pcs_discharge_efficiency must be in (0, 1].")
        if self.auxiliary_power_W < 0.0:
            raise ValueError("auxiliary_power_W cannot be negative.")
        self._cached_limits: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[Tuple[float, float, float]] = None

    def _key(self, dt_s: float) -> Tuple[float, float, float]:
        return (
            float(self.device.model.state()["t"]),
            float(dt_s),
            self.temperature_C,
        )

    def read_state(
        self, dt_s: float, timestamp_s: Optional[float] = None
    ) -> DeviceBMSState:
        """读取锂电BMS状态和边界，并换算到交流并网计量点。"""
        state = self.device.get_state(self.temperature_C)
        limits = self.device.get_power_limits(dt_s, self.temperature_C)
        self._cached_limits = dict(limits)
        self._cache_key = self._key(dt_s)
        temperature_factor = float(limits.get("temperature_factor", 1.0))
        discharge_W = max(
            0.0,
            self.pcs_discharge_efficiency * float(limits["discharge_W"])
            - self.auxiliary_power_W,
        )
        charge_stack_W = float(limits["charge_magnitude_W"])
        charge_W = (
            charge_stack_W / self.pcs_charge_efficiency + self.auxiliary_power_W
            if charge_stack_W > 0.0
            else 0.0
        )
        available = temperature_factor > 0.0 and (
            discharge_W > 0.0 or charge_W > 0.0
        )
        soh_factor = float(limits.get("soh_factor", 1.0))
        if temperature_factor <= 0.0:
            operating_state = "FAULT"
        elif not available:
            operating_state = "BLOCKED"
        elif temperature_factor < 1.0 or soh_factor < 1.0:
            operating_state = "DERATED"
        else:
            operating_state = "READY"
        result = DeviceBMSState(
            name=LITHIUM,
            soc=float(state["soc"]),
            max_discharge_power_W=discharge_W,
            max_charge_power_W=charge_W,
            available=available,
            fault=temperature_factor <= 0.0,
            data_age_s=0.0,
            soc_min=float(self.device.config.soc_min),
            soc_max=float(self.device.config.soc_max),
            operating_state=operating_state,
            soh=float(state.get("soh", 1.0)),
            charge_efficiency=self.pcs_charge_efficiency,
            discharge_efficiency=self.pcs_discharge_efficiency,
            temperature_factor=temperature_factor,
            energy_capacity_Wh=max(
                1.0e-12,
                sum(float(value) for value in self.device.model.params["Q"])
                / float(self.device.model.N_s)
                * float(state["stack_voltage_V"])
                * float(state.get("soh", 1.0)),
            ),
            soe=float(state["soc"]),
        )
        result.validate()
        return result

    def apply_power(
        self,
        power_W: float,
        dt_s: float,
        timestamp_s: Optional[float] = None,
    ) -> DevicePowerFeedback:
        """执行EMS并网点功率，并返回锂电池实际并网功率。"""
        requested_grid_W = _finite(power_W, "lithium grid power")
        if requested_grid_W > 0.0:
            stack_power_W = (
                requested_grid_W + self.auxiliary_power_W
            ) / self.pcs_discharge_efficiency
        elif requested_grid_W < 0.0:
            stack_power_W = min(
                0.0,
                (requested_grid_W + self.auxiliary_power_W)
                * self.pcs_charge_efficiency,
            )
        else:
            stack_power_W = 0.0

        cached = self._cached_limits if self._cache_key == self._key(dt_s) else None
        output = self.device.step_power(
            stack_power_W,
            dt_s,
            self.temperature_C,
            power_limits=cached,
        )
        self._cached_limits = None
        self._cache_key = None
        if output.power_W > 0.0:
            actual_grid_W = (
                output.power_W * self.pcs_discharge_efficiency
                - self.auxiliary_power_W
            )
        elif output.power_W < 0.0:
            actual_grid_W = (
                output.power_W / self.pcs_charge_efficiency
                - self.auxiliary_power_W
            )
        else:
            actual_grid_W = 0.0
        return DevicePowerFeedback(
            name=self.name,
            requested_power_W=requested_grid_W,
            actual_power_W=float(actual_grid_W),
            accepted=not bool(output.fault),
            raw=output,
        )


class SupercapacitorEMSAdapter(StorageDeviceAdapter):
    """把EDLC或混合超级电容的动态模型接入统一EMS接口。"""

    def __init__(
        self,
        device: Any,
        name: str,
        temperature_C: float = 25.0,
        pcs_charge_efficiency: float = 1.0,
        pcs_discharge_efficiency: float = 1.0,
        auxiliary_power_W: float = 0.0,
    ) -> None:
        if name not in (DOUBLE_LAYER_SUPERCAPACITOR, HYBRID_SUPERCAPACITOR):
            raise ValueError("超级电容适配器名称必须是EDLC或混合超级电容。")
        self.device = device
        self.name = name
        self.temperature_C = float(temperature_C)
        self.pcs_charge_efficiency = float(pcs_charge_efficiency)
        self.pcs_discharge_efficiency = float(pcs_discharge_efficiency)
        self.auxiliary_power_W = float(auxiliary_power_W)
        if not 0.0 < self.pcs_charge_efficiency <= 1.0:
            raise ValueError("pcs_charge_efficiency must be in (0, 1].")
        if not 0.0 < self.pcs_discharge_efficiency <= 1.0:
            raise ValueError("pcs_discharge_efficiency must be in (0, 1].")
        if self.auxiliary_power_W < 0.0:
            raise ValueError("auxiliary_power_W cannot be negative.")
        self._cached_limits: Optional[Dict[str, Any]] = None
        self._cache_key: Optional[Tuple[float, float, float]] = None

    def _key(self, dt_s: float) -> Tuple[float, float, float]:
        state = self.device.get_state(self.temperature_C)
        return float(state["time_s"]), float(dt_s), self.temperature_C

    def _energy_capacity_Wh(self) -> float:
        """由当前有效电容和电压窗口估算设备可用能量。"""
        if hasattr(self.device, "branch_cells"):
            cells = [
                cell
                for branch in self.device.branch_cells
                for cell in branch
            ]
        else:
            cells = [self.device]
        energy_Wh = 0.0
        for cell in cells:
            cell_state = cell.get_state(self.temperature_C)
            capacitance_F = float(cell_state["effective_capacitance_F"])
            energy_Wh += (
                0.5
                * capacitance_F
                * (float(cell.U_nom) ** 2 - float(cell.U_cutoff) ** 2)
                / 3600.0
            )
        return max(energy_Wh, 1.0e-12)

    def read_state(
        self, dt_s: float, timestamp_s: Optional[float] = None
    ) -> DeviceBMSState:
        """读取状态和动态功率边界，并换算到并网计量点。"""
        state = self.device.get_state(self.temperature_C)
        limits = self.device.get_power_limits(dt_s, self.temperature_C)
        self._cached_limits = dict(limits)
        self._cache_key = self._key(dt_s)
        temperature_factor = float(limits.get("temperature_factor", 1.0))
        soh_factor = float(limits.get("soh_factor", 1.0))
        discharge_W = max(
            0.0,
            float(limits["discharge_W"]) * self.pcs_discharge_efficiency
            - self.auxiliary_power_W,
        )
        charge_stack_W = float(limits["charge_magnitude_W"])
        charge_W = (
            charge_stack_W / self.pcs_charge_efficiency + self.auxiliary_power_W
            if charge_stack_W > 0.0
            else 0.0
        )
        available = temperature_factor > 0.0 and (discharge_W > 0.0 or charge_W > 0.0)
        if temperature_factor <= 0.0:
            operating_state = "FAULT"
        elif not available:
            operating_state = "BLOCKED"
        elif temperature_factor < 1.0 or soh_factor < 1.0:
            operating_state = "DERATED"
        else:
            operating_state = "READY"
        result = DeviceBMSState(
            name=self.name,
            soc=float(state["soc"]),
            max_discharge_power_W=discharge_W,
            max_charge_power_W=charge_W,
            available=available,
            fault=temperature_factor <= 0.0,
            data_age_s=0.0,
            soc_min=0.0,
            soc_max=1.0,
            operating_state=operating_state,
            soh=float(state.get("soh", 1.0)),
            charge_efficiency=self.pcs_charge_efficiency,
            discharge_efficiency=self.pcs_discharge_efficiency,
            temperature_factor=temperature_factor,
            energy_capacity_Wh=self._energy_capacity_Wh(),
            soe=float(state.get("soe", state["soc"])),
        )
        result.validate()
        return result

    def apply_power(
        self,
        power_W: float,
        dt_s: float,
        timestamp_s: Optional[float] = None,
    ) -> DevicePowerFeedback:
        """执行EMS并网功率，并返回设备实际达到的并网功率。"""
        requested_grid_W = _finite(power_W, "%s grid power" % self.name)
        if requested_grid_W > 0.0:
            device_power_W = (
                requested_grid_W + self.auxiliary_power_W
            ) / self.pcs_discharge_efficiency
        elif requested_grid_W < 0.0:
            device_power_W = min(
                0.0,
                (requested_grid_W + self.auxiliary_power_W)
                * self.pcs_charge_efficiency,
            )
        else:
            device_power_W = 0.0
        cached = self._cached_limits if self._cache_key == self._key(dt_s) else None
        output = self.device.step_power(
            device_power_W,
            dt_s,
            self.temperature_C,
            power_limits=cached,
        )
        self._cached_limits = None
        self._cache_key = None
        if output.power_W > 0.0:
            actual_grid_W = (
                output.power_W * self.pcs_discharge_efficiency
                - self.auxiliary_power_W
            )
        elif output.power_W < 0.0:
            actual_grid_W = (
                output.power_W / self.pcs_charge_efficiency
                - self.auxiliary_power_W
            )
        else:
            actual_grid_W = 0.0
        return DevicePowerFeedback(
            name=self.name,
            requested_power_W=requested_grid_W,
            actual_power_W=float(actual_grid_W),
            accepted=not bool(output.fault),
            raw=output,
        )


class EDLCEMSAdapter(SupercapacitorEMSAdapter):
    """双电层超级电容EMS适配器。"""

    def __init__(self, device: Any, **kwargs: Any) -> None:
        super().__init__(device, DOUBLE_LAYER_SUPERCAPACITOR, **kwargs)


class HybridSupercapacitorEMSAdapter(SupercapacitorEMSAdapter):
    """混合超级电容EMS适配器。"""

    def __init__(self, device: Any, **kwargs: Any) -> None:
        super().__init__(device, HYBRID_SUPERCAPACITOR, **kwargs)


class StaticStorageDeviceAdapter(StorageDeviceAdapter):
    """用于联调的静态设备适配器，不模拟SOC变化或设备内部动态。"""

    def __init__(self, state: DeviceBMSState) -> None:
        state.validate()
        self.name = state.name
        self.state = state

    def read_state(
        self, dt_s: float, timestamp_s: Optional[float] = None
    ) -> DeviceBMSState:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        return self.state

    def apply_power(
        self,
        power_W: float,
        dt_s: float,
        timestamp_s: Optional[float] = None,
    ) -> DevicePowerFeedback:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        requested = _finite(power_W, "%s power" % self.name)
        return DevicePowerFeedback(
            name=self.name,
            requested_power_W=requested,
            actual_power_W=requested,
            accepted=True,
        )


class ScaledStorageDeviceAdapter(StorageDeviceAdapter):
    """把一个代表性储能单元等比例聚合成电站级设备。

    ``unit_count`` 个单元被假定具有相同SOC、SOH、温度和功率边界。读取状态时，
    功率边界与能量容量乘以单元数量；执行功率时，场站指令先除以单元数量，
    只推进一次代表性物理模型，再把实际单元功率乘回场站功率。这样可以在不
    虚增单体电流、也不逐台计算成百上千个相同设备的前提下进行MW级仿真。

    该聚合器不模拟单元间SOC和老化差异。需要研究场站不一致性时，应建立多组
    具有不同初始状态的代表性单元，再分别聚合。
    """

    def __init__(
        self, unit_adapter: StorageDeviceAdapter, unit_count: int
    ) -> None:
        if not isinstance(unit_adapter, StorageDeviceAdapter):
            raise TypeError("unit_adapter must inherit StorageDeviceAdapter.")
        if isinstance(unit_count, bool) or int(unit_count) != unit_count:
            raise ValueError("unit_count must be a positive integer.")
        if int(unit_count) <= 0:
            raise ValueError("unit_count must be a positive integer.")
        self.unit_adapter = unit_adapter
        self.unit_count = int(unit_count)
        self.name = unit_adapter.name

    def read_state(
        self, dt_s: float, timestamp_s: Optional[float] = None
    ) -> DeviceBMSState:
        """读取代表性单元，并把功率边界和能量容量放大到场站规模。"""
        unit_state = self.unit_adapter.read_state(
            dt_s, timestamp_s=timestamp_s
        )
        scaled_energy = (
            None
            if unit_state.energy_capacity_Wh is None
            else unit_state.energy_capacity_Wh * self.unit_count
        )
        station_state = replace(
            unit_state,
            max_discharge_power_W=(
                unit_state.max_discharge_power_W * self.unit_count
            ),
            max_charge_power_W=(
                unit_state.max_charge_power_W * self.unit_count
            ),
            energy_capacity_Wh=scaled_energy,
        )
        station_state.validate()
        return station_state

    def apply_power(
        self,
        power_W: float,
        dt_s: float,
        timestamp_s: Optional[float] = None,
    ) -> DevicePowerFeedback:
        """把场站总功率均分给代表性单元，并返回聚合后的实际功率。"""
        station_request_W = _finite(power_W, "%s station power" % self.name)
        unit_feedback = self.unit_adapter.apply_power(
            station_request_W / self.unit_count,
            dt_s,
            timestamp_s=timestamp_s,
        )
        return DevicePowerFeedback(
            name=self.name,
            requested_power_W=station_request_W,
            actual_power_W=(
                unit_feedback.actual_power_W * self.unit_count
            ),
            accepted=unit_feedback.accepted,
            raw={
                "unit_count": self.unit_count,
                "unit_requested_power_W": (
                    station_request_W / self.unit_count
                ),
                "unit_feedback": unit_feedback,
            },
        )


class CallbackStorageDeviceAdapter(StorageDeviceAdapter):
    """通过两个回调函数快速接入锂电池或超级电容等外部模型。"""

    def __init__(
        self,
        name: str,
        state_reader: Callable[[float, Optional[float]], DeviceBMSState],
        power_executor: Callable[
            [float, float, Optional[float]], DevicePowerFeedback | float
        ],
    ) -> None:
        if name not in DEVICE_NAMES:
            raise ValueError("Unknown device name: %s" % name)
        self.name = name
        self._state_reader = state_reader
        self._power_executor = power_executor

    def read_state(
        self, dt_s: float, timestamp_s: Optional[float] = None
    ) -> DeviceBMSState:
        state = self._state_reader(float(dt_s), timestamp_s)
        if state.name != self.name:
            raise ValueError("Adapter name and DeviceBMSState.name do not match.")
        state.validate()
        return state

    def apply_power(
        self,
        power_W: float,
        dt_s: float,
        timestamp_s: Optional[float] = None,
    ) -> DevicePowerFeedback:
        requested = _finite(power_W, "%s power" % self.name)
        feedback = self._power_executor(requested, float(dt_s), timestamp_s)
        if isinstance(feedback, DevicePowerFeedback):
            if feedback.name != self.name:
                raise ValueError("Adapter name and feedback.name do not match.")
            return feedback
        return DevicePowerFeedback(
            name=self.name,
            requested_power_W=requested,
            actual_power_W=_finite(feedback, "%s actual power" % self.name),
            accepted=True,
        )


class HybridDeviceBridge(BMSReader, PCSInterface):
    """统一读取并执行四种储能设备适配器的EMS桥接器。"""

    def __init__(
        self, adapters: Mapping[str, StorageDeviceAdapter]
    ) -> None:
        if set(adapters) != set(DEVICE_NAMES):
            raise ValueError("Adapters are required for all four storage devices.")
        self.adapters = {name: adapters[name] for name in DEVICE_NAMES}
        for name, adapter in self.adapters.items():
            if not isinstance(adapter, StorageDeviceAdapter):
                raise TypeError("All adapters must inherit StorageDeviceAdapter.")
            if adapter.name != name:
                raise ValueError("Adapter mapping key and adapter.name do not match.")
        self._dt_s = 1.0
        self._timestamp_s: Optional[float] = None
        self.last_feedback: Dict[str, DevicePowerFeedback] = {}

    def read_for_cycle(
        self, dt_s: float, timestamp_s: float
    ) -> Mapping[str, DeviceBMSState]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        self._dt_s = float(dt_s)
        self._timestamp_s = float(timestamp_s)
        return {
            name: adapter.read_state(
                self._dt_s, timestamp_s=self._timestamp_s
            )
            for name, adapter in self.adapters.items()
        }

    def read(self) -> Mapping[str, DeviceBMSState]:
        return {
            name: adapter.read_state(
                self._dt_s, timestamp_s=self._timestamp_s
            )
            for name, adapter in self.adapters.items()
        }

    def send_power_setpoints(
        self, setpoints_W: Mapping[str, float], timestamp_s: float
    ) -> PCSDispatchFeedback:
        missing = set(DEVICE_NAMES) - set(setpoints_W)
        if missing:
            raise ValueError("Missing device setpoints: %s" % sorted(missing))
        feedback = {
            name: adapter.apply_power(
                float(setpoints_W[name]),
                self._dt_s,
                timestamp_s=float(timestamp_s),
            )
            for name, adapter in self.adapters.items()
        }
        self.last_feedback = feedback
        return PCSDispatchFeedback(
            accepted=all(item.accepted for item in feedback.values()),
            actual_power_W={
                name: item.actual_power_W for name, item in feedback.items()
            },
        )


class FlowBatteryHybridBridge(BMSReader, PCSInterface):
    """把一台动态液流电池与其余储能状态接入 ``run_cycle`` 的仿真桥。"""

    def __init__(
        self,
        flow_adapter: FlowBatteryEMSAdapter,
        other_states: Mapping[str, DeviceBMSState],
        temperature_C: float = 25.0,
        hydraulic: Any = None,
        other_pcs: Optional[PCSInterface] = None,
    ) -> None:
        self.flow_adapter = flow_adapter
        self.other_states = dict(other_states)
        self.temperature_C = float(temperature_C)
        self.hydraulic = hydraulic
        self.other_pcs = other_pcs
        self._dt_s = 1.0
        self.last_flow_feedback: Optional[FlowBatteryPowerFeedback] = None

    def read_for_cycle(
        self, dt_s: float, timestamp_s: float
    ) -> Mapping[str, DeviceBMSState]:
        self._dt_s = float(dt_s)
        states = dict(self.other_states)
        states[FLOW_BATTERY] = self.flow_adapter.read_state(
            dt_s, self.temperature_C, self.hydraulic
        )
        return states

    def read(self) -> Mapping[str, DeviceBMSState]:
        return self.read_for_cycle(self._dt_s, 0.0)

    def send_power_setpoints(
        self, setpoints_W: Mapping[str, float], timestamp_s: float
    ) -> PCSDispatchFeedback:
        accepted = True
        actual = {name: float(setpoints_W[name]) for name in DEVICE_NAMES}
        if self.other_pcs is not None:
            downstream = self.other_pcs.send_power_setpoints(setpoints_W, timestamp_s)
            if isinstance(downstream, PCSDispatchFeedback):
                accepted = downstream.accepted
                if downstream.actual_power_W is not None:
                    actual.update(downstream.actual_power_W)
            else:
                accepted = bool(downstream)
        self.last_flow_feedback = self.flow_adapter.apply_power(
            setpoints_W[FLOW_BATTERY],
            self._dt_s,
            self.temperature_C,
            self.hydraulic,
        )
        actual[FLOW_BATTERY] = self.last_flow_feedback.actual_grid_power_W
        return PCSDispatchFeedback(accepted=accepted, actual_power_W=actual)


class HybridStorageEMS:
    """四类混合储能的实时有功功率分配控制器。

    储能不直接承担完整AGC，而只补偿火电相对基线参考的跟踪偏差。一级分频
    区分电池组和超级电容，二级分频进一步使液流电池承担慢分量、锂电池承担
    中频残差。SOC/SOE恢复是独立慢速计划，并通过火电参考功率协调能量来源。
    """

    def __init__(self, config: Optional[EMSConfig] = None) -> None:
        """创建EMS控制器并初始化有状态低通滤波器。

        参数：
            config: 可选EMS配置。为None时创建一套默认演示参数。

        初始化过程会完整校验配置；非法时间常数、缺失设备策略等问题会在系统
        启动时立即抛出，而不是等到第一次调度才暴露。滤波器状态初始为未初始化。
        """

        self.config = config if config is not None else EMSConfig()
        self.config.validate()
        self._power_gap_filter = FirstOrderLowPass(
            self.config.frequency_split_tau_s
        )
        self._aorfb_slow_filter = FirstOrderLowPass(
            self.config.low_group_split_tau_s
        )
        self._last_agc_timestamp_s: Optional[float] = None

    def reset(self, tracking_power_W: Optional[float] = None) -> None:
        """重置EMS中的低频功率记忆。

        参数：
            tracking_power_W: 若提供，则直接作为上一时刻低频状态，实现控制器重启、
                手自动切换时的无扰衔接；若为None，则清除初始化标志，下一个
                调度周期按 ``bumpless_filter_initialization`` 决定首样本行为。

        本函数不会清除BMS状态，因为BMS状态由每次 ``dispatch`` 的参数传入。
        """

        self._power_gap_filter.reset(tracking_power_W)
        self._aorfb_slow_filter.reset(tracking_power_W)
        self._last_agc_timestamp_s = None

    def _resolve_dt(self, timestamp_s: float, supplied_dt_s: float) -> float:
        """优先用连续AGC时间戳计算真实步长，并拒绝重复或倒序报文。"""
        if supplied_dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self._last_agc_timestamp_s is None or not self.config.use_agc_timestamp_delta:
            return float(supplied_dt_s)
        timestamp_dt_s = float(timestamp_s) - self._last_agc_timestamp_s
        if timestamp_dt_s <= 0.0:
            raise ValueError("AGC timestamps must be strictly increasing.")
        return timestamp_dt_s

    def run_cycle(
        self,
        agc_reader: AGCReader,
        bms_reader: BMSReader,
        pcs: PCSInterface,
        dt_s: float,
        thermal_reader: Optional[ThermalPowerReader] = None,
    ) -> DispatchResult:
        """执行一次“读取—计算—下发”的完整在线调度周期。

        参数：
            agc_reader: AGC读取接口的具体实现。
            bms_reader: 四类储能BMS读取接口的具体实现。
            pcs: PCS下发接口的具体实现。
            dt_s: 本周期实际时间步长，单位秒。
            thermal_reader: 火电功率读取接口。传入后启用火储联合场站目标模式；
                不传时保留旧版“AGC就是储能净指令”的兼容模式。

        返回：
            ``DispatchResult``，其中 ``pcs_accepted`` 记录下发结果。

        调用顺序固定为先读AGC、再读BMS、最后调用 ``dispatch``。本函数不捕获
        通信异常，工程应用可在更外层调度循环中执行重试、告警和安全置零。
        """

        agc = agc_reader.read()
        effective_dt_s = self._resolve_dt(agc.timestamp_s, dt_s)
        read_for_cycle = getattr(bms_reader, "read_for_cycle", None)
        bms_states = (
            read_for_cycle(effective_dt_s, agc.timestamp_s)
            if callable(read_for_cycle)
            else bms_reader.read()
        )
        if thermal_reader is None:
            return self.dispatch(agc, bms_states, effective_dt_s, pcs)
        thermal = thermal_reader.read()
        return self.dispatch(
            agc,
            bms_states,
            effective_dt_s,
            pcs,
            thermal=thermal,
        )

    def run_plant_cycle(
        self,
        agc_reader: AGCReader,
        thermal_reader: ThermalPowerReader,
        bms_reader: BMSReader,
        pcs: PCSInterface,
        dt_s: float,
    ) -> DispatchResult:
        """按“AGC—火电实发—BMS—PCS”顺序执行一次火储联合周期。"""
        return self.run_cycle(
            agc_reader,
            bms_reader,
            pcs,
            dt_s,
            thermal_reader=thermal_reader,
        )

    def dispatch(
        self,
        agc: AGCCommand,
        bms_states: Mapping[str, DeviceBMSState],
        dt_s: float,
        pcs: Optional[PCSInterface] = None,
        thermal: Optional[ThermalPowerState] = None,
    ) -> DispatchResult:
        """根据场站AGC、可选火电实发和四套BMS边界完成一次功率分配。

        参数：
            agc: 场站并网点总有功目标；兼容模式下是储能净功率目标。
            bms_states: 四台设备的SOC、SOH、效率、温度裕度、寿命代价和实时
                充放电功率边界。
            dt_s: 与上一调度周期的时间间隔，单位秒，必须大于0。
            pcs: 可选PCS接口；不传时只计算，适用于离线仿真和单元测试。
            thermal: 火电实发与其他电源功率。传入时启用火储联合模式；不传时
                自动构造零火电状态，从而兼容旧版储能独立AGC调用。

        处理过程：
            1. 根据BMS边界生成慢速SOC/SOE恢复计划；
            2. 用恢复计划修正火电参考值，再计算火电跟踪偏差；
            3. 一级分频区分电池与超容，二级分频区分液流与锂电；
            4. 高频、锂电中频、液流慢分量分别进行有界分配；
            5. 按时间尺度优先级转移各分量缺口；
            6. 形成场站级跟踪残差并可选地下发PCS指令。

        返回：
            ``DispatchResult``。火储联合模式下
            ``tracking_residual_W = AGC - 火电实发 - 其他实发 - 储能实发``。

        安全性：
            该算法只在有效BMS边界内分配，不会为了完全跟踪AGC而突破边界。
        """

        agc.validate()
        dt_s = self._resolve_dt(agc.timestamp_s, dt_s)
        legacy_mode = thermal is None
        if thermal is None:
            thermal = ThermalPowerState(0.0, agc.timestamp_s)
        thermal.validate(self.config.max_thermal_data_age_s)
        states = self._validate_bms_states(bms_states)

        # 先形成当前周期有效BMS边界；恢复计划和动态跟踪都必须服从它。
        effective_limits: Dict[str, Tuple[float, float]] = {}
        reasons: Dict[str, List[str]] = {}
        for name in DEVICE_NAMES:
            charge_limit, discharge_limit, device_reasons = self._effective_limits(
                states[name], self.config.device_policies[name]
            )
            effective_limits[name] = (charge_limit, discharge_limit)
            reasons[name] = device_reasons

        recovery, recovery_reasons = self._soc_recovery_schedule(
            states, effective_limits
        )
        for name, device_reasons in recovery_reasons.items():
            reasons[name].extend(device_reasons)
        recovery_total = sum(recovery.values())

        # 恢复充电时火电参考提高，恢复放电时火电参考降低。储能动态分量只补偿
        # 火电相对该参考值的偏差，因此恢复计划不会凭空改变场站总功率。
        thermal_reference = agc.power_W - thermal.other_power_W - recovery_total
        raw_tracking_error = thermal_reference - thermal.actual_power_W
        tracking_request = self._apply_tracking_deadband(raw_tracking_error)

        frequency_components = self._split_frequency(tracking_request, dt_s)
        low_components = self._split_low_group(
            frequency_components[LOW_FREQUENCY], dt_s
        )

        # 恢复计划是基线；按“超容高频—锂电中频—液流慢分量”依次叠加。
        current = dict(recovery)
        high_result, high_added, high_residual, high_weights = self._allocate_bounded(
            initial=current,
            limits=effective_limits,
            target_delta_W=frequency_components[HIGH_FREQUENCY],
            states=states,
            eligible_names=HIGH_FREQUENCY_GROUP,
        )
        current = high_result
        lithium_result, lithium_added, lithium_residual, lithium_weights = self._allocate_bounded(
            initial=current,
            limits=effective_limits,
            target_delta_W=low_components[LITHIUM_MEDIUM],
            states=states,
            eligible_names=(LITHIUM,),
        )
        current = lithium_result
        aorfb_result, aorfb_added, aorfb_residual, aorfb_weights = self._allocate_bounded(
            initial=current,
            limits=effective_limits,
            target_delta_W=low_components[AORFB_SLOW],
            states=states,
            eligible_names=(FLOW_BATTERY,),
        )
        primary = aorfb_result

        group_setpoints = {
            LOW_FREQUENCY: {
                FLOW_BATTERY: aorfb_added[FLOW_BATTERY],
                LITHIUM: lithium_added[LITHIUM],
            },
            HIGH_FREQUENCY: {
                name: high_added[name] for name in HIGH_FREQUENCY_GROUP
            },
        }

        primary_residuals = {
            AORFB_SLOW: aorfb_residual,
            LITHIUM_MEDIUM: lithium_residual,
            HIGH_FREQUENCY: high_residual,
        }
        redistribution_reasons: List[str] = []
        if abs(aorfb_residual) > self.config.tracking_tolerance_W:
            redistribution_reasons.append("flow_battery_slow_component_shortfall")
            self._mark_group_boundary_reasons(
                reasons, primary, effective_limits, (FLOW_BATTERY,), aorfb_residual
            )
        if abs(lithium_residual) > self.config.tracking_tolerance_W:
            redistribution_reasons.append("lithium_medium_component_shortfall")
            self._mark_group_boundary_reasons(
                reasons, primary, effective_limits, (LITHIUM,), lithium_residual
            )
        if abs(high_residual) > self.config.tracking_tolerance_W:
            redistribution_reasons.append("high_frequency_group_shortfall")
            self._mark_group_boundary_reasons(
                reasons,
                primary,
                effective_limits,
                HIGH_FREQUENCY_GROUP,
                high_residual,
            )

        # 高频不足先由锂电补；默认不让液流接管快速功率，除非显式开启紧急支撑。
        final = dict(primary)
        redistribution = {name: 0.0 for name in DEVICE_NAMES}
        cross_weights: Dict[str, float] = {}
        high_stages: List[Sequence[str]] = [(LITHIUM,)]
        if self.config.allow_aorfb_fast_emergency_support:
            high_stages.append((FLOW_BATTERY,))
        final, added, high_unmet, scores = self._allocate_in_priority_stages(
            final, effective_limits, high_residual, states, high_stages
        )
        self._accumulate_power(redistribution, added)
        cross_weights.update(scores)

        # 锂电中频不足时先调用两类超容，液流只在紧急快速支撑开启时兜底。
        medium_stages: List[Sequence[str]] = [HIGH_FREQUENCY_GROUP]
        if self.config.allow_aorfb_fast_emergency_support:
            medium_stages.append((FLOW_BATTERY,))
        final, added, medium_unmet, scores = self._allocate_in_priority_stages(
            final, effective_limits, lithium_residual, states, medium_stages
        )
        self._accumulate_power(redistribution, added)
        cross_weights.update(scores)

        # 液流慢分量不足时，依次使用锂电和超容剩余能力。
        final, added, slow_unmet, scores = self._allocate_in_priority_stages(
            final,
            effective_limits,
            aorfb_residual,
            states,
            [(LITHIUM,), HIGH_FREQUENCY_GROUP],
        )
        self._accumulate_power(redistribution, added)
        cross_weights.update(scores)

        if any(
            abs(value) > self.config.tracking_tolerance_W
            for value in redistribution.values()
        ):
            redistribution_reasons.append("adjacent_group_redistribution")

        # 旧版独立储能调用且未提供能量容量时，保留零和SOC内部平衡作为兼容。
        legacy_corrections = {name: 0.0 for name in DEVICE_NAMES}
        if legacy_mode and all(
            state.energy_capacity_Wh is None for state in states.values()
        ):
            final, legacy_corrections = self._apply_soc_balancing(
                final, effective_limits, states
            )

        storage_reference = recovery_total + tracking_request
        storage_residual = storage_reference - sum(final.values())
        plant_residual = agc.power_W - (
            thermal.actual_power_W + thermal.other_power_W + sum(final.values())
        )
        if abs(storage_residual) > self.config.tracking_tolerance_W:
            redistribution_reasons.append("unserved_tracking_request")
            if storage_reference > 0.0:
                total_direction_limit = sum(limit[1] for limit in effective_limits.values())
            else:
                total_direction_limit = sum(limit[0] for limit in effective_limits.values())
            if abs(storage_reference) > total_direction_limit:
                redistribution_reasons.append("total_capability_shortage")
            else:
                redistribution_reasons.append("time_scale_restriction")

        corrections = {
            name: recovery[name] + legacy_corrections[name]
            for name in DEVICE_NAMES
        }

        result = DispatchResult(
            timestamp_s=float(agc.timestamp_s),
            dt_s=float(dt_s),
            agc_requested_W=float(agc.power_W),
            thermal_actual_W=float(thermal.actual_power_W),
            other_power_W=float(thermal.other_power_W),
            thermal_reference_W=float(thermal_reference),
            raw_thermal_tracking_error_W=float(raw_tracking_error),
            storage_tracking_request_W=float(tracking_request),
            storage_reference_W=float(storage_reference),
            soc_recovery_requested_W=dict(recovery),
            soc_recovery_total_W=float(recovery_total),
            frequency_components_W=frequency_components,
            low_group_components_W=low_components,
            group_setpoints_W=group_setpoints,
            allocation_weights={
                AORFB_SLOW: aorfb_weights,
                LITHIUM_MEDIUM: lithium_weights,
                HIGH_FREQUENCY: high_weights,
                "cross_group": cross_weights,
            },
            group_residuals_W=primary_residuals,
            decomposed_power_W=primary,
            soc_correction_W=corrections,
            soc_corrected_power_W=final,
            bms_limited_power_W=primary,
            bms_limits_W={
                name: {
                    "max_charge_magnitude_W": effective_limits[name][0],
                    "max_discharge_W": effective_limits[name][1],
                }
                for name in DEVICE_NAMES
            },
            limit_reasons=reasons,
            gap_before_redistribution_W=sum(primary_residuals.values()),
            redistribution_W=redistribution,
            final_setpoints_W=final,
            tracking_residual_W=plant_residual,
            agc_fully_met=(
                abs(plant_residual)
                <= max(
                    self.config.tracking_tolerance_W,
                    self.config.thermal_tracking_deadband_W,
                )
            ),
            redistribution_reasons=redistribution_reasons,
        )

        # 第八步：下发到 PCS；不传 PCS 时可用于离线仿真和单元测试。
        if pcs is not None:
            feedback = pcs.send_power_setpoints(final, float(agc.timestamp_s))
            if isinstance(feedback, PCSDispatchFeedback):
                result.pcs_accepted = bool(feedback.accepted)
                if feedback.actual_power_W is not None:
                    result.actual_power_W = dict(feedback.actual_power_W)
                    result.actual_tracking_residual_W = agc.power_W - (
                        thermal.actual_power_W
                        + thermal.other_power_W
                        + sum(result.actual_power_W.values())
                    )
            else:
                result.pcs_accepted = bool(feedback)
        self._last_agc_timestamp_s = float(agc.timestamp_s)
        return result

    def dispatch_plant(
        self,
        agc: AGCCommand,
        thermal: ThermalPowerState,
        bms_states: Mapping[str, DeviceBMSState],
        dt_s: float,
        pcs: Optional[PCSInterface] = None,
    ) -> DispatchResult:
        """显式的火储联合入口，参数顺序与场站控制流程一致。"""
        return self.dispatch(agc, bms_states, dt_s, pcs, thermal=thermal)

    def _validate_bms_states(
        self, bms_states: Mapping[str, DeviceBMSState]
    ) -> Dict[str, DeviceBMSState]:
        """检查并按固定设备顺序整理四套BMS输入。

        参数：
            bms_states: 设备名称到BMS状态的映射。

        返回：
            新的普通字典，按 ``DEVICE_NAMES`` 顺序包含四台设备。固定顺序可使
            日志、测试和后续遍历结果保持一致。

        校验规则：
            四个标准设备键必须全部存在；映射键必须与对象内部 ``state.name``
            一致；每个状态还要通过 ``DeviceBMSState.validate``。额外的未知键
            当前会被忽略，但不会进入任何调度计算。
        """

        missing = set(DEVICE_NAMES) - set(bms_states)
        if missing:
            raise ValueError("Missing BMS states: %s" % sorted(missing))
        states = {name: bms_states[name] for name in DEVICE_NAMES}
        for name, state in states.items():
            if state.name != name:
                raise ValueError("BMS mapping key and state.name do not match: %s" % name)
            state.validate()
        return states

    def _split_frequency(self, power_gap_W: float, dt_s: float) -> Dict[str, float]:
        """把净功率缺口分解成低频和高频两个互补分量。

        参数：
            power_gap_W: 待分配的净功率，通常就是当前AGC指令，正放电负充电。
            dt_s: 实际调度时间步长，单位秒。

        计算：
            ``P_low(k)=alpha*P_low(k-1)+(1-alpha)*P_gap(k)``，
            ``alpha=exp(-dt/tau)``；随后令
            ``P_high(k)=P_gap(k)-P_low(k)``。

        返回：
            包含 ``low_frequency`` 和 ``high_frequency`` 的字典。两者之和在
            浮点精度范围内严格等于输入功率，低频交给电池组、高频交给超容组。

        状态变化：
            调用会更新内部低通滤波器，不能对同一周期重复调用后再丢弃结果。
        """

        low = self._power_gap_filter.update(
            power_gap_W, dt_s, self.config.bumpless_filter_initialization
        )
        return {
            LOW_FREQUENCY: low,
            HIGH_FREQUENCY: power_gap_W - low,
        }

    def _apply_tracking_deadband(self, error_W: float) -> float:
        """对火电跟踪偏差应用连续死区，避免储能追逐测量噪声。"""
        deadband = self.config.thermal_tracking_deadband_W
        if abs(error_W) <= deadband:
            return 0.0
        return math.copysign(abs(error_W) - deadband, error_W)

    def _split_low_group(self, low_power_W: float, dt_s: float) -> Dict[str, float]:
        """把电池低频分量进一步分为液流慢分量和锂电中频残差。"""
        aorfb_slow = self._aorfb_slow_filter.update(
            low_power_W, dt_s, self.config.bumpless_filter_initialization
        )
        return {
            AORFB_SLOW: aorfb_slow,
            LITHIUM_MEDIUM: low_power_W - aorfb_slow,
        }

    def _soc_recovery_schedule(
        self,
        states: Mapping[str, DeviceBMSState],
        limits: Mapping[str, Tuple[float, float]],
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """依据SOC/SOE能量偏差生成受BMS边界约束的慢速恢复计划。"""
        recovery = {name: 0.0 for name in DEVICE_NAMES}
        reasons = {name: [] for name in DEVICE_NAMES}
        tolerance = self.config.tracking_tolerance_W
        for name in DEVICE_NAMES:
            state = states[name]
            policy = self.config.device_policies[name]
            energy_state = state.soe if state.soe is not None else state.soc
            error = energy_state - policy.soc_target
            if abs(error) <= policy.soc_recovery_deadband:
                continue
            if state.energy_capacity_Wh is None:
                reasons[name].append("soc_recovery_energy_unavailable")
                continue
            effective_error = math.copysign(
                abs(error) - policy.soc_recovery_deadband, error
            )
            energy_error_Wh = state.energy_capacity_Wh * effective_error
            if energy_error_Wh > 0.0:
                raw_power = (
                    state.discharge_efficiency
                    * energy_error_Wh
                    * 3600.0
                    / policy.soc_recovery_time_s
                )
            else:
                raw_power = (
                    energy_error_Wh
                    * 3600.0
                    / (state.charge_efficiency * policy.soc_recovery_time_s)
                )
            recovery_cap = (
                policy.rated_power_W * policy.soc_recovery_power_fraction
            )
            charge_limit, discharge_limit = limits[name]
            bounded = _clip(
                raw_power,
                -min(charge_limit, recovery_cap),
                min(discharge_limit, recovery_cap),
            )
            recovery[name] = bounded
            if abs(bounded - raw_power) > tolerance:
                reasons[name].append("soc_recovery_power_limited")
        return recovery, reasons

    def _effective_limits(
        self, state: DeviceBMSState, policy: DevicePolicy
    ) -> Tuple[float, float, List[str]]:
        """根据设备状态形成当前周期真正可用的充放电功率边界。

        参数：
            state: BMS上报的实时状态和原始功率边界。
            policy: 对应设备的EMS默认SOC工作区间。

        返回：
            三元组 ``(charge_limit, discharge_limit, reasons)``。前两项都是非负
            功率幅值；``reasons`` 记录导致能力关闭或削减的原因代码。

        规则：
            - 故障、不可用或BMS数据过期时，充放电能力都置0；
            - SOC到达下限时只禁止放电，仍允许充电恢复；
            - SOC到达上限时只禁止充电，仍允许放电恢复；
            - BMS提供 ``soc_min/soc_max`` 时优先于EMS默认配置。

        注意：
            电压、温度、流量、SOH等物理约束应已体现在BMS上报的最大功率中；
            EMS在这里不重复建立设备物理模型。
        """

        charge = state.max_charge_power_W
        discharge = state.max_discharge_power_W
        reasons: List[str] = []

        if state.fault:
            reasons.append("bms_fault")
        if not state.available:
            reasons.append("device_unavailable")
        if state.data_age_s > self.config.max_bms_data_age_s:
            reasons.append("bms_data_stale")
        if reasons:
            return 0.0, 0.0, reasons

        if charge > policy.rated_power_W:
            charge = policy.rated_power_W
            reasons.append("charge_capped_by_rated_power")
        if discharge > policy.rated_power_W:
            discharge = policy.rated_power_W
            reasons.append("discharge_capped_by_rated_power")

        soc_min = policy.soc_min if state.soc_min is None else state.soc_min
        soc_max = policy.soc_max if state.soc_max is None else state.soc_max
        if state.soc <= soc_min:
            discharge = 0.0
            reasons.append("soc_at_or_below_min")
        if state.soc >= soc_max:
            charge = 0.0
            reasons.append("soc_at_or_above_max")
        return charge, discharge, reasons

    def _apply_soc_balancing(
        self,
        setpoints: Mapping[str, float],
        limits: Mapping[str, Tuple[float, float]],
        states: Mapping[str, DeviceBMSState],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """在不改变总AGC功率的前提下进行慢速SOC回中。

        高于目标SOC的设备作为放电侧，低于目标SOC的设备作为充电侧。
        只有两侧同时存在时才形成内部能量转移，避免凭空改变并网总功率。
        """
        corrected = {name: float(setpoints[name]) for name in DEVICE_NAMES}
        corrections = {name: 0.0 for name in DEVICE_NAMES}
        gain = self.config.soc_balancing_gain_W_per_soc
        maximum = self.config.soc_balancing_max_power_W
        deadband = self.config.soc_balancing_deadband
        if gain <= 0.0 or maximum <= 0.0:
            return corrected, corrections

        donor_requests: Dict[str, float] = {}
        receiver_requests: Dict[str, float] = {}
        for name in DEVICE_NAMES:
            state = states[name]
            if state.fault or not state.available:
                continue
            target = self.config.device_policies[name].soc_target
            error = state.soc - target
            charge_limit, discharge_limit = limits[name]
            if error > deadband:
                headroom = max(0.0, discharge_limit - corrected[name])
                donor_requests[name] = min(
                    gain * (error - deadband), maximum, headroom
                )
            elif error < -deadband:
                headroom = max(0.0, corrected[name] + charge_limit)
                receiver_requests[name] = min(
                    gain * (-error - deadband), maximum, headroom
                )

        donor_total = sum(donor_requests.values())
        receiver_total = sum(receiver_requests.values())
        transfer_W = min(donor_total, receiver_total)
        if transfer_W <= self.config.tracking_tolerance_W:
            return corrected, corrections

        for name, request in donor_requests.items():
            delta = transfer_W * request / donor_total
            corrections[name] = delta
            corrected[name] += delta
        for name, request in receiver_requests.items():
            delta = -transfer_W * request / receiver_total
            corrections[name] = delta
            corrected[name] += delta
        return corrected, corrections

    def _soc_margin(
        self,
        state: DeviceBMSState,
        policy: DevicePolicy,
        discharge: bool,
    ) -> float:
        """计算当前功率方向上的归一化SOC可用裕度。

        参数：
            state: 当前设备SOC及可选的实时SOC上下限。
            policy: 当BMS未提供实时边界时使用的默认SOC上下限。
            discharge: True表示计算放电裕度，False表示计算充电裕度。

        公式：
            放电：``(SOC-SOC_min)/(SOC_max-SOC_min)``；
            充电：``(SOC_max-SOC)/(SOC_max-SOC_min)``。
            结果最终限制在0～1。

        返回：
            方向SOC裕度。放电时SOC越高越接近1；充电时SOC越低越接近1。

        这里衡量的是到安全边界的距离，不是到 ``soc_target`` 的距离。因此它
        实现“靠近边界时少承担功率”，但不单独保证SOC主动回到目标值。
        """

        soc_min = policy.soc_min if state.soc_min is None else state.soc_min
        soc_max = policy.soc_max if state.soc_max is None else state.soc_max
        usable_range = max(soc_max - soc_min, 1.0e-12)
        if discharge:
            margin = (state.soc - soc_min) / usable_range
        else:
            margin = (soc_max - state.soc) / usable_range
        return _clip(margin, 0.0, 1.0)

    def _soc_target_factor(
        self,
        state: DeviceBMSState,
        policy: DevicePolicy,
        discharge: bool,
    ) -> float:
        """高于目标时偏好放电、低于目标时偏好充电，且保持连续正权重。"""
        energy_state = state.soe if state.soe is not None else state.soc
        usable_range = max(policy.soc_max - policy.soc_min, 1.0e-12)
        normalized_error = (energy_state - policy.soc_target) / usable_range
        direction_error = normalized_error if discharge else -normalized_error
        raw_factor = 1.0 + policy.soc_target_weight_gain * direction_error
        return _clip(
            raw_factor,
            policy.soc_target_factor_min,
            policy.soc_target_factor_max,
        )

    def _weighted_capabilities(
        self,
        setpoints: Mapping[str, float],
        limits: Mapping[str, Tuple[float, float]],
        remaining_W: float,
        states: Mapping[str, DeviceBMSState],
        eligible_names: Sequence[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算候选设备的方向剩余能力和动态分配分数。

        参数：
            setpoints: 当前迭代中每台设备已经承担的有符号功率。
            limits: 每台设备的 ``(充电幅值上限, 放电上限)``。
            remaining_W: 本轮仍待分配的有符号功率；正值触发放电计算，负值
                触发充电计算。
            states: 四台设备的BMS状态。
            eligible_names: 本轮允许参与的设备集合。组内分配时只传相应组，
                跨组补偿时传全部设备。

        剩余能力：
            放电 ``C_i=max(0, P_dis_max-P_i)``；
            充电 ``C_i=max(0, P_i+P_chg_max)``。充电边界以正幅值存储，而
            实际充电指令为负，因此采用后一表达式。

        动态分数：
            ``score_i = C_i * k_i * M_soc^g_soc * eta^g_eta``
            ``          * SOH^g_soh * f_T^g_T / (1+k_life*L_i)``。
            分数只决定相对分配比例，并不是直接下发的功率。

        返回：
            ``(capacities, scores)``。``capacities``保留所有候选设备的剩余
            能力；``scores``只包含能力大于容差且最终分数大于0的设备。

        工程含义：
            优先使用剩余功率大、SOC裕度大、效率高、SOH好、温度状态好且
            边际寿命代价低的设备。各指数为0时可关闭对应因素。
        """

        discharge = remaining_W > 0.0
        capacities: Dict[str, float] = {}
        scores: Dict[str, float] = {}
        tolerance = self.config.tracking_tolerance_W

        for name in eligible_names:
            charge_limit, discharge_limit = limits[name]
            if discharge:
                capability = max(0.0, discharge_limit - setpoints[name])
            else:
                capability = max(0.0, setpoints[name] + charge_limit)
            capacities[name] = capability
            if capability <= tolerance:
                continue

            state = states[name]
            policy = self.config.device_policies[name]
            soc_factor = self._soc_margin(state, policy, discharge) ** (
                policy.soc_weight_exponent
            )
            soc_target_factor = self._soc_target_factor(
                state, policy, discharge
            )
            efficiency = (
                state.discharge_efficiency
                if discharge
                else state.charge_efficiency
            )
            efficiency_factor = efficiency ** policy.efficiency_weight_exponent
            soh_factor = state.soh ** policy.soh_weight_exponent
            temperature_factor = state.temperature_factor ** (
                policy.temperature_weight_exponent
            )
            life_factor = 1.0 / (
                1.0 + policy.life_cost_weight * state.marginal_life_cost
            )
            score = (
                capability
                * policy.redistribution_weight
                * soc_factor
                * soc_target_factor
                * efficiency_factor
                * soh_factor
                * temperature_factor
                * life_factor
            )
            if score > 0.0:
                scores[name] = score
        return capacities, scores

    def _allocate_bounded(
        self,
        initial: Mapping[str, float],
        limits: Mapping[str, Tuple[float, float]],
        target_delta_W: float,
        states: Mapping[str, DeviceBMSState],
        eligible_names: Sequence[str],
    ) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, float]]:
        """按动态权重迭代分配有符号功率，并保证不突破BMS边界。

        参数：
            initial: 本轮开始前四台设备已有的功率。组内初次分配通常全为0，
                跨组补偿时则是低频组和高频组已经得到的初次结果。
            limits: 每台设备的 ``(充电幅值上限, 放电上限)``。
            target_delta_W: 希望在 ``initial`` 基础上新增的总功率；正值表示
                增加放电，负值表示增加充电。
            states: 用于动态权重计算的BMS状态。
            eligible_names: 本轮允许参与分配的设备。

        算法：
            1. 根据当前剩余功率重新计算各设备能力和score；
            2. 按 ``score_i/sum(score)`` 计算本轮应承担份额；
            3. 单台设备的实际移动量取“请求份额”和“剩余能力”的较小值；
            4. 扣除已完成分配后继续迭代，使被限幅设备留下的缺口转移给其他
               仍有能力的设备；
            5. 最后再次执行硬限幅，消除浮点误差并计算残差。

        返回：
            ``final``：包含初始值和本轮增量的最终功率；
            ``added``：本轮每台设备实际增加的有符号功率；
            ``residual``：``target_delta_W-sum(added)``，非零表示能力不足；
            ``first_scores``：第一次迭代的score，用于结果审计和策略分析。

        迭代次数最多为候选设备数加1。每轮至少应有设备达到边界或明显减小
        缺口；若分数全为0或分配不再推进，则提前停止，绝不强行越界。
        """

        final = {name: float(initial[name]) for name in DEVICE_NAMES}
        added = {name: 0.0 for name in DEVICE_NAMES}
        remaining = float(target_delta_W)
        tolerance = self.config.tracking_tolerance_W
        first_scores: Dict[str, float] = {}

        for iteration in range(len(eligible_names) + 1):
            if abs(remaining) <= tolerance:
                break
            capacities, scores = self._weighted_capabilities(
                final,
                limits,
                remaining,
                states,
                eligible_names,
            )
            if iteration == 0:
                first_scores = dict(scores)
            score_sum = sum(scores.values())
            if score_sum <= 0.0:
                break

            before = remaining
            requested_magnitude = abs(remaining)
            for name, score in scores.items():
                share = requested_magnitude * score / score_sum
                movement = min(share, capacities[name])
                delta = movement if remaining > 0.0 else -movement
                final[name] += delta
                added[name] += delta
            remaining = target_delta_W - sum(added.values())
            if math.isclose(remaining, before, abs_tol=tolerance):
                break

        # 最后统一消除浮点漂移，确保任何返回指令都不超过BMS硬边界。
        for name in DEVICE_NAMES:
            charge_limit, discharge_limit = limits[name]
            final[name] = _clip(final[name], -charge_limit, discharge_limit)
            added[name] = final[name] - initial[name]
        residual = target_delta_W - sum(added.values())
        return final, added, residual, first_scores

    def _allocate_in_priority_stages(
        self,
        initial: Mapping[str, float],
        limits: Mapping[str, Tuple[float, float]],
        target_delta_W: float,
        states: Mapping[str, DeviceBMSState],
        priority_stages: Sequence[Sequence[str]],
    ) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, float]]:
        """按照时间尺度优先级逐级转移未满足功率。"""
        final = {name: float(initial[name]) for name in DEVICE_NAMES}
        total_added = {name: 0.0 for name in DEVICE_NAMES}
        remaining = float(target_delta_W)
        audit_scores: Dict[str, float] = {}
        for stage_index, eligible_names in enumerate(priority_stages):
            if abs(remaining) <= self.config.tracking_tolerance_W:
                break
            final, added, remaining, scores = self._allocate_bounded(
                initial=final,
                limits=limits,
                target_delta_W=remaining,
                states=states,
                eligible_names=eligible_names,
            )
            self._accumulate_power(total_added, added)
            for name, score in scores.items():
                audit_scores["stage_%d:%s" % (stage_index + 1, name)] = score
        return final, total_added, remaining, audit_scores

    @staticmethod
    def _accumulate_power(
        destination: Dict[str, float], increment: Mapping[str, float]
    ) -> None:
        """把一组设备功率增量按名称累加到目标字典。"""
        for name in DEVICE_NAMES:
            destination[name] += float(increment.get(name, 0.0))

    def _mark_group_boundary_reasons(
        self,
        reasons: Dict[str, List[str]],
        setpoints: Mapping[str, float],
        limits: Mapping[str, Tuple[float, float]],
        group_names: Sequence[str],
        residual_W: float,
    ) -> None:
        """在组内能力不足时，为触及边界的设备补充审计原因。

        参数：
            reasons: 可原地修改的 ``{设备: 原因列表}``。
            setpoints: 组内第一次有界分配后的设备功率。
            limits: 每台设备当前有效充放电边界。
            group_names: 发生缺口的低频组或高频组设备列表。
            residual_W: 该组未满足功率；正值表示放电不足，负值表示充电不足。

        若设备已到相应方向边界，则分别追加
        ``group_discharge_boundary_reached`` 或
        ``group_charge_boundary_reached``。该函数只记录原因，不调整功率；真正
        的缺口补偿由随后面向全部设备的 ``_allocate_bounded`` 完成。
        """

        tolerance = self.config.tracking_tolerance_W
        for name in group_names:
            charge_limit, discharge_limit = limits[name]
            if residual_W > tolerance and setpoints[name] >= discharge_limit - tolerance:
                reasons[name].append("group_discharge_boundary_reached")
            elif residual_W < -tolerance and setpoints[name] <= -charge_limit + tolerance:
                reasons[name].append("group_charge_boundary_reached")


class SequenceAGCReader(AGCReader):
    """离线仿真和单元测试使用的AGC序列读取器。

    它不连接真实调度系统，而是按给定顺序逐条返回内存中的AGC命令，便于用
    可重复的数据回放EMS策略。
    """

    def __init__(self, commands: Sequence[AGCCommand]) -> None:
        """保存AGC序列的迭代器。

        参数：
            commands: 按时间顺序排列的AGC命令序列。

        这里使用迭代器而不是索引；每次 ``read`` 消耗一条命令，不会自动循环。
        """

        self._commands = iter(commands)

    def read(self) -> AGCCommand:
        """返回序列中的下一条AGC命令。

        当所有命令都已读取后，Python迭代器会抛出 ``StopIteration``。离线回放
        调用方应按序列长度控制循环，或自行捕获该异常。
        """

        return next(self._commands)


class SequenceThermalPowerReader(ThermalPowerReader):
    """离线仿真和单元测试使用的火电机组功率序列读取器。

    每次调用 ``read`` 都按顺序返回一条火电实发功率、其他机组功率以及对应
    时间戳。该类只负责提供测量值，不模拟火电机组的调速、爬坡和迟滞动态。
    """

    def __init__(self, states: Sequence[ThermalPowerState]) -> None:
        """保存按时间顺序排列的火电功率状态序列。"""

        self._states = iter(states)

    def read(self) -> ThermalPowerState:
        """返回序列中的下一条火电功率状态。"""

        return next(self._states)


class StaticBMSReader(BMSReader):
    """示例和测试使用的静态BMS读取器。

    每次读取都返回同一组设备状态，不会自动更新SOC、SOH或功率边界。它适合
    验证EMS分配算法，不代表真实设备动态；现场应替换为BMS通信适配器。
    """

    def __init__(self, states: Mapping[str, DeviceBMSState]) -> None:
        """复制并保存一组静态BMS状态映射。

        这里只复制字典结构，内部 ``DeviceBMSState`` 对象仍是同一对象；测试
        代码可以修改对象字段来模拟边界变化。
        """

        self.states = dict(states)

    def read(self) -> Mapping[str, DeviceBMSState]:
        """返回当前保存的四设备BMS状态映射。

        本函数没有硬件读取动作，也不会增加 ``data_age_s``。真实实现应在每个
        调度周期刷新数据并计算其年龄。
        """

        return self.states


class ConsolePCSInterface(PCSInterface):
    """控制台演示PCS：打印指令并保存最后一次下发值。

    该类不连接真实变流器，不具有通信确认或硬件保护作用，只用于展示EMS最终
    输出以及验证四台设备功率之和。
    """

    def __init__(self) -> None:
        """创建控制台PCS适配器，并初始化最后一次指令记录。

        ``last_setpoints_W`` 初始为空；第一次调用
        ``send_power_setpoints`` 后，它会保存四台设备的完整功率指令副本。
        """

        self.last_setpoints_W: Dict[str, float] = {}

    def send_power_setpoints(
        self, setpoints_W: Mapping[str, float], timestamp_s: float
    ) -> bool:
        """模拟向PCS下发四台设备功率，并在控制台打印结果。

        参数：
            setpoints_W: 四台设备的最终有符号功率指令。
            timestamp_s: 本次指令时间戳。

        返回：
            始终返回True，表示演示接口假定全部PCS接收成功。

        状态变化：
            将本次指令复制到 ``last_setpoints_W``，便于测试或外部代码读取。
        """

        self.last_setpoints_W = dict(setpoints_W)
        print(
            "t=%6.1fs | flow_battery=%8.1f W | lithium=%8.1f W | "
            "hybrid_sc=%8.1f W | EDLC=%8.1f W | total=%8.1f W"
            % (
                timestamp_s,
                setpoints_W[FLOW_BATTERY],
                setpoints_W[LITHIUM],
                setpoints_W[HYBRID_SUPERCAPACITOR],
                setpoints_W[DOUBLE_LAYER_SUPERCAPACITOR],
                sum(setpoints_W.values()),
            )
        )
        return True


def demo() -> None:
    """运行一个无需真实AGC、火电测量、BMS和PCS硬件的最小联合演示。

    AGC给出场站并网总功率目标，火电读取器给出机组实发功率，EMS只把两者的
    动态偏差分配给四类储能。控制台会显示各储能最终指令和总功率；若场站
    能力不足，还会显示未满足AGC残差。

    该示例只证明接口和分配流程可以运行。静态BMS不会根据已分配功率更新SOC，
    默认额定值和时间常数也不是工程标定结果，不能直接作为现场控制参数。
    """

    agc_reader = SequenceAGCReader(
        [
            AGCCommand(1000.0, 0.0),
            AGCCommand(5000.0, 1.0),
            AGCCommand(-3500.0, 2.0),
            AGCCommand(7000.0, 3.0),
            AGCCommand(0.0, 4.0),
        ]
    )
    thermal_reader = SequenceThermalPowerReader(
        [
            ThermalPowerState(800.0, 0.0),
            ThermalPowerState(4300.0, 1.0),
            ThermalPowerState(-3000.0, 2.0),
            ThermalPowerState(6200.0, 3.0),
            ThermalPowerState(200.0, 4.0),
        ]
    )
    bms_reader = StaticBMSReader(
        {
            FLOW_BATTERY: DeviceBMSState(
                FLOW_BATTERY,
                0.62,
                4200.0,
                4000.0,
                energy_capacity_Wh=10000.0,
                soe=0.62,
            ),
            LITHIUM: DeviceBMSState(
                LITHIUM,
                0.48,
                2600.0,
                2400.0,
                energy_capacity_Wh=5000.0,
                soe=0.48,
            ),
            HYBRID_SUPERCAPACITOR: DeviceBMSState(
                HYBRID_SUPERCAPACITOR,
                0.52,
                2200.0,
                2200.0,
                energy_capacity_Wh=300.0,
                soe=0.52,
            ),
            DOUBLE_LAYER_SUPERCAPACITOR: DeviceBMSState(
                DOUBLE_LAYER_SUPERCAPACITOR,
                0.50,
                1800.0,
                1800.0,
                energy_capacity_Wh=150.0,
                soe=0.50,
            ),
        }
    )
    pcs = ConsolePCSInterface()
    ems = HybridStorageEMS()

    print("Positive power = discharge; negative power = charge")
    for _ in range(5):
        result = ems.run_plant_cycle(
            agc_reader, thermal_reader, bms_reader, pcs, dt_s=1.0
        )
        if not result.agc_fully_met:
            print("  Unmet AGC residual: %.1f W" % result.tracking_residual_W)


if __name__ == "__main__":
    demo()
