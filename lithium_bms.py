"""分数阶锂电池包模型的BMS边界层。

功率和电流统一采用：正值放电、负值充电。底层 ``battery_pack`` 负责电压、
SOC和分数阶RC状态；本文件负责边界预测、限流、限功率及EMS所需状态整理。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import numpy as np

# 锂电模型来自同仓库的 SS0.1 子工程。该目录名包含小数点，不能作为普通
# Python 包导入，因此在这里显式加入模型目录。仍优先兼容外部部署时提供的
# ``frac_battery_model`` 模块。
_BATTERY_MODEL_IMPORT_ERROR: Optional[BaseException] = None
try:
    from frac_battery_model import battery_pack
except (ModuleNotFoundError, ImportError):
    _MODEL_DIR = (
        Path(__file__).resolve().parent
        / "battery"
        / "SS0.1"
        / "battery_cluster"
        / "model"
    )
    if str(_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(_MODEL_DIR))
    try:
        from battery_pack import battery_pack
    except ImportError as exc:
        # 允许BMS接口和系统集成代码在模型运行依赖不兼容时仍可导入；真正构造
        # 锂电模型时再给出明确错误。当前部分环境中的Numba尚不支持NumPy 2.x。
        battery_pack = Any  # type: ignore[misc,assignment]
        _BATTERY_MODEL_IMPORT_ERROR = exc


def require_lithium_model_dependencies() -> None:
    """确认分数阶锂电模型运行依赖可用。"""
    if _BATTERY_MODEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "分数阶锂电模型依赖不可用；请安装与当前NumPy兼容的Numba，或将"
            "NumPy降到Numba支持的版本。"
        ) from _BATTERY_MODEL_IMPORT_ERROR


@dataclass
class LithiumBatteryBMSConfig:
    """锂电池包的电气、SOC和温度安全边界。"""

    soc_min: float = 0.10
    soc_max: float = 0.90
    cell_voltage_min_V: float = 2.50
    cell_voltage_max_V: float = 3.65
    stack_voltage_min_V: Optional[float] = None
    stack_voltage_max_V: Optional[float] = None
    charge_current_max_A: float = 480.0
    discharge_current_max_A: float = 480.0
    temperature_min_C: float = 0.0
    temperature_max_C: float = 50.0
    temperature_low_derate_start_C: float = 5.0
    temperature_high_derate_start_C: float = 45.0
    current_ramp_rate_A_s: Optional[float] = None
    soh: float = 1.0
    soh_derate_start: Optional[float] = None
    soh_min: Optional[float] = None
    current_solver_tolerance_A: float = 1.0e-3
    current_solver_max_iterations: int = 18
    enforce_limits: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("SOC边界必须满足 0 <= soc_min < soc_max <= 1。")
        if self.cell_voltage_min_V >= self.cell_voltage_max_V:
            raise ValueError("单体最低电压必须小于最高电压。")
        if (
            self.stack_voltage_min_V is not None
            and self.stack_voltage_max_V is not None
            and self.stack_voltage_min_V >= self.stack_voltage_max_V
        ):
            raise ValueError("电池包最低电压必须小于最高电压。")
        if min(self.charge_current_max_A, self.discharge_current_max_A) < 0.0:
            raise ValueError("最大充放电电流不能为负。")
        if not (
            self.temperature_min_C
            < self.temperature_low_derate_start_C
            <= self.temperature_high_derate_start_C
            < self.temperature_max_C
        ):
            raise ValueError("温度停机点和降额起点的顺序不正确。")
        if self.current_ramp_rate_A_s is not None and self.current_ramp_rate_A_s <= 0.0:
            raise ValueError("电流爬坡率必须为正数。")
        if not 0.0 <= self.soh <= 1.0:
            raise ValueError("SOH必须位于[0, 1]。")
        if self.soh_derate_start is not None and self.soh_min is not None:
            if not 0.0 <= self.soh_min < self.soh_derate_start <= 1.0:
                raise ValueError("SOH降额边界顺序不正确。")
        if self.current_solver_tolerance_A <= 0.0:
            raise ValueError("电流求解容差必须为正数。")
        if self.current_solver_max_iterations <= 0:
            raise ValueError("电流求解迭代次数必须为正整数。")


@dataclass
class LithiumBatteryDeviceOutput:
    """锂电池BMS执行一个时间步后的状态与反馈。"""

    time_s: float
    dt_s: float
    requested_current_A: float
    current_A: float
    stack_voltage_V: float
    cell_voltage_min_V: float
    cell_voltage_max_V: float
    power_W: float
    soc: float
    cell_soc_min: float
    cell_soc_max: float
    soh: float
    temperature_C: float
    i_charge_max_A: float
    i_discharge_max_A: float
    limited: bool
    limit_reason: str
    fault: bool
    violations: list[str] = field(default_factory=list)
    cell_soc: list[float] = field(default_factory=list)
    cell_currents_A: list[float] = field(default_factory=list)
    group_voltages_V: list[float] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LithiumBatteryDevice:
    """把 ``battery_pack`` 封装为带动态边界的锂电池BMS设备。"""

    device_type = "LITHIUM_BATTERY"
    current_sign_convention = "positive_discharge_negative_charge"

    def __init__(
        self,
        model: battery_pack,
        config: Optional[LithiumBatteryBMSConfig] = None,
    ) -> None:
        if model.params is None:
            raise ValueError("请先调用battery_pack.set_cell_parameters()设置单体参数。")
        if model.memory_limit <= 0:
            raise ValueError("在线BMS要求battery_pack使用有限的memory_limit。")
        self.model = model
        self.config = config if config is not None else LithiumBatteryBMSConfig()
        self.config.validate()
        # 只确保在线状态已初始化，不重置调用方可能已经运行过的电池包。
        self.model.state()
        self._last_current_A = 0.0

    def _validate_dt(self, dt_s: float) -> None:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s必须是有限正数。")
        if not np.isclose(float(dt_s), float(self.model.dt), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                f"EMS时间步{dt_s} s必须与锂电模型dt={self.model.dt} s一致。"
            )

    def _weighted_soc(self, soc: np.ndarray) -> float:
        capacities = np.asarray(self.model.params["Q"], dtype=float)
        return float(np.sum(soc * capacities) / np.sum(capacities))

    def _temperature_factor(self, temperature_C: float) -> float:
        cfg = self.config
        if not np.isfinite(temperature_C):
            return 0.0
        if temperature_C <= cfg.temperature_min_C or temperature_C >= cfg.temperature_max_C:
            return 0.0
        if temperature_C < cfg.temperature_low_derate_start_C:
            return float(
                (temperature_C - cfg.temperature_min_C)
                / (cfg.temperature_low_derate_start_C - cfg.temperature_min_C)
            )
        if temperature_C > cfg.temperature_high_derate_start_C:
            return float(
                (cfg.temperature_max_C - temperature_C)
                / (cfg.temperature_max_C - cfg.temperature_high_derate_start_C)
            )
        return 1.0

    def _soh_factor(self) -> float:
        cfg = self.config
        if cfg.soh_derate_start is None or cfg.soh_min is None:
            return 1.0
        if cfg.soh <= cfg.soh_min:
            return 0.0
        if cfg.soh >= cfg.soh_derate_start:
            return 1.0
        return float(
            (cfg.soh - cfg.soh_min) / (cfg.soh_derate_start - cfg.soh_min)
        )

    def _prediction_violations(
        self, prediction: Mapping[str, Any], discharge: bool
    ) -> list[str]:
        cfg = self.config
        soc = np.asarray(prediction["SOC"], dtype=float)
        group_voltage = np.asarray(prediction["V_group"], dtype=float)
        stack_voltage = float(prediction["V_term"])
        issues: list[str] = []
        if discharge:
            if float(np.min(soc)) < cfg.soc_min - 1.0e-10:
                issues.append("cell_soc_below_min")
            if float(np.min(group_voltage)) < cfg.cell_voltage_min_V:
                issues.append("cell_voltage_below_min")
            if (
                cfg.stack_voltage_min_V is not None
                and stack_voltage < cfg.stack_voltage_min_V
            ):
                issues.append("stack_voltage_below_min")
        else:
            if float(np.max(soc)) > cfg.soc_max + 1.0e-10:
                issues.append("cell_soc_above_max")
            if float(np.max(group_voltage)) > cfg.cell_voltage_max_V:
                issues.append("cell_voltage_above_max")
            if (
                cfg.stack_voltage_max_V is not None
                and stack_voltage > cfg.stack_voltage_max_V
            ):
                issues.append("stack_voltage_above_max")
        return issues

    def predict_step(
        self, current_A: float, dt_s: float, temperature_C: float = 25.0
    ) -> dict[str, Any]:
        """预测一步但不改变锂电池模型内部状态。"""
        self._validate_dt(dt_s)
        if not np.isfinite(current_A):
            raise ValueError("current_A必须是有限数。")
        return self.model.predict_step(float(current_A), T=float(temperature_C))

    def _max_feasible_current(
        self, discharge: bool, upper_A: float, dt_s: float, temperature_C: float
    ) -> tuple[float, str]:
        state = self.model.state()
        soc = np.asarray(state["SOC"], dtype=float)
        if discharge and float(np.min(soc)) <= self.config.soc_min:
            return 0.0, "cell_soc_min"
        if not discharge and float(np.max(soc)) >= self.config.soc_max:
            return 0.0, "cell_soc_max"
        if upper_A <= 0.0:
            return 0.0, "derating_or_fault"

        sign = 1.0 if discharge else -1.0

        def feasible(magnitude_A: float) -> tuple[bool, list[str]]:
            prediction = self.predict_step(sign * magnitude_A, dt_s, temperature_C)
            issues = self._prediction_violations(prediction, discharge)
            return not issues, issues

        upper_ok, upper_issues = feasible(upper_A)
        if upper_ok:
            return float(upper_A), "rated_current_limit"
        low, high = 0.0, float(upper_A)
        for _ in range(self.config.current_solver_max_iterations):
            if high - low <= self.config.current_solver_tolerance_A:
                break
            middle = 0.5 * (low + high)
            if feasible(middle)[0]:
                low = middle
            else:
                high = middle
        return low, upper_issues[0] if upper_issues else "model_boundary"

    def get_current_limits(
        self, dt_s: float, temperature_C: float = 25.0
    ) -> dict[str, Any]:
        """返回下一时间步最大允许充电和放电电流幅值。"""
        self._validate_dt(dt_s)
        temperature_factor = self._temperature_factor(temperature_C)
        soh_factor = self._soh_factor()
        common_factor = min(temperature_factor, soh_factor)
        charge_upper = self.config.charge_current_max_A * common_factor
        discharge_upper = self.config.discharge_current_max_A * common_factor
        if self.config.current_ramp_rate_A_s is not None:
            delta_A = self.config.current_ramp_rate_A_s * dt_s
            discharge_upper = min(
                discharge_upper, max(0.0, self._last_current_A + delta_A)
            )
            charge_upper = min(
                charge_upper, max(0.0, -self._last_current_A + delta_A)
            )
        charge_A, charge_reason = self._max_feasible_current(
            False, charge_upper, dt_s, temperature_C
        )
        discharge_A, discharge_reason = self._max_feasible_current(
            True, discharge_upper, dt_s, temperature_C
        )
        return {
            "charge_A": charge_A,
            "discharge_A": discharge_A,
            "charge_reason": charge_reason,
            "discharge_reason": discharge_reason,
            "temperature_factor": temperature_factor,
            "soh_factor": soh_factor,
        }

    def get_power_limits(
        self, dt_s: float, temperature_C: float = 25.0
    ) -> dict[str, Any]:
        """返回下一时间步电池包直流端动态功率边界。"""
        limits = self.get_current_limits(dt_s, temperature_C)
        charge_A = float(limits["charge_A"])
        discharge_A = float(limits["discharge_A"])
        charge_prediction = self.predict_step(-charge_A, dt_s, temperature_C)
        discharge_prediction = self.predict_step(discharge_A, dt_s, temperature_C)
        charge_W = -charge_A * float(charge_prediction["V_term"])
        discharge_W = discharge_A * float(discharge_prediction["V_term"])
        return {
            **limits,
            "charge_W": min(0.0, charge_W),
            "charge_magnitude_W": max(0.0, -charge_W),
            "discharge_W": max(0.0, discharge_W),
            "charge_voltage_V": float(charge_prediction["V_term"]),
            "discharge_voltage_V": float(discharge_prediction["V_term"]),
            "power_reference": "battery_pack_dc_terminal",
        }

    def _current_for_power(
        self,
        power_W: float,
        dt_s: float,
        temperature_C: float,
        limits: Mapping[str, Any],
    ) -> float:
        if abs(power_W) <= 1.0e-12:
            return 0.0
        discharge = power_W > 0.0
        upper_A = float(limits["discharge_A"] if discharge else limits["charge_A"])
        target_W = abs(float(power_W))
        low, high = 0.0, upper_A
        sign = 1.0 if discharge else -1.0
        for _ in range(self.config.current_solver_max_iterations):
            middle = 0.5 * (low + high)
            prediction = self.predict_step(sign * middle, dt_s, temperature_C)
            predicted_W = abs(middle * float(prediction["V_term"]))
            if abs(predicted_W - target_W) <= max(0.1, 1.0e-6 * target_W):
                return sign * middle
            if predicted_W < target_W:
                low = middle
            else:
                high = middle
        return sign * 0.5 * (low + high)

    def _commit_current(
        self,
        requested_current_A: float,
        dt_s: float,
        temperature_C: float,
        limits: Mapping[str, Any],
    ) -> LithiumBatteryDeviceOutput:
        clipped_A = float(
            np.clip(
                requested_current_A,
                -float(limits["charge_A"]),
                float(limits["discharge_A"]),
            )
        )
        actual_A = clipped_A if self.config.enforce_limits else requested_current_A
        raw = self.model.step(actual_A, T=temperature_C)
        self._last_current_A = actual_A
        soc = np.asarray(raw["SOC"], dtype=float)
        group_voltage = np.asarray(raw["V_group"], dtype=float)
        discharge = actual_A >= 0.0
        violations = self._prediction_violations(raw, discharge)
        limited = not np.isclose(actual_A, requested_current_A, atol=1.0e-9, rtol=0.0)
        if limited:
            reason = limits["discharge_reason"] if requested_current_A > 0.0 else limits["charge_reason"]
        else:
            reason = "未限流"
        return LithiumBatteryDeviceOutput(
            time_s=float(raw["t"]),
            dt_s=float(dt_s),
            requested_current_A=float(requested_current_A),
            current_A=actual_A,
            stack_voltage_V=float(raw["V_term"]),
            cell_voltage_min_V=float(np.min(group_voltage)),
            cell_voltage_max_V=float(np.max(group_voltage)),
            power_W=float(raw["V_term"]) * actual_A,
            soc=self._weighted_soc(soc),
            cell_soc_min=float(np.min(soc)),
            cell_soc_max=float(np.max(soc)),
            soh=float(self.config.soh),
            temperature_C=float(temperature_C),
            i_charge_max_A=float(limits["charge_A"]),
            i_discharge_max_A=float(limits["discharge_A"]),
            limited=limited,
            limit_reason=reason,
            fault=bool(violations),
            violations=violations,
            cell_soc=soc.tolist(),
            cell_currents_A=np.asarray(raw["I_cell"], dtype=float).tolist(),
            group_voltages_V=group_voltage.tolist(),
            raw=dict(raw),
        )

    def step(
        self, current_A: float, dt_s: float, temperature_C: float = 25.0
    ) -> LithiumBatteryDeviceOutput:
        """限幅并执行一个电流控制时间步。"""
        if not np.isfinite(current_A):
            raise ValueError("current_A必须是有限数。")
        limits = self.get_current_limits(dt_s, temperature_C)
        return self._commit_current(current_A, dt_s, temperature_C, limits)

    def step_power(
        self,
        power_W: float,
        dt_s: float,
        temperature_C: float = 25.0,
        power_limits: Optional[Mapping[str, Any]] = None,
    ) -> LithiumBatteryDeviceOutput:
        """限幅并执行一个电池包直流端功率控制时间步。"""
        if not np.isfinite(power_W):
            raise ValueError("power_W必须是有限数。")
        limits = (
            self.get_power_limits(dt_s, temperature_C)
            if power_limits is None
            else dict(power_limits)
        )
        requested_W = float(power_W)
        clipped_W = float(
            np.clip(
                requested_W,
                -float(limits["charge_magnitude_W"]),
                float(limits["discharge_W"]),
            )
        )
        requested_A = self._current_for_power(
            clipped_W, dt_s, temperature_C, limits
        )
        output = self._commit_current(requested_A, dt_s, temperature_C, limits)
        if not np.isclose(requested_W, clipped_W, atol=1.0e-6, rtol=0.0):
            output.limited = True
            output.limit_reason = (
                limits["discharge_reason"] if requested_W > 0.0 else limits["charge_reason"]
            )
        output.raw.update(
            {
                "requested_power_W": requested_W,
                "clipped_power_W": clipped_W,
                "power_reference": "battery_pack_dc_terminal",
            }
        )
        return output

    def get_state(self, temperature_C: float = 25.0) -> dict[str, Any]:
        """读取当前锂电池状态，不推进时间。"""
        state = self.model.state()
        prediction = self.model.predict_step(0.0, T=temperature_C)
        soc = np.asarray(state["SOC"], dtype=float)
        group_voltage = np.asarray(prediction["V_group"], dtype=float)
        return {
            "device_type": self.device_type,
            "time_s": float(state["t"]),
            "soc": self._weighted_soc(soc),
            "cell_soc_min": float(np.min(soc)),
            "cell_soc_max": float(np.max(soc)),
            "soh": float(self.config.soh),
            "stack_voltage_V": float(prediction["V_term"]),
            "cell_voltage_min_V": float(np.min(group_voltage)),
            "cell_voltage_max_V": float(np.max(group_voltage)),
            "temperature_C": float(temperature_C),
            "cell_soc": soc.copy(),
            "state": state,
        }

    def reset(self, soc0: Optional[Any] = None) -> None:
        """复位锂电池模型和BMS上一时刻电流。"""
        self.model.reset(soc0=soc0)
        self._last_current_A = 0.0
