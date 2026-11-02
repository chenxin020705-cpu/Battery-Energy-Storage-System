# -*- coding: utf-8 -*-
"""EMS/MPC可切换的四储能-PCS统一闭环。

公共约定：功率单位W，正值放电、负值充电；状态读取不推进模型，只有
``MultiDevicePCS.send_power_setpoints`` 会让每台储能模型推进一次。
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from hybrid_ems import (
    AGCCommand,
    DEVICE_NAMES,
    DeviceBMSState,
    DevicePowerFeedback,
    DispatchResult,
    EMSConfig,
    HybridStorageEMS,
    PCSDispatchFeedback,
    PCSInterface,
    StorageDeviceAdapter,
    ThermalPowerState,
)
from MPC_three_models import PowerMPCController, PowerMPCResult


def _load_pcs_module() -> Any:
    """加载目录名含小数点的原PCS模块。"""
    path = Path(__file__).resolve().parent / "battery" / "SS0.1" / "pcs" / "control_sys.py"
    spec = importlib.util.spec_from_file_location("storage_pcs_control_sys", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PCS module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PCS_MODULE = _load_pcs_module()
pcs_per_unit = _PCS_MODULE.pcs_per_unit
pcs_control_sys = _PCS_MODULE.pcs_control_sys


class ControlMode(str, Enum):
    EMS = "ems"
    MPC = "mpc"
    SAFE = "safe"


@dataclass
class PCSChannelConfig:
    """一台设备对应的一条PCS有功通道配置。"""

    rated_power_W: float
    power_factor: float = 0.98
    overload_factor: float = 1.0
    ramp_rate_W_s: Optional[float] = None
    enable_frequency_droop: bool = False
    nominal_frequency_Hz: float = 50.0
    ac_voltage_V: float = 400.0

    def validate(self) -> None:
        if self.rated_power_W <= 0.0:
            raise ValueError("rated_power_W must be positive.")
        if not 0.0 < self.power_factor <= 1.0:
            raise ValueError("power_factor must be in (0, 1].")
        if self.overload_factor <= 0.0:
            raise ValueError("overload_factor must be positive.")
        if self.ramp_rate_W_s is not None and self.ramp_rate_W_s <= 0.0:
            raise ValueError("ramp_rate_W_s must be positive when provided.")
        if self.nominal_frequency_Hz <= 0.0 or self.ac_voltage_V <= 0.0:
            raise ValueError("Nominal frequency and AC voltage must be positive.")


@dataclass
class PCSChannelFeedback:
    name: str
    requested_power_W: float
    pcs_command_W: float
    actual_power_W: float
    frequency_support_W: float
    limited: bool
    accepted: bool
    device_feedback: DevicePowerFeedback


class MultiDevicePCS(PCSInterface):
    """四通道PCS执行层，并负责唯一一次设备状态提交。

    调度级模型复用原PCS的标幺基值与 ``steady_power_pu`` 规律。默认关闭PCS
    内部频率下垂，避免EMS/MPC已经跟踪AGC后再次叠加调频功率。
    """

    def __init__(
        self,
        adapters: Mapping[str, StorageDeviceAdapter],
        channel_configs: Mapping[str, PCSChannelConfig],
    ) -> None:
        if set(adapters) != set(DEVICE_NAMES):
            raise ValueError("Adapters are required for all four storage devices.")
        if set(channel_configs) != set(DEVICE_NAMES):
            raise ValueError("PCS channel configs are required for all four devices.")
        self.adapters = {name: adapters[name] for name in DEVICE_NAMES}
        self.configs = {name: channel_configs[name] for name in DEVICE_NAMES}
        self._pu: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        for name in DEVICE_NAMES:
            adapter = self.adapters[name]
            if not isinstance(adapter, StorageDeviceAdapter) or adapter.name != name:
                raise ValueError(f"Invalid adapter for {name}.")
            cfg = self.configs[name]
            cfg.validate()
            self._pu[name] = pcs_per_unit(
                P_rated_kW=cfg.rated_power_W/1000.0,
                PF=cfg.power_factor,
                f_base_Hz=cfg.nominal_frequency_Hz,
                U_base_V=cfg.ac_voltage_V,
            )
            self._models[name] = pcs_control_sys(
                q_mode="PQ", pf=cfg.power_factor, build_dynamic=False
            )
        self.frequency_Hz = 50.0
        self.dt_s = 1.0
        self.timestamp_s: Optional[float] = None
        self.last_actual_power_W = {name: 0.0 for name in DEVICE_NAMES}
        self.last_channel_feedback: Dict[str, PCSChannelFeedback] = {}
        self.execution_count = {name: 0 for name in DEVICE_NAMES}

    def set_grid_conditions(self, frequency_Hz: float) -> None:
        if not math.isfinite(frequency_Hz) or frequency_Hz <= 0.0:
            raise ValueError("frequency_Hz must be finite and positive.")
        self.frequency_Hz = float(frequency_Hz)

    def read_for_cycle(self, dt_s: float, timestamp_s: float) -> Dict[str, DeviceBMSState]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive.")
        if self.timestamp_s is not None and timestamp_s <= self.timestamp_s:
            raise ValueError("Cycle timestamps must be strictly increasing.")
        states = {
            name: self.adapters[name].read_state(float(dt_s), timestamp_s=float(timestamp_s))
            for name in DEVICE_NAMES
        }
        self.dt_s = float(dt_s)
        self.timestamp_s = float(timestamp_s)
        return states

    def read(self) -> Dict[str, DeviceBMSState]:
        timestamp = 0.0 if self.timestamp_s is None else self.timestamp_s
        return {
            name: self.adapters[name].read_state(self.dt_s, timestamp_s=timestamp)
            for name in DEVICE_NAMES
        }

    def _pcs_command(self, name: str, requested_W: float) -> tuple[float, float, bool]:
        cfg, pu, model = self.configs[name], self._pu[name], self._models[name]
        frequency = self.frequency_Hz if cfg.enable_frequency_droop else cfg.nominal_frequency_Hz
        p_pu, frc_pu = model.steady_power_pu(
            frequency/cfg.nominal_frequency_Hz,
            1.0,
            (requested_W/1000.0)/pu.S_base_kVA,
        )
        raw_W = float(p_pu*pu.S_base_kVA*1000.0)
        support_W = float(frc_pu*pu.S_base_kVA*1000.0)
        hardware_limit = cfg.rated_power_W*cfg.overload_factor
        command_W = min(max(raw_W, -hardware_limit), hardware_limit)
        if cfg.ramp_rate_W_s is not None:
            maximum_change = cfg.ramp_rate_W_s*self.dt_s
            previous = self.last_actual_power_W[name]
            command_W = min(max(command_W, previous-maximum_change), previous+maximum_change)
        limited = not math.isclose(command_W, requested_W, abs_tol=1.0e-6, rel_tol=0.0)
        return command_W, support_W, limited

    def send_power_setpoints(
        self, setpoints_W: Mapping[str, float], timestamp_s: float
    ) -> PCSDispatchFeedback:
        if set(setpoints_W) != set(DEVICE_NAMES):
            raise ValueError("PCS setpoints must contain exactly the four standard devices.")
        if self.timestamp_s is None or not math.isclose(float(timestamp_s), self.timestamp_s):
            raise ValueError("PCS command timestamp does not match the active control cycle.")
        actual: Dict[str, float] = {}
        details: Dict[str, PCSChannelFeedback] = {}
        for name in DEVICE_NAMES:
            requested = float(setpoints_W[name])
            if not math.isfinite(requested):
                raise ValueError(f"PCS setpoint for {name} must be finite.")
            command, support, limited = self._pcs_command(name, requested)
            device_feedback = self.adapters[name].apply_power(
                command, self.dt_s, timestamp_s=float(timestamp_s)
            )
            self.execution_count[name] += 1
            actual[name] = float(device_feedback.actual_power_W)
            details[name] = PCSChannelFeedback(
                name=name,
                requested_power_W=requested,
                pcs_command_W=command,
                actual_power_W=actual[name],
                frequency_support_W=support,
                limited=limited or not math.isclose(actual[name], command, abs_tol=1.0e-6),
                accepted=bool(device_feedback.accepted),
                device_feedback=device_feedback,
            )
        self.last_actual_power_W = dict(actual)
        self.last_channel_feedback = details
        return PCSDispatchFeedback(
            accepted=all(item.accepted for item in details.values()),
            actual_power_W=actual,
        )


@dataclass
class IntegratedCycleResult:
    timestamp_s: float
    requested_mode: ControlMode
    controller_used: ControlMode
    storage_target_W: float
    controller_setpoints_W: Dict[str, float]
    safe_setpoints_W: Dict[str, float]
    actual_power_W: Dict[str, float]
    actual_tracking_residual_W: float
    pcs_accepted: bool
    bms_states: Dict[str, DeviceBMSState]
    ems_result: Optional[DispatchResult] = None
    mpc_result: Optional[PowerMPCResult] = None
    fallback_reason: Optional[str] = None


class SwitchableStorageSystem:
    """在EMS、MPC和安全模式之间切换的唯一场站控制入口。"""

    def __init__(
        self,
        pcs: MultiDevicePCS,
        mpc: PowerMPCController,
        ems: Optional[HybridStorageEMS] = None,
        initial_mode: ControlMode | str = ControlMode.EMS,
        fallback_to_ems: bool = True,
    ) -> None:
        self.pcs = pcs
        self.mpc = mpc
        self.ems = ems if ems is not None else HybridStorageEMS(EMSConfig())
        self.mode = ControlMode(initial_mode)
        self.fallback_to_ems = bool(fallback_to_ems)
        self.switch_log: list[dict[str, Any]] = []

    def switch_mode(
        self, mode: ControlMode | str, timestamp_s: Optional[float] = None, reason: str = "manual"
    ) -> None:
        new_mode = ControlMode(mode)
        if new_mode == self.mode:
            return
        previous = self.mode
        self.mode = new_mode
        # 两套控制器均从PCS实发功率接续；最终斜坡由PCS统一执行。
        self.mpc.synchronize_actual(self.pcs.last_actual_power_W)
        if new_mode == ControlMode.EMS:
            self.ems.reset(tracking_power_W=sum(self.pcs.last_actual_power_W.values()))
        self.switch_log.append({
            "timestamp_s": timestamp_s,
            "from": previous.value,
            "to": new_mode.value,
            "reason": str(reason),
        })

    @staticmethod
    def _safe_clip(
        setpoints_W: Mapping[str, float], states: Mapping[str, DeviceBMSState]
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name in DEVICE_NAMES:
            state = states[name]
            value = float(setpoints_W.get(name, 0.0))
            if not state.available or state.fault:
                value = 0.0
            soc_min = 0.0 if state.soc_min is None else float(state.soc_min)
            soc_max = 1.0 if state.soc_max is None else float(state.soc_max)
            if state.soc <= soc_min and value > 0.0:
                value = 0.0
            if state.soc >= soc_max and value < 0.0:
                value = 0.0
            result[name] = min(
                max(value, -float(state.max_charge_power_W)),
                float(state.max_discharge_power_W),
            )
        return result

    def run_cycle(
        self,
        agc: AGCCommand,
        dt_s: float,
        thermal: Optional[ThermalPowerState] = None,
        forecast_storage_power_W: Optional[Sequence[float]] = None,
        frequency_Hz: float = 50.0,
    ) -> IntegratedCycleResult:
        """完成一次读取、选择控制器、安全限幅、PCS执行和反馈同步。"""
        agc.validate()
        if thermal is None:
            thermal = ThermalPowerState(0.0, agc.timestamp_s)
        thermal.validate(self.ems.config.max_thermal_data_age_s)
        self.pcs.set_grid_conditions(frequency_Hz)
        states = self.pcs.read_for_cycle(dt_s, agc.timestamp_s)
        storage_target = float(agc.power_W-thermal.actual_power_W-thermal.other_power_W)
        ems_result: Optional[DispatchResult] = None
        mpc_result: Optional[PowerMPCResult] = None
        fallback_reason: Optional[str] = None
        controller_used = self.mode

        if self.mode == ControlMode.SAFE:
            raw = {name: 0.0 for name in DEVICE_NAMES}
        elif self.mode == ControlMode.EMS:
            ems_result = self.ems.dispatch(agc, states, dt_s, pcs=None, thermal=thermal)
            raw = dict(ems_result.final_setpoints_W)
        else:
            try:
                mpc_result = self.mpc.calculate(
                    storage_target,
                    states,
                    dt_s,
                    forecast_power_W=forecast_storage_power_W,
                )
                raw = dict(mpc_result.setpoints_W)
            except Exception as exc:
                if not self.fallback_to_ems:
                    raise
                fallback_reason = f"MPC error: {exc}"
                controller_used = ControlMode.EMS
                self.ems.reset(tracking_power_W=sum(self.pcs.last_actual_power_W.values()))
                ems_result = self.ems.dispatch(agc, states, dt_s, pcs=None, thermal=thermal)
                raw = dict(ems_result.final_setpoints_W)

        safe = self._safe_clip(raw, states)
        feedback = self.pcs.send_power_setpoints(safe, agc.timestamp_s)
        actual = dict(feedback.actual_power_W or {name: 0.0 for name in DEVICE_NAMES})
        self.mpc.synchronize_actual(actual)
        residual = float(agc.power_W-thermal.actual_power_W-thermal.other_power_W-sum(actual.values()))
        return IntegratedCycleResult(
            timestamp_s=float(agc.timestamp_s),
            requested_mode=self.mode,
            controller_used=controller_used,
            storage_target_W=storage_target,
            controller_setpoints_W=dict(raw),
            safe_setpoints_W=safe,
            actual_power_W=actual,
            actual_tracking_residual_W=residual,
            pcs_accepted=bool(feedback.accepted),
            bms_states=dict(states),
            ems_result=ems_result,
            mpc_result=mpc_result,
            fallback_reason=fallback_reason,
        )

