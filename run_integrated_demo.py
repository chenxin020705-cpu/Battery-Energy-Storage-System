# -*- coding: utf-8 -*-
"""EMS/MPC/PCS切换闭环最小示例。

这里的DemoStorageAdapter只用于快速验证控制链；工程仿真时应替换为
hybrid_ems.py中已有的四种物理模型适配器。
"""

from dataclasses import replace

from hybrid_ems import (
    AGCCommand,
    DEVICE_NAMES,
    DeviceBMSState,
    DevicePowerFeedback,
    EMSConfig,
    HybridStorageEMS,
    StorageDeviceAdapter,
    ThermalPowerState,
)
from integrated_control import (
    ControlMode,
    MultiDevicePCS,
    PCSChannelConfig,
    SwitchableStorageSystem,
)
from MPC_three_models import PowerMPCConfig, PowerMPCController


class DemoStorageAdapter(StorageDeviceAdapter):
    def __init__(self, name: str, soc: float, power_W: float, energy_Wh: float):
        self.name = name
        self.state = DeviceBMSState(
            name=name,
            soc=soc,
            max_discharge_power_W=power_W,
            max_charge_power_W=power_W,
            soc_min=0.1,
            soc_max=0.9,
            energy_capacity_Wh=energy_Wh,
            soe=soc,
        )

    def read_state(self, dt_s, timestamp_s=None):
        return replace(self.state)

    def apply_power(self, power_W, dt_s, timestamp_s=None):
        actual = min(max(float(power_W), -self.state.max_charge_power_W), self.state.max_discharge_power_W)
        soc = self.state.soc-actual*dt_s/(self.state.energy_capacity_Wh*3600.0)
        self.state = replace(self.state, soc=soc, soe=soc)
        return DevicePowerFeedback(self.name, float(power_W), actual, True)


def build_system() -> SwitchableStorageSystem:
    ratings = {
        "flow_battery": 5000.0,
        "lithium": 3000.0,
        "hybrid_supercapacitor": 2500.0,
        "double_layer_supercapacitor": 2000.0,
    }
    energies = {
        "flow_battery": 40000.0,
        "lithium": 15000.0,
        "hybrid_supercapacitor": 1000.0,
        "double_layer_supercapacitor": 500.0,
    }
    adapters = {
        name: DemoStorageAdapter(name, 0.5, ratings[name], energies[name])
        for name in DEVICE_NAMES
    }
    channels = {
        name: PCSChannelConfig(
            rated_power_W=ratings[name],
            ramp_rate_W_s=ratings[name]*0.5,
            enable_frequency_droop=False,
        )
        for name in DEVICE_NAMES
    }
    ems_config = EMSConfig(use_agc_timestamp_delta=False)
    for name, policy in ems_config.device_policies.items():
        policy.rated_power_W = ratings[name]
    ems = HybridStorageEMS(ems_config)
    mpc = PowerMPCController(
        DEVICE_NAMES,
        soc_targets={name: ems_config.device_policies[name].soc_target for name in DEVICE_NAMES},
        rated_power_W=ratings,
        config=PowerMPCConfig(predict_horizon=8),
    )
    return SwitchableStorageSystem(MultiDevicePCS(adapters, channels), mpc, ems)


def main() -> None:
    system = build_system()
    commands = [3000.0, 6000.0, 7000.0, -4000.0, -3000.0, 2000.0]
    for index, command in enumerate(commands):
        timestamp = float(index+1)
        if index == 2:
            system.switch_mode(ControlMode.MPC, timestamp, "演示切换到MPC")
        if index == 5:
            system.switch_mode(ControlMode.EMS, timestamp, "演示切回EMS")
        result = system.run_cycle(
            AGCCommand(command, timestamp),
            dt_s=1.0,
            thermal=ThermalPowerState(0.0, timestamp),
            forecast_storage_power_W=[command]*8,
            frequency_Hz=50.0,
        )
        print(
            f"t={timestamp:.0f}s mode={result.controller_used.value:3s} "
            f"target={result.storage_target_W:8.1f}W "
            f"actual={sum(result.actual_power_W.values()):8.1f}W "
            f"residual={result.actual_tracking_residual_W:8.1f}W"
        )
    print("switch log:", system.switch_log)


if __name__ == "__main__":
    main()
