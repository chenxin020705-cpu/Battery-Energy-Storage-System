# -*- coding: utf-8 -*-
"""使用四种真实储能模型验证 EMS/MPC—PCS 闭环。"""

from pathlib import Path

from bms import FlowBatteryDevice
from EDLC_HSC_Model import EDLC_Supercapacitor, HSC_HybridSupercapacitor
from hybrid_ems import (
    AGCCommand,
    DEVICE_NAMES,
    DOUBLE_LAYER_SUPERCAPACITOR,
    EDLCEMSAdapter,
    EMSConfig,
    FLOW_BATTERY,
    FlowBatteryEMSAdapter,
    HYBRID_SUPERCAPACITOR,
    HybridStorageEMS,
    HybridSupercapacitorEMSAdapter,
    LITHIUM,
    LithiumBatteryEMSAdapter,
)
from integrated_control import (
    ControlMode,
    MultiDevicePCS,
    PCSChannelConfig,
    SwitchableStorageSystem,
)
from lithium_bms import LithiumBatteryBMSConfig, LithiumBatteryDevice, battery_pack
from MPC_three_models import PowerMPCConfig, PowerMPCController


def build_lithium(dt_s: float) -> LithiumBatteryDevice:
    root = Path(__file__).resolve().parent
    ocv_dir = root / "battery" / "SS0.1" / "battery_cluster" / "ocv"
    model = battery_pack(
        12,
        N_p=1,
        dt=dt_s,
        memory_limit=30,
        store_history=False,
        ocv_dis_csv=str(ocv_dir / "ocv_branch_dis_seg.csv"),
        ocv_chg_csv=str(ocv_dir / "ocv_branch_chg_seg.csv"),
    )
    model.set_cell_parameters(
        R0=0.001,
        R1=0.002,
        C1=2000.0,
        alpha1=0.9,
        R2=0.004,
        C2=10000.0,
        alpha2=0.7,
        Q=40.0,
        soc0=0.5,
        eta=0.99,
    )
    return LithiumBatteryDevice(
        model,
        LithiumBatteryBMSConfig(
            charge_current_max_A=20.0,
            discharge_current_max_A=20.0,
        ),
    )


def build_system(dt_s: float = 1.0) -> tuple[SwitchableStorageSystem, dict]:
    flow = FlowBatteryDevice(initial_soc=0.5)
    lithium = build_lithium(dt_s)
    hsc = HSC_HybridSupercapacitor(initial_soc=0.5, dt=dt_s)
    edlc = EDLC_Supercapacitor(initial_soc=0.5, dt=dt_s)
    adapters = {
        FLOW_BATTERY: FlowBatteryEMSAdapter(flow, 0.98, 0.98),
        LITHIUM: LithiumBatteryEMSAdapter(lithium, pcs_charge_efficiency=0.98, pcs_discharge_efficiency=0.98),
        HYBRID_SUPERCAPACITOR: HybridSupercapacitorEMSAdapter(hsc, pcs_charge_efficiency=0.98, pcs_discharge_efficiency=0.98),
        DOUBLE_LAYER_SUPERCAPACITOR: EDLCEMSAdapter(edlc, pcs_charge_efficiency=0.98, pcs_discharge_efficiency=0.98),
    }
    ratings = {
        FLOW_BATTERY: 3000.0,
        LITHIUM: 1000.0,
        HYBRID_SUPERCAPACITOR: 300.0,
        DOUBLE_LAYER_SUPERCAPACITOR: 2200.0,
    }
    channels = {
        name: PCSChannelConfig(
            rated_power_W=ratings[name],
            ramp_rate_W_s=ratings[name],
            enable_frequency_droop=False,
        )
        for name in DEVICE_NAMES
    }
    config = EMSConfig(use_agc_timestamp_delta=False)
    for name in DEVICE_NAMES:
        config.device_policies[name].rated_power_W = ratings[name]
    ems = HybridStorageEMS(config)
    mpc = PowerMPCController(
        DEVICE_NAMES,
        soc_targets={name: config.device_policies[name].soc_target for name in DEVICE_NAMES},
        rated_power_W=ratings,
        config=PowerMPCConfig(predict_horizon=4),
    )
    system = SwitchableStorageSystem(MultiDevicePCS(adapters, channels), mpc, ems)
    return system, {"flow": flow, "lithium": lithium, "hsc": hsc, "edlc": edlc}


def main() -> None:
    system, devices = build_system()
    ems_result = system.run_cycle(AGCCommand(1000.0, 1.0), 1.0)
    system.switch_mode(ControlMode.MPC, 2.0, "physical smoke test")
    mpc_result = system.run_cycle(
        AGCCommand(1500.0, 2.0),
        1.0,
        forecast_storage_power_W=[1500.0]*4,
    )
    assert all(count == 2 for count in system.pcs.execution_count.values())
    assert devices["flow"].time_s == 2.0
    assert devices["lithium"].model.state()["t"] == 2.0
    assert devices["hsc"].sim_time == 2.0
    assert devices["edlc"].sim_time == 2.0
    assert mpc_result.mpc_result is not None and not mpc_result.mpc_result.used_fallback
    print("EMS actual W:", ems_result.actual_power_W)
    print("MPC actual W:", mpc_result.actual_power_W)
    print("MPC solver:", mpc_result.mpc_result.solver)
    print("Each physical model advanced exactly two steps: OK")


if __name__ == "__main__":
    main()
