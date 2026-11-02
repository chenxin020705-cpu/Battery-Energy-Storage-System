import unittest
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


class CountingAdapter(StorageDeviceAdapter):
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.state = DeviceBMSState(
            name=name,
            soc=0.55,
            max_discharge_power_W=5000.0,
            max_charge_power_W=5000.0,
            soc_min=0.1,
            soc_max=0.9,
            energy_capacity_Wh=10000.0,
        )

    def read_state(self, dt_s, timestamp_s=None):
        return replace(self.state)

    def apply_power(self, power_W, dt_s, timestamp_s=None):
        self.calls += 1
        next_soc = self.state.soc-float(power_W)*dt_s/(self.state.energy_capacity_Wh*3600.0)
        self.state = replace(self.state, soc=max(0.1, min(0.9, next_soc)))
        return DevicePowerFeedback(
            name=self.name,
            requested_power_W=float(power_W),
            actual_power_W=float(power_W),
            accepted=True,
        )


def build_system(initial_mode=ControlMode.EMS):
    adapters = {name: CountingAdapter(name) for name in DEVICE_NAMES}
    channels = {
        name: PCSChannelConfig(
            rated_power_W=5000.0,
            ramp_rate_W_s=2000.0,
            enable_frequency_droop=False,
        )
        for name in DEVICE_NAMES
    }
    pcs = MultiDevicePCS(adapters, channels)
    ems_cfg = EMSConfig(use_agc_timestamp_delta=False)
    for policy in ems_cfg.device_policies.values():
        policy.rated_power_W = 5000.0
    ems = HybridStorageEMS(ems_cfg)
    mpc = PowerMPCController(
        DEVICE_NAMES,
        soc_targets={name: 0.5 for name in DEVICE_NAMES},
        rated_power_W={name: 5000.0 for name in DEVICE_NAMES},
        config=PowerMPCConfig(predict_horizon=4, default_ramp_fraction_per_s=0.4),
    )
    return SwitchableStorageSystem(pcs, mpc, ems, initial_mode), adapters


class IntegratedControlTests(unittest.TestCase):
    def test_ems_and_mpc_share_one_pcs_execution_path(self):
        system, adapters = build_system()
        thermal = ThermalPowerState(1000.0, 1.0)
        first = system.run_cycle(AGCCommand(5000.0, 1.0), 1.0, thermal, frequency_Hz=49.5)
        self.assertEqual(first.controller_used, ControlMode.EMS)
        self.assertEqual(set(first.actual_power_W), set(DEVICE_NAMES))
        self.assertTrue(first.pcs_accepted)
        self.assertTrue(all(adapter.calls == 1 for adapter in adapters.values()))

        system.switch_mode(ControlMode.MPC, timestamp_s=2.0)
        second = system.run_cycle(
            AGCCommand(5000.0, 2.0),
            1.0,
            ThermalPowerState(1000.0, 2.0),
            forecast_storage_power_W=[4000.0]*4,
            frequency_Hz=49.5,
        )
        self.assertEqual(second.controller_used, ControlMode.MPC)
        self.assertIsNotNone(second.mpc_result)
        self.assertTrue(all(adapter.calls == 2 for adapter in adapters.values()))
        for name, value in second.safe_setpoints_W.items():
            self.assertLessEqual(value, second.bms_states[name].max_discharge_power_W)
            self.assertGreaterEqual(value, -second.bms_states[name].max_charge_power_W)

    def test_safe_mode_and_switch_are_ramp_limited(self):
        system, _ = build_system(ControlMode.MPC)
        first = system.run_cycle(AGCCommand(16000.0, 1.0), 1.0)
        system.switch_mode(ControlMode.SAFE, timestamp_s=2.0, reason="test")
        second = system.run_cycle(AGCCommand(16000.0, 2.0), 1.0)
        for name in DEVICE_NAMES:
            self.assertLessEqual(
                abs(second.actual_power_W[name]-first.actual_power_W[name]),
                2000.0+1.0e-6,
            )
        self.assertEqual(second.controller_setpoints_W, {name: 0.0 for name in DEVICE_NAMES})
        self.assertEqual(system.switch_log[-1]["reason"], "test")

    def test_mpc_is_pure_calculation(self):
        system, adapters = build_system(ControlMode.MPC)
        states = {name: adapter.read_state(1.0) for name, adapter in adapters.items()}
        before = {name: adapter.state.soc for name, adapter in adapters.items()}
        result = system.mpc.calculate(3000.0, states, 1.0, [3000.0]*4)
        self.assertEqual(set(result.setpoints_W), set(DEVICE_NAMES))
        self.assertEqual(before, {name: adapter.state.soc for name, adapter in adapters.items()})
        self.assertTrue(all(adapter.calls == 0 for adapter in adapters.values()))


if __name__ == "__main__":
    unittest.main()
