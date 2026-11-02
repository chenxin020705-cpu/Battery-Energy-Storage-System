# -*- coding: utf-8 -*-
"""EDLC 与混合超级电容在线模型。

统一符号：电流/功率为正表示放电，为负表示充电。读取状态不会推进模型；
只有 step() 和 step_power() 会改变状态。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np


def _dataframe(records):
    """仅在导出表格时加载pandas，避免核心在线模型被可选依赖阻塞。"""
    try:
        import pandas as pd
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("导出DataFrame需要与当前NumPy兼容的pandas。") from exc
    return pd.DataFrame(records)


@dataclass
class SupercapacitorOutput:
    time_s: float
    dt_s: float
    requested_current_A: float
    current_A: float
    voltage_V: float
    power_W: float
    soc: float
    soe: float
    soh: float
    capacity_fade: float
    power_fade: float
    eir_ohm: float
    temperature_C: float
    i_charge_max_A: float
    i_discharge_max_A: float
    limited: bool
    limit_reason: str
    fault: bool
    violations: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def stack_voltage_V(self) -> float:
        return self.voltage_V

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _SecondOrderSupercapacitor:
    """二阶RC公共模型，极化支路采用解析离散以保证数值稳定。"""

    device_type = "SUPERCAPACITOR"
    current_sign_convention = "positive_discharge_negative_charge"

    def __init__(self, dev_c=0.0, dev_r=0.0, dev_soc=0.0,
                 initial_soc=0.5, dt=0.5):
        self._set_nominal_parameters()
        if 1.0 + dev_c <= 0.0 or 1.0 + dev_r <= 0.0:
            raise ValueError("参数偏差不能使电阻或电容变为非正数。")
        if dt <= 0.0:
            raise ValueError("dt必须为正。")
        self.C_nom *= 1.0 + dev_c
        self.R0_nom *= 1.0 + dev_r
        self.R1_nom *= 1.0 + dev_r
        self.R2_nom *= 1.0 + dev_r
        self.C1_nom *= 1.0 + dev_c
        self.C2_nom *= 1.0 + dev_c
        self.dt = float(dt)
        soc0 = float(np.clip(initial_soc + dev_soc, 0.0, 1.0))
        self.Uc = self.U_cutoff + soc0 * (self.U_nom - self.U_cutoff)
        self.U1 = self.U2 = self.I = 0.0
        self.T = self.T_ref
        self.sim_time = 0.0
        self.cycle_count = 0.0
        self.aging_equivalent_cycles = 0.0
        self.capacity_fade = self.power_fade = 0.0
        self.R_aging_factor = 1.0
        self.sim_data: List[Dict[str, Any]] = []
        self.cycle_summary: List[Dict[str, Any]] = []
        self.bms_alarm = ""
        self.bms = {
            "U_max": self.U_nom, "U_min": self.U_cutoff,
            "I_charge_max": -self.charge_current_max_A,
            "I_discharge_max": self.discharge_current_max_A,
            "T_min": self.temperature_min_C, "T_max": self.temperature_max_C,
            "U_margin": self.voltage_margin_V,
        }

    def _set_nominal_parameters(self):
        raise NotImplementedError

    def _temperature_factor(self, T):
        if not np.isfinite(T) or T <= self.temperature_min_C or T >= self.temperature_max_C:
            return 0.0
        if T < self.temperature_low_derate_start_C:
            return (T-self.temperature_min_C)/(self.temperature_low_derate_start_C-self.temperature_min_C)
        if T > self.temperature_high_derate_start_C:
            return (self.temperature_max_C-T)/(self.temperature_max_C-self.temperature_high_derate_start_C)
        return 1.0

    def _aging_temperature_factor(self, T):
        ref = self.T_ref + 273.15
        kelvin = max(1.0, float(T) + 273.15)
        exponent = self.Ea/self.Rg * (1.0/ref - 1.0/kelvin)
        return math.exp(float(np.clip(exponent, -20.0, 20.0)))

    def _update_aging(self):
        max_fade = 1.0 - self.SOH_threshold/100.0
        self.capacity_fade = float(np.clip(
            max_fade*self.aging_equivalent_cycles/self.cycle_nom, 0.0, max_fade))
        fraction = self.capacity_fade/max_fade if max_fade else 0.0
        self.R_aging_factor = 1.0 + self.resistance_growth_at_eol*fraction
        self.power_fade = float(np.clip(
            1.0-(1.0-self.capacity_fade)/self.R_aging_factor, 0.0, 1.0))

    def _parameters(self, T=None):
        T = self.T if T is None else float(T)
        dT = T-self.T_ref
        rf = math.exp(-self.resistance_temp_coefficient*dT)*self.R_aging_factor
        cf = max(0.5, 1.0+self.capacitance_temp_coefficient*dT)
        return (self.R0_nom*rf, self.R1_nom*rf, self.R2_nom*rf,
                self.C_nom*cf*(1.0-self.capacity_fade),
                self.C1_nom*cf, self.C2_nom*cf)

    def temp_correction(self):
        """兼容旧接口；纯读取有效参数。"""
        return self._parameters(self.T)

    def _terminal_voltage(self, I=0.0, Uc=None, U1=None, U2=None, T=None):
        R0 = self._parameters(T)[0]
        return float((self.Uc if Uc is None else Uc)-I*R0-
                     (self.U1 if U1 is None else U1)-
                     (self.U2 if U2 is None else U2))

    def _predict_raw(self, I, dt, T):
        R0, R1, R2, C, C1, C2 = self._parameters(T)
        a1, a2 = math.exp(-dt/(R1*C1)), math.exp(-dt/(R2*C2))
        U1 = a1*self.U1 + R1*I*(1.0-a1)
        U2 = a2*self.U2 + R2*I*(1.0-a2)
        Uc = self.Uc-I*dt/C
        U = Uc-I*R0-U1-U2
        return {"Uc": Uc, "U1": U1, "U2": U2,
                "voltage_V": U, "power_W": U*I}

    def predict_step(self, current_A, dt_s=None, temperature_C=None):
        """预测一步但不提交状态。"""
        dt = self.dt if dt_s is None else float(dt_s)
        T = self.T if temperature_C is None else float(temperature_C)
        if dt <= 0.0 or not np.isfinite(current_A):
            raise ValueError("dt必须为正且电流必须为有限数。")
        return self._predict_raw(float(current_A), dt, T)

    def _violations(self, prediction, discharge):
        result = []
        if discharge:
            if prediction["Uc"] < self.U_cutoff-1e-9: result.append("soc_below_min")
            if prediction["voltage_V"] < self.U_cutoff: result.append("voltage_below_min")
        else:
            if prediction["Uc"] > self.U_nom+1e-9: result.append("soc_above_max")
            if prediction["voltage_V"] > self.U_nom: result.append("voltage_above_max")
        return result

    def _max_current(self, discharge, upper, dt, T):
        if discharge and self._soc() <= 1e-12: return 0.0, "soc_min"
        if not discharge and self._soc() >= 1.0-1e-12: return 0.0, "soc_max"
        if upper <= 0.0: return 0.0, "temperature_or_soh_derating"
        sign = 1.0 if discharge else -1.0
        def feasible(x): return not self._violations(self._predict_raw(sign*x, dt, T), discharge)
        if feasible(upper): return float(upper), "rated_current_limit"
        low, high = 0.0, float(upper)
        for _ in range(25):
            middle = (low+high)/2.0
            if feasible(middle): low = middle
            else: high = middle
        return low, "voltage_or_soc_boundary"

    def get_current_limits(self, dt_s=None, temperature_C=None):
        dt = self.dt if dt_s is None else float(dt_s)
        T = self.T if temperature_C is None else float(temperature_C)
        tf = self._temperature_factor(T)
        soh = 1.0-self.capacity_fade
        soh_end = self.SOH_threshold/100.0
        sf = float(np.clip((soh-soh_end)/(1.0-soh_end), 0.0, 1.0))
        factor = min(tf, sf)
        ca, cr = self._max_current(False, self.charge_current_max_A*factor, dt, T)
        da, dr = self._max_current(True, self.discharge_current_max_A*factor, dt, T)
        return {"charge_A": ca, "discharge_A": da,
                "charge_reason": cr, "discharge_reason": dr,
                "temperature_factor": tf, "soh_factor": sf}

    def get_power_limits(self, dt_s=None, temperature_C=None):
        dt = self.dt if dt_s is None else float(dt_s)
        T = self.T if temperature_C is None else float(temperature_C)
        limits = self.get_current_limits(dt, T)
        ca, da = limits["charge_A"], limits["discharge_A"]
        cp, dp = self._predict_raw(-ca, dt, T), self._predict_raw(da, dt, T)
        cw, dw = -ca*cp["voltage_V"], da*dp["voltage_V"]
        return {**limits, "charge_W": min(0.0, cw),
                "charge_magnitude_W": max(0.0, -cw),
                "discharge_W": max(0.0, dw),
                "charge_voltage_V": cp["voltage_V"],
                "discharge_voltage_V": dp["voltage_V"],
                "power_reference": "supercapacitor_dc_terminal"}

    def step(self, current_A, dt_s=None, temperature_C=None, current_limits=None):
        """限幅并推进一个时间步。"""
        dt = self.dt if dt_s is None else float(dt_s)
        T = self.T if temperature_C is None else float(temperature_C)
        limits = self.get_current_limits(dt, T) if current_limits is None else dict(current_limits)
        requested = float(current_A)
        actual = float(np.clip(requested, -limits["charge_A"], limits["discharge_A"]))
        p = self._predict_raw(actual, dt, T)
        self.Uc = float(np.clip(p["Uc"], self.U_cutoff, self.U_nom))
        self.U1, self.U2, self.I, self.T = p["U1"], p["U2"], actual, T
        self.sim_time += dt
        usable_q = self.C_nom*(self.U_nom-self.U_cutoff)
        cycles = abs(actual)*dt/max(2.0*usable_q, 1e-12)
        self.cycle_count += cycles
        self.aging_equivalent_cycles += cycles*self._aging_temperature_factor(T)
        self._update_aging()
        state = self.get_state(T)
        committed = {"Uc": self.Uc, "voltage_V": state["voltage_V"]}
        violations = self._violations(committed, actual >= 0.0)
        limited = not np.isclose(actual, requested)
        reason = (limits["discharge_reason"] if requested > 0 else limits["charge_reason"]) if limited else "未限流"
        self.bms_alarm = reason if limited else ""
        out = SupercapacitorOutput(
            self.sim_time, dt, requested, actual, state["voltage_V"],
            state["voltage_V"]*actual, state["soc"], state["soe"], state["soh"],
            self.capacity_fade, self.power_fade, state["eir_ohm"], T,
            limits["charge_A"], limits["discharge_A"], limited, reason,
            bool(violations) or self._temperature_factor(T) <= 0.0, violations,
            {"Uc": self.Uc, "U1": self.U1, "U2": self.U2})
        self._record(out)
        return out

    def _current_for_power(self, power, dt, T, limits):
        if abs(power) <= 1e-12: return 0.0
        sign = 1.0 if power > 0.0 else -1.0
        high = limits["discharge_A"] if sign > 0 else limits["charge_A"]
        low, target = 0.0, abs(power)
        for _ in range(28):
            mid = (low+high)/2.0
            pred = self._predict_raw(sign*mid, dt, T)
            if abs(mid*pred["voltage_V"]) < target: low = mid
            else: high = mid
        return sign*(low+high)/2.0

    def step_power(self, power_W, dt_s=None, temperature_C=None, power_limits=None):
        dt = self.dt if dt_s is None else float(dt_s)
        T = self.T if temperature_C is None else float(temperature_C)
        limits = self.get_power_limits(dt, T) if power_limits is None else dict(power_limits)
        requested = float(power_W)
        clipped = float(np.clip(requested, -limits["charge_magnitude_W"], limits["discharge_W"]))
        current = self._current_for_power(clipped, dt, T, limits)
        out = self.step(current, dt, T, limits)
        if not np.isclose(requested, clipped):
            out.limited, out.limit_reason = True, "power_boundary"
        out.raw.update({"requested_power_W": requested, "clipped_power_W": clipped,
                        "power_reference": "supercapacitor_dc_terminal"})
        return out

    def _soc(self):
        return float(np.clip((self.Uc-self.U_cutoff)/(self.U_nom-self.U_cutoff), 0.0, 1.0))

    def _soe(self):
        return float(np.clip((self.Uc**2-self.U_cutoff**2)/
                             (self.U_nom**2-self.U_cutoff**2), 0.0, 1.0))

    def _rul(self):
        left = max(0.0, self.cycle_nom-self.aging_equivalent_cycles)
        factor = self._aging_temperature_factor(self.T)
        return int(left/factor) if factor else 0

    def get_state(self, temperature_C=None):
        """纯读取状态，不改变时间、电压和老化。"""
        T = self.T if temperature_C is None else float(temperature_C)
        R0, R1, R2, C, _, _ = self._parameters(T)
        voltage = self._terminal_voltage(self.I, T=T)
        return {"device_type": self.device_type, "time_s": self.sim_time,
                "voltage_V": voltage, "stack_voltage_V": voltage,
                "current_A": self.I, "soc": self._soc(), "soe": self._soe(),
                "soh": 1.0-self.capacity_fade, "capacity_fade": self.capacity_fade,
                "power_fade": self.power_fade, "eir_ohm": R0+R1+R2,
                "effective_capacitance_F": C, "temperature_C": T,
                "cycle_count": self.cycle_count, "rul_cycles": self._rul()}

    def calculate_states(self):
        """兼容旧中文字段；现为纯读取函数。"""
        s = self.get_state()
        return {"时间(s)": round(s["time_s"], 6), "端电压(V)": round(s["voltage_V"], 6),
                "SOC(%)": round(100*s["soc"], 4), "SOE(%)": round(100*s["soe"], 4),
                "SOH(%)": round(100*s["soh"], 4),
                "容量衰减(%)": round(100*s["capacity_fade"], 6),
                "EIR(Ω)": round(s["eir_ohm"], 8),
                "功率衰减(%)": round(100*s["power_fade"], 6),
                "RUL(循环)": s["rul_cycles"], "循环次数": s["cycle_count"]}

    def _record(self, o):
        self.sim_data.append({"时间(s)": o.time_s, "端电压(V)": o.voltage_V,
            "电流(A)": o.current_A, "功率(W)": o.power_W,
            "SOC(%)": 100*o.soc, "SOE(%)": 100*o.soe, "SOH(%)": 100*o.soh,
            "容量衰减(%)": 100*o.capacity_fade, "EIR(Ω)": o.eir_ohm,
            "功率衰减(%)": 100*o.power_fade, "RUL(循环)": self._rul(),
            "循环次数": self.cycle_count})

    def update_voltage(self):
        """旧接口：按当前I推进一次。"""
        return self.step(self.I, self.dt, self.T).voltage_V

    def bms_current_limit(self, I_cmd, Ut=None):
        limits = self.get_current_limits(self.dt, self.T)
        return float(np.clip(I_cmd, -limits["charge_A"], limits["discharge_A"]))

    def cycle_aging_update(self, equivalent_cycles=1.0):
        if equivalent_cycles < 0: raise ValueError("等效循环数不能为负。")
        self.cycle_count += equivalent_cycles
        self.aging_equivalent_cycles += equivalent_cycles*self._aging_temperature_factor(self.T)
        self._update_aging()

    def simulate_rest(self, rest_time=10):
        elapsed = 0.0
        while elapsed < rest_time-1e-12:
            dt = min(self.dt, rest_time-elapsed); self.step(0.0, dt, self.T); elapsed += dt

    def _constant_stage(self, current, max_time=None):
        if abs(current) <= 1e-12: raise ValueError("电流不能为0。")
        if max_time is None:
            max_time = 2*self.C_nom*(self.U_nom-self.U_cutoff)/abs(current)
        elapsed = 0.0
        while elapsed < max_time-1e-12:
            dt = min(self.dt, max_time-elapsed); out = self.step(current, dt, self.T); elapsed += dt
            if out.soc <= 1e-9 or out.soc >= 1-1e-9 or abs(out.current_A) <= 1e-12: break

    def simulate_charge(self, current_mode="constant", current_params=None,
                        max_charge_time=None, charge_current=None):
        params = {} if current_params is None else dict(current_params)
        if current_mode != "constant":
            return self._custom_stage(True, params, max_charge_time)
        value = charge_current if charge_current is not None else params.get("current", 10.0)
        self._constant_stage(-abs(float(value)), max_charge_time)

    def simulate_discharge(self, current_mode="constant", current_params=None,
                           max_discharge_time=None, discharge_current=None):
        params = {} if current_params is None else dict(current_params)
        if current_mode != "constant":
            return self._custom_stage(False, params, max_discharge_time)
        value = discharge_current if discharge_current is not None else params.get("current", 10.0)
        self._constant_stage(abs(float(value)), max_discharge_time)

    def _custom_stage(self, charge, params, max_time):
        if max_time is None or not callable(params.get("func")):
            raise ValueError("自定义工况必须提供func和最大持续时间。")
        elapsed = 0.0
        while elapsed < max_time-1e-12:
            dt = min(self.dt, max_time-elapsed)
            value = abs(float(params["func"](elapsed)))
            self.step(-value if charge else value, dt, self.T); elapsed += dt

    def simulate_mix(self, current_params=None, max_time=None):
        if max_time is None: raise ValueError("混合工况必须提供max_time。")
        params = {} if current_params is None else dict(current_params)
        if not callable(params.get("func")): raise ValueError("混合工况必须提供func。")
        elapsed = 0.0
        while elapsed < max_time-1e-12:
            dt = min(self.dt, max_time-elapsed)
            self.step(-float(params["func"](elapsed)), dt, self.T); elapsed += dt

    def simulate_full_cycle(self, charge_current=10, discharge_current=15, if_rest=False, **kwargs):
        self.simulate_charge(charge_current=charge_current)
        if if_rest: self.simulate_rest()
        self.simulate_discharge(discharge_current=discharge_current)
        if if_rest: self.simulate_rest()
        s = self.calculate_states(); self.cycle_summary.append(s)

    def reset(self, initial_soc=0.5):
        self.Uc = self.U_cutoff+float(np.clip(initial_soc, 0, 1))*(self.U_nom-self.U_cutoff)
        self.U1 = self.U2 = self.I = 0.0; self.sim_time = 0.0
        self.sim_data.clear(); self.cycle_summary.clear()

    def get_simulation_result(self): return _dataframe(self.sim_data)
    def get_cycle_summary(self): return _dataframe(self.cycle_summary)


class EDLC_Supercapacitor(_SecondOrderSupercapacitor):
    device_type = "DOUBLE_LAYER_SUPERCAPACITOR"
    def _set_nominal_parameters(self):
        self.C_nom, self.U_nom, self.U_cutoff = 500.0, 48.0, 24.0  #10000.0 820.0 400.0
        self.R0_nom, self.R1_nom, self.R2_nom = 0.0032, 0.03, 0.05
        self.C1_nom, self.C2_nom, self.T_ref = 30.0, 50.0, 25.0
        self.resistance_temp_coefficient, self.capacitance_temp_coefficient = 0.008, 0.0015
        self.SOH_threshold, self.cycle_nom = 80.0, 500000.0
        self.Ea, self.Rg, self.resistance_growth_at_eol = 5000.0, 8.314, 0.5
        self.charge_current_max_A = self.discharge_current_max_A = 60.0 #800.0
        self.temperature_min_C, self.temperature_low_derate_start_C = -20.0, -10.0
        self.temperature_high_derate_start_C, self.temperature_max_C = 55.0, 65.0
        self.voltage_margin_V = 20.0


class HSC_HybridSupercapacitor(_SecondOrderSupercapacitor):
    device_type = "HYBRID_SUPERCAPACITOR"
    def _set_nominal_parameters(self):
        self.C_nom, self.U_nom, self.U_cutoff = 10000.0, 3.8, 2.0
        self.R0_nom, self.R1_nom, self.R2_nom = 0.005, 0.01, 0.02
        self.C1_nom, self.C2_nom, self.T_ref = 800.0, 1200.0, 25.0
        self.resistance_temp_coefficient, self.capacitance_temp_coefficient = 0.012, 0.002
        self.SOH_threshold, self.cycle_nom = 80.0, 500000.0
        self.Ea, self.Rg, self.resistance_growth_at_eol = 6500.0, 8.314, 0.8
        self.charge_current_max_A = self.discharge_current_max_A = 200.0
        self.temperature_min_C, self.temperature_low_derate_start_C = -20.0, -10.0
        self.temperature_high_derate_start_C, self.temperature_max_C = 55.0, 65.0
        self.voltage_margin_V = 0.1

    @property
    def C_hyb_nom(self): return self.C_nom
    @C_hyb_nom.setter
    def C_hyb_nom(self, value): self.C_nom = float(value)


class BatteryPack:
    """Np条并联支路、每条Ns个串联单体；读取函数均不推进状态。"""
    current_sign_convention = "positive_discharge_negative_charge"
    def __init__(self, cell_class, Ns=3, Np=2, dev_std=0.03,
                 R_contact=0.0001, R_busbar=0.0002, seed=42):
        if Ns <= 0 or Np <= 0: raise ValueError("Ns和Np必须为正。")
        self.Ns, self.Np, self.R_contact, self.R_busbar = Ns, Np, R_contact, R_busbar
        rng = np.random.default_rng(seed)
        self.branch_cells = [[cell_class(float(rng.normal(0, dev_std)),
            float(rng.normal(0, dev_std)), float(rng.normal(0, dev_std*0.8)))
            for _ in range(Ns)] for _ in range(Np)]
        self.dt, self.T, self.pack_I, self.sim_time = self.branch_cells[0][0].dt, 25.0, 0.0, 0.0
        self.process_data, self.cycle_summary, self.pack_cycle = [], [], 0

    def set_temp(self, T):
        self.T = float(T)
        for b in self.branch_cells:
            for c in b: c.T = self.T

    def _equivalent(self, branch):
        source, resistance = 0.0, (self.Ns-1)*self.R_contact+self.R_busbar
        for c in branch:
            source += c.Uc-c.U1-c.U2; resistance += c._parameters(self.T)[0]
        return source, max(resistance, 1e-12)

    def current_distribute(self, current=None):
        current = self.pack_I if current is None else float(current)
        eq = [self._equivalent(b) for b in self.branch_cells]
        g = sum(1/r for _, r in eq)
        voltage = (sum(e/r for e, r in eq)-current)/g
        return [(e-voltage)/r for e, r in eq]

    def calc_branch_impedance(self, branch):
        return sum(c.get_state(self.T)["eir_ohm"] for c in branch)+(self.Ns-1)*self.R_contact+self.R_busbar

    def get_pack_total_voltage(self):
        return float(np.mean([sum(c._terminal_voltage(0, T=self.T) for c in b) for b in self.branch_cells]))

    def get_state(self, temperature_C=None):
        T = self.T if temperature_C is None else float(temperature_C)
        states = [c.get_state(T) for b in self.branch_cells for c in b]
        voltage = self.get_pack_total_voltage()
        return {"time_s": self.sim_time, "voltage_V": voltage, "stack_voltage_V": voltage,
                "soc": float(np.mean([s["soc"] for s in states])),
                "soe": float(np.mean([s["soe"] for s in states])),
                "soh": float(np.min([s["soh"] for s in states])),
                "eir_ohm": float(1/sum(1/self.calc_branch_impedance(b) for b in self.branch_cells)),
                "temperature_C": T, "cell_states": states}

    def get_current_limits(self, dt_s=None, temperature_C=None):
        dt, T = self.dt if dt_s is None else dt_s, self.T if temperature_C is None else temperature_C
        charge = discharge = 0.0; tf = sf = 1.0
        for b in self.branch_cells:
            limits = [c.get_current_limits(dt, T) for c in b]
            charge += min(x["charge_A"] for x in limits); discharge += min(x["discharge_A"] for x in limits)
            tf = min([tf]+[x["temperature_factor"] for x in limits]); sf = min([sf]+[x["soh_factor"] for x in limits])
        return {"charge_A": charge, "discharge_A": discharge,
                "charge_reason": "cell_boundary", "discharge_reason": "cell_boundary",
                "temperature_factor": tf, "soh_factor": sf}

    def _predict_voltage(self, I, dt, T):
        currents = self.current_distribute(I); voltages = []
        for b, bi in zip(self.branch_cells, currents):
            voltages.append(sum(c.predict_step(bi, dt, T)["voltage_V"] for c in b))
        return float(np.mean(voltages))

    def get_power_limits(self, dt_s=None, temperature_C=None):
        dt, T = self.dt if dt_s is None else dt_s, self.T if temperature_C is None else temperature_C
        limits = self.get_current_limits(dt, T); ca, da = limits["charge_A"], limits["discharge_A"]
        cv, dv = self._predict_voltage(-ca, dt, T), self._predict_voltage(da, dt, T)
        return {**limits, "charge_W": min(0.0, -ca*cv), "charge_magnitude_W": max(0.0, ca*cv),
                "discharge_W": max(0.0, da*dv), "charge_voltage_V": cv,
                "discharge_voltage_V": dv, "power_reference": "supercapacitor_pack_dc_terminal"}

    def step(self, current_A, dt_s=None, temperature_C=None, current_limits=None):
        dt, T = self.dt if dt_s is None else float(dt_s), self.T if temperature_C is None else float(temperature_C)
        limits = self.get_current_limits(dt, T) if current_limits is None else dict(current_limits)
        requested = float(current_A); actual = float(np.clip(requested, -limits["charge_A"], limits["discharge_A"]))
        currents = self.current_distribute(actual); outputs = []
        for b, bi in zip(self.branch_cells, currents):
            outputs.extend(c.step(bi, dt, T) for c in b)
        self.pack_I, self.T, self.sim_time = actual, T, self.sim_time+dt
        voltage = float(np.mean([sum(o.voltage_V for o in outputs[i*self.Ns:(i+1)*self.Ns]) for i in range(self.Np)]))
        s = self.get_state(T); limited = not np.isclose(requested, actual)
        out = SupercapacitorOutput(self.sim_time, dt, requested, actual, voltage, voltage*actual,
            s["soc"], s["soe"], s["soh"], 1-s["soh"], max(o.power_fade for o in outputs),
            s["eir_ohm"], T, limits["charge_A"], limits["discharge_A"], limited,
            "cell_boundary" if limited else "未限流", any(o.fault for o in outputs),
            [v for o in outputs for v in o.violations], {"branch_currents_A": currents, "cell_outputs": outputs})
        self.process_data.append({"时间(s)": self.sim_time, "包总电压(V)": voltage,
            "包总电流(A)": actual, "包总功率(W)": out.power_W, "包SOH(%)": 100*out.soh})
        return out

    def step_power(self, power_W, dt_s=None, temperature_C=None, power_limits=None):
        dt, T = self.dt if dt_s is None else float(dt_s), self.T if temperature_C is None else float(temperature_C)
        limits = self.get_power_limits(dt, T) if power_limits is None else dict(power_limits)
        clipped = float(np.clip(power_W, -limits["charge_magnitude_W"], limits["discharge_W"]))
        sign = 1 if clipped > 0 else -1; low = 0.0
        high = limits["discharge_A"] if sign > 0 else limits["charge_A"]
        for _ in range(28):
            mid = (low+high)/2; p = abs(sign*mid*self._predict_voltage(sign*mid, dt, T))
            if p < abs(clipped): low = mid
            else: high = mid
        out = self.step(0.0 if abs(clipped)<1e-12 else sign*(low+high)/2, dt, T, limits)
        if not np.isclose(power_W, clipped): out.limited, out.limit_reason = True, "pack_power_boundary"
        out.raw.update({"requested_power_W": power_W, "clipped_power_W": clipped}); return out

    def step_calc(self): return self.step(self.pack_I, self.dt, self.T)
    def rest(self, rest_time=10):
        for _ in range(int(math.ceil(rest_time/self.dt))): self.step(0, self.dt, self.T)
    def get_process_df(self): return _dataframe(self.process_data)
    def get_cycle_df(self): return _dataframe(self.cycle_summary)


class BatteryCluster:
    """Ncp条并联支路、每条Ncs个串联Pack的簇级模型。"""
    def __init__(self, cell_class, Ncs=2, Ncp=2, Ns=3, Np=2, dev_std=0.03,
                 R_contact=1e-4, R_bus=2e-4, R_cable=0.001, seed=42):
        rng = np.random.default_rng(seed); self.Ncs, self.Ncp, self.R_cable = Ncs, Ncp, R_cable
        self.pack_branches = [[BatteryPack(cell_class, Ns, Np, dev_std, R_contact, R_bus,
            int(rng.integers(0, 2**31-1))) for _ in range(Ncs)] for _ in range(Ncp)]
        self.dt, self.T, self.cluster_I, self.sim_time = self.pack_branches[0][0].dt, 25.0, 0.0, 0.0
        self.process_data = []

    def set_temp(self, T):
        self.T = float(T)
        for b in self.pack_branches:
            for p in b: p.set_temp(T)

    def _equivalent(self, branch):
        return (sum(p.get_state()["voltage_V"] for p in branch),
                sum(p.get_state()["eir_ohm"] for p in branch)+self.R_cable)

    def current_distribute(self, current=None):
        current = self.cluster_I if current is None else float(current)
        eq = [self._equivalent(b) for b in self.pack_branches]; g = sum(1/r for _, r in eq)
        voltage = (sum(e/r for e, r in eq)-current)/g
        return [(e-voltage)/r for e, r in eq]

    def get_cluster_total_voltage(self):
        return float(np.mean([sum(p.get_pack_total_voltage() for p in b) for b in self.pack_branches]))

    def get_state(self):
        states = [p.get_state() for b in self.pack_branches for p in b]; voltage = self.get_cluster_total_voltage()
        return {"time_s": self.sim_time, "voltage_V": voltage, "stack_voltage_V": voltage,
                "soc": float(np.mean([s["soc"] for s in states])),
                "soe": float(np.mean([s["soe"] for s in states])),
                "soh": float(np.min([s["soh"] for s in states])), "pack_states": states}

    def step(self, current_A, dt_s=None, temperature_C=None):
        dt, T = self.dt if dt_s is None else float(dt_s), self.T if temperature_C is None else float(temperature_C)
        currents = self.current_distribute(current_A); outputs = []
        for b, bi in zip(self.pack_branches, currents): outputs.extend(p.step(bi, dt, T) for p in b)
        self.cluster_I, self.T, self.sim_time = float(current_A), T, self.sim_time+dt
        voltage = self.get_cluster_total_voltage(); power = voltage*self.cluster_I
        result = {"time_s": self.sim_time, "voltage_V": voltage, "current_A": self.cluster_I,
                  "power_W": power, "branch_currents_A": currents, "pack_outputs": outputs}
        self.process_data.append({"时间(s)": self.sim_time, "簇总电压(V)": voltage,
            "簇总电流(A)": self.cluster_I, "簇总功率(W)": power}); return result

    def step_calc(self): return self.step(self.cluster_I, self.dt, self.T)
    def rest(self, rest_time=10):
        for _ in range(int(math.ceil(rest_time/self.dt))): self.step(0, self.dt, self.T)
    def get_process_df(self): return _dataframe(self.process_data)


__all__ = ["SupercapacitorOutput", "EDLC_Supercapacitor",
           "HSC_HybridSupercapacitor", "BatteryPack", "BatteryCluster"]
