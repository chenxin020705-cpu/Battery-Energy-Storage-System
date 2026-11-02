r"""
bms_layer.py
============
BMS 约束层: 在**不修改物理层 battery_pack** 的前提下, 通过"输入限幅 + 输出解释"
把电堆约束在指定的工作SOC区间内运行。

分层
----
    物理层 battery_pack : 状态 = 物理SOC(内部直接查 OCV 表); 输入 = 电堆串电流 I(A,
                          放电为正); 输出 = V_term / 逐单体 SOC。本层不动它。
    BMS 约束层(本模块) :
        ① 映射  物理SOC <-> 工作SOC(显示SOC)
        ② 限幅  对电流指令逐约束限幅(SOC 窗口 / 电流上限), 前验
        ③ 校验  逐步后验检查(单体电压 / 温度), 记录越界标志
    闭环调用 battery_pack 的在线接口 reset()/state()/step()。

SOC 映射
--------
    物理SOC ∈ [soc_lo, soc_hi] (默认 [0.104, 0.575], 来自 50Ah/480Ah 与 276Ah/480Ah)
        工作SOC = (soc_phy - soc_lo) / (soc_hi - soc_lo)      ∈ [0, 1]
        soc_phy = soc_lo + (soc_hi - soc_lo) * 工作SOC

约束
----
    前验(限制电流):
        * 电流上限 |I| <= i_max
        * SOC 窗口: 放电不得使 soc_phy < soc_lo, 充电不得使 soc_phy > soc_hi;
          按"本步 SOC 变化量 = I*dt/(3600*Q_stack)"预测限流, 避免离散步越界。
        * 可选 soc_taper: 在边界前该带宽内线性降流(默认 0 = 关闭, 硬限)。
    后验(仅记录, 不回溯):
        * 单体过/欠压 v_cell_max / v_cell_min
        * 温度 t_min / t_max

欠压故障状态机(可选, v_uv_cut 非 None 时启用)
------------------------------------------------
    NORMAL  --触发-->  FAULT(欠压锁定)  --恢复-->  NORMAL
    触发 : min(V_group) <= v_uv_cut (带载端电压), 连续 uv_confirm 步满足;
           默认仅放电(I>0)时判(uv_discharge_only)
    保持 : 电流强制为 i_fault(默认 0, 充放全禁); 对外显示SOC 强制为 soc_fault(默认 0)
    恢复 : 持续时长 >= t_hold 且 min(V_group) >= v_recover

高端钳位(可选, 锁存式, v_ov_cut 非 None 时启用)
---------------------------------------------
    触发 : max(V_group) >= v_ov_cut (带载端电压) -> 对外显示 SOC 置为 soc_ov(默认 1.0,
           即 100%); **仅改显示值**, 不改物理层、不封锁电流。
    锁存 : 一旦触发即**保持锁存**, 不回退到阈前的映射曲线。此后以触发时刻的物理SOC 为
           锚点, 显示SOC 从 soc_ov 起按**同一 SOC 增量**继续演化(等价于从 100% 起的安时
           积分), 即 显示SOC = soc_ov + (soc_phy - soc_phy_锚)/ (soc_hi - soc_lo)。
    说明 : 这是"充电截止电压触发置满 + 顶部重标"的机制(与低端钳位对称)。实测数据并未
           支持该机制存在(高端更接近"静置期OCV重标 + 顶部钳位"), 故默认关闭, 仅作
           占位/对比开关。

显示容量(可选, 与SOC 解耦)
--------------------------
    容量作为独立于"显示SOC 映射"的量(cap_total_ah / cap_dis0_ah 均为 None 时不输出):
        可放电量(t) = cap_dis0_ah + (soc_phy(t) - soc_phy(0)) * q_stack_ah
        可充电量(t) = cap_total_ah - 可放电量(t)
    高压触发(与显示SOC 锁存同步, 首次 max(V_group) >= v_ov_cut)时重整定:
        可放电量 -> cap_ov_dis_ah(默认 cap_total_ah), 可充电量 -> cap_ov_chg_ah(默认 0),
    之后按安时变化继续演化。该口径不经过 soc_lo/soc_hi, 避免 SOC 映射口径污染容量。

静置重估(可选, rest_reest=True 时启用)
--------------------------------------
    机制 : 长放电之后出现持续静置(|I| < i_rest_eps 且时长 >= t_rest_min, 且静置前
           rest_lookback_s 内的净放电 >= rest_discharge_ah_min)时, 显示SOC 与显示容量
           向各自目标值(rest_soc_target / rest_cap_dis|chg_target_ah)按时间常数 tau_rest
           一阶弛豫, 模拟 BMS 在静置期按 OCV/滤波重新估算的过程(tau_rest<=0 即阶跃)。
    说明 : 真实重估算法未知, 这里以"目标值 + 时间常数"参数化; 默认关闭。

用法
----
    from battery_pack import battery_pack
    from bms_layer import BMSConstraint

    pack = battery_pack(216, N_p=12, dt=5.0, memory_limit=1500, store_history=False)
    pack.set_cell_parameters(R0=1.4e-3, R1=8e-4, C1=5e4, alpha1=0.9,
                             R2=1.2e-3, C2=2.5e5, alpha2=0.7,
                             Q=40.0, soc0=0.575, eta=1.0)

    bms = BMSConstraint(pack, soc_lo=0.104, soc_hi=0.575, q_stack_ah=480.0, i_max=480.0)
    res = bms.run(I_ref, T=25.0)

    res["SOC_phy"]   # (n,) 物理SOC
    res["SOC_work"]  # (n,) 工作SOC(显示SOC)
    res["I"]         # (n,) 实际下发电流(已限幅)
    res["I_ref"]     # (n,) 期望电流
    res["V_term"]    # (n,) 电堆端电压
    res["reasons"]   # list[list[str]], 每步命中的约束
"""

import math

import numpy as np


class BMSConstraint:
    """物理SOC -> 工作SOC 映射 + 电流限幅 + 后验校验的 BMS 约束层。

    Parameters
    ----------
    pack : battery_pack
        物理层模型(须已 set_cell_parameters), 且必须用**有限的 memory_limit** 构造
        (在线接口 state()/step() 的前提)。
    soc_lo, soc_hi : float
        物理SOC工作区间端点(比例), 默认 0.104 / 0.575。
    q_stack_ah : float
        电堆满容量 (Ah), 用于把电流换算成单步 SOC 变化; 默认 480.0。
    i_max : float or None
        电堆电流幅值上限 (A); None 表示不限。
    soc_taper : float
        边界前线性降流带宽(比例); 0 表示硬限。默认 0.0。
    v_cell_max, v_cell_min : float or None
        单体电压上/下限 (V), 仅后验记录; None 表示不检查。
    t_min, t_max : float or None
        温度下限/上限 (℃), 仅后验记录; None 表示不检查。
    v_uv_cut : float or None
        欠压触发阈值 (V), 判据为 min(V_group)(带载端电压); None = 关闭故障状态机。
    v_recover : float or None
        故障恢复阈值 (V), 判据为 min(V_group); None 表示恢复只看 t_hold。
    t_hold : float
        故障最短保持时长 (s), 默认 0.0。
    i_fault : float
        故障期间允许的电流 (A), 默认 0.0(充放全禁)。
    soc_fault : float
        故障期间对外显示的工作SOC, 默认 0.0(强制归零)。
    uv_confirm : int
        触发去抖: 连续满足阈值的步数, 默认 1。
    uv_discharge_only : bool
        是否仅在放电(I>0)时判触发, 默认 True。
    v_ov_cut : float or None
        高端钳位触发阈值 (V), 判据为 max(V_group)(带载端电压); None = 关闭高端钳位。
        触发后**锁存**: 显示SOC 置 soc_ov 并以此为锚点继续演化, 不回退。
    soc_ov : float
        高端钳位触发时显示SOC 的锚值, 默认 1.0(即 100%)。
    cap_total_ah : float or None
        显示容量总量 (Ah), 与 soc_lo/soc_hi 解耦; None = 不输出显示容量。
    cap_dis0_ah : float or None
        起始显示可放电量 (Ah, 整定锚点), 需与 cap_total_ah 同时给出。
    cap_ov_dis_ah : float or None
        高压触发时显示可放电量的重整定值; None = 取 cap_total_ah。
    cap_ov_chg_ah : float
        高压触发时显示可充电量的重整定值, 默认 0.0。
    rest_reest : bool
        是否启用静置重估, 默认 False。
    i_rest_eps : float
        静置判定电流阈值 (A), |I| < i_rest_eps 视为静置, 默认 2.0。
    t_rest_min : float
        触发重估所需的最短连续静置时长 (s), 默认 300.0。
    rest_lookback_s : float
        判定"长放电"的回看窗口 (s), 默认 1800.0。
    rest_discharge_ah_min : float or None
        回看窗口内净放电量门槛 (Ah); None = 不要求前置放电。
    tau_rest : float
        重估弛豫时间常数 (s); <=0 表示阶跃, 默认 600.0。
    rest_soc_target : float or None
        静置重估的显示SOC 目标值 (0~1); None = 不重估SOC。
    rest_cap_dis_target_ah, rest_cap_chg_target_ah : float or None
        静置重估的显示可放/可充电量目标值 (Ah); None = 不重估对应容量。
    q_disp_ah : float or None
        显示SOC 的独立积分容量基准 (Ah)。给出时显示SOC 不再由物理SOC 映射/锁存得到,
        而是按 d(S_disp)/dt = -I/(3600*q_disp_ah) 独立积分(高压触发/故障仍可重锚定);
        None = 沿用原机制(由物理SOC 映射 + 高压锁存)。
    soc_disp0 : float or None
        独立积分时显示SOC 的初值(0~1); None = 取 to_work_soc(soc0)。

    静置重估机制(rest_reest=True, 默认关闭)
    --------------------------------------
    逻辑全部在本类 run() 内实现(运行文件只需传参), 只改显示层, 不动物理层/ECM:
      1) 识别连续静置段: |I| < i_rest_eps 且时长 >= t_rest_min;
      2) 校验前置长放电: 静置前 rest_lookback_s 内净放电 >= rest_discharge_ah_min;
         故障封锁(FAULT)段不视为静置;
      3) 从 t_rest_min 起, 显示SOC/容量向目标值按 tau_rest 一阶弛豫;
      4) 段末把重估偏移传播到后续所有步, 恢复充电后从重估值连续演化(不回跳);
         仅 soc_fault(欠压强制锁存)允许显示状态阶跃。
    """

    def __init__(self, pack, soc_lo=0.104, soc_hi=0.575, q_stack_ah=480.0,
                 i_max=None, soc_taper=0.0,
                 v_cell_max=None, v_cell_min=None, t_min=None, t_max=None,
                 v_uv_cut=None, v_recover=None, t_hold=0.0, i_fault=0.0,
                 soc_fault=0.0, uv_confirm=1, uv_discharge_only=True,
                 v_ov_cut=None, soc_ov=1.0,
                 cap_total_ah=None, cap_dis0_ah=None,
                 cap_ov_dis_ah=None, cap_ov_chg_ah=0.0,
                 rest_reest=False, i_rest_eps=2.0, t_rest_min=300.0,
                 rest_lookback_s=1800.0, rest_discharge_ah_min=None,
                 tau_rest=600.0, rest_soc_target=None,
                 rest_cap_dis_target_ah=None, rest_cap_chg_target_ah=None,
                 q_disp_ah=None, soc_disp0=None):
        if not (0.0 <= soc_lo < soc_hi <= 1.0):
            raise ValueError("要求 0 <= soc_lo < soc_hi <= 1")
        self.pack = pack
        self.soc_lo = float(soc_lo)
        self.soc_hi = float(soc_hi)
        self.q_stack_ah = float(q_stack_ah)
        self.i_max = None if i_max is None else float(i_max)
        self.soc_taper = float(soc_taper)
        self.v_cell_max = v_cell_max
        self.v_cell_min = v_cell_min
        self.t_min = t_min
        self.t_max = t_max
        # 欠压故障状态机
        self.v_uv_cut = None if v_uv_cut is None else float(v_uv_cut)
        self.v_recover = None if v_recover is None else float(v_recover)
        self.t_hold = float(t_hold)
        self.i_fault = float(i_fault)
        self.soc_fault = float(soc_fault)
        self.uv_confirm = int(uv_confirm)
        self.uv_discharge_only = bool(uv_discharge_only)
        # 高端钳位
        self.v_ov_cut = None if v_ov_cut is None else float(v_ov_cut)
        self.soc_ov = float(soc_ov)
        # 显示容量(与SOC 解耦)
        self.cap_total_ah = None if cap_total_ah is None else float(cap_total_ah)
        self.cap_dis0_ah = None if cap_dis0_ah is None else float(cap_dis0_ah)
        self.cap_ov_dis_ah = None if cap_ov_dis_ah is None else float(cap_ov_dis_ah)
        self.cap_ov_chg_ah = float(cap_ov_chg_ah)
        # 静置重估(可选, 默认关闭; 机制见类文档)
        #   rest_reest 开关 / i_rest_eps 静置电流阈值 / t_rest_min 最短静置时长 /
        #   rest_lookback_s + rest_discharge_ah_min 前置长放电判定 /
        #   tau_rest 一阶弛豫时间常数 / rest_*_target 重估目标值
        self.rest_reest = bool(rest_reest)
        self.i_rest_eps = float(i_rest_eps)
        self.t_rest_min = float(t_rest_min)
        self.rest_lookback_s = float(rest_lookback_s)
        self.rest_discharge_ah_min = (None if rest_discharge_ah_min is None
                                      else float(rest_discharge_ah_min))
        self.tau_rest = float(tau_rest)
        self.rest_soc_target = (None if rest_soc_target is None
                                else float(rest_soc_target))
        self.rest_cap_dis_target_ah = (None if rest_cap_dis_target_ah is None
                                       else float(rest_cap_dis_target_ah))
        self.rest_cap_chg_target_ah = (None if rest_cap_chg_target_ah is None
                                       else float(rest_cap_chg_target_ah))
        # 显示SOC 独立积分(方案3): q_disp_ah 为显示容量基准
        self.q_disp_ah = None if q_disp_ah is None else float(q_disp_ah)
        self.soc_disp0 = None if soc_disp0 is None else float(soc_disp0)

    # ------------------------------------------------------------------ #
    # ① 映射
    # ------------------------------------------------------------------ #
    @property
    def window_ah(self):
        """工作窗口容量 (Ah) = (soc_hi - soc_lo) * q_stack_ah。"""
        return (self.soc_hi - self.soc_lo) * self.q_stack_ah

    def to_work_soc(self, soc_phy):
        """物理SOC -> 工作SOC, 结果裁到 [0, 1]。"""
        w = (np.asarray(soc_phy, dtype=float) - self.soc_lo) / (self.soc_hi - self.soc_lo)
        return np.clip(w, 0.0, 1.0)

    def to_phy_soc(self, soc_work):
        """工作SOC -> 物理SOC, 输入先裁到 [0, 1]。"""
        w = np.clip(np.asarray(soc_work, dtype=float), 0.0, 1.0)
        return self.soc_lo + (self.soc_hi - self.soc_lo) * w

    # ------------------------------------------------------------------ #
    # ② 前验限幅
    # ------------------------------------------------------------------ #
    def limit(self, I_cmd, soc_phy):
        """对电流指令限幅, 返回 (允许电流 I, 命中约束列表)。

        电流符号: 放电为正、充电为负。
        """
        I = float(I_cmd)
        reasons = []

        # (1) 电流上限
        if self.i_max is not None and abs(I) > self.i_max:
            I = math.copysign(self.i_max, I)
            reasons.append("I_limit")

        # (2) SOC 窗口(可选 taper + 预测式硬限, 保证单步不越界)
        if I > 0.0:                                     # 放电
            if self.soc_taper > 0.0 and soc_phy < self.soc_lo + self.soc_taper:
                f = max(0.0, (soc_phy - self.soc_lo) / self.soc_taper)
                if f < 1.0:
                    I *= f
                    reasons.append("SOC_lo_taper")
            room = max(soc_phy - self.soc_lo, 0.0)
            i_step = room * 3600.0 * self.q_stack_ah / self.pack.dt
            if I > i_step:
                I = i_step
                reasons.append("SOC_lo")
        elif I < 0.0:                                   # 充电
            if self.soc_taper > 0.0 and soc_phy > self.soc_hi - self.soc_taper:
                f = max(0.0, (self.soc_hi - soc_phy) / self.soc_taper)
                if f < 1.0:
                    I *= f
                    reasons.append("SOC_hi_taper")
            room = max(self.soc_hi - soc_phy, 0.0)
            i_step = room * 3600.0 * self.q_stack_ah / self.pack.dt
            if -I > i_step:
                I = -i_step
                reasons.append("SOC_hi")

        return I, reasons

    # ------------------------------------------------------------------ #
    # ③ 闭环运行
    # ------------------------------------------------------------------ #
    def run(self, I_ref, T=25.0, soc0=None):
        """按参考电流工况闭环推进: 读状态 -> 限幅 -> 单步, 逐点记录。

        Parameters
        ----------
        I_ref : array_like
            期望的电堆电流时序 (A), 放电为正、充电为负。
        T : float, 默认 25.0
            温度 (℃); 物理层按标量常数处理。
        soc0 : float, optional
            起始物理SOC; 默认取 soc_hi(从工作上限起步)。

        Returns
        -------
        dict
            "t"          : (n,) 时间 (s)
            "SOC_phy"    : (n,) 物理SOC(每步**步末**值)
            "SOC_work"   : (n,) 对外显示的工作SOC(0~1); 故障期间强制为 soc_fault;
                           默认由物理SOC 映射; 给出 q_disp_ah 时按该容量基准独立安时积分;
                           高端钳位触发后锁存为 soc_ov 并以此为锚继续演化(不回退);
                           静置重估启用时在长放电后的静置段内弛豫到 rest_soc_target,
                           段后偏移延续(不回跳)
            "I"          : (n,) 实际下发电流 (A, 已限幅/故障封锁)
            "I_ref"      : (n,) 期望电流 (A)
            "V_term"     : (n,) 电堆端电压 (V)
            "V_cell_max" : (n,) 步末最高单体电压 (V) = max(各并联组端电压)
            "V_cell_min" : (n,) 步末最低单体电压 (V)
            "fault"      : (n,) bool, 该步是否处于欠压故障(FAULT)状态
            "reasons"    : list[list[str]] 每步命中的约束名(故障步含 "FAULT";
                           高端钳位触发步含 "OV_clamp", 锁存保持步含 "OV_latch")
            "Cap_dis"    : (n,) 显示可放电量 (Ah); cap_total_ah/cap_dis0_ah 未给出时为 None;
                           静置重估启用时同步弛豫到 rest_cap_dis_target_ah
            "Cap_chg"    : (n,) 显示可充电量 (Ah); 同上(目标 rest_cap_chg_target_ah)
        """
        I_ref = np.asarray(I_ref, dtype=float).ravel()
        n = I_ref.size
        start = self.soc_hi if soc0 is None else float(soc0)

        self.pack.reset(soc0=start)

        t = np.zeros(n)
        soc_phy = np.zeros(n)
        soc_work = np.zeros(n)
        I_act = np.zeros(n)
        V_term = np.zeros(n)
        Vcmax = np.full(n, np.nan)
        Vcmin = np.full(n, np.nan)
        fault = np.zeros(n, dtype=bool)
        reasons = [[] for _ in range(n)]

        in_fault = False        # 故障状态机当前状态
        t_fault = 0.0           # 进入故障的时刻 (s)
        uv_cnt = 0              # 触发去抖计数
        ov_latched = False      # 高端钳位是否已锁存
        ov_anchor_phy = 0.0     # 锁存时刻的物理SOC(显示SOC = soc_ov 的锚点)
        k_ov = None             # 高端钳位首次触发的步索引(用于容量重整定)
        # 显示SOC 独立积分(方案3): 以 q_disp_ah 为容量基准
        disp_ind = self.q_disp_ah is not None
        s_disp = ((self.soc_disp0 if self.soc_disp0 is not None
                   else float(self.to_work_soc(start))) if disp_ind else 0.0)

        for k in range(n):
            st = self.pack.state()
            sp = float(np.mean(st["SOC"]))              # 物理SOC(堆=单体均值)

            was_fault = in_fault
            if was_fault:
                I_a, rs = self.i_fault, ["FAULT"]            # 故障: 电流封锁
            else:
                I_a, rs = self.limit(float(I_ref[k]), sp)    # ② 限幅

            out = self.pack.step(I_a, T=T)              # 物理层单步推进

            vg = out["V_group"]
            vcmax = float(np.max(vg))
            vcmin = float(np.min(vg))

            # ③ 后验校验(仅记录)
            if self.v_cell_max is not None and vcmax > self.v_cell_max:
                rs = rs + ["V_cell_high"]
            if self.v_cell_min is not None and vcmin < self.v_cell_min:
                rs = rs + ["V_cell_low"]
            if self.t_min is not None and T < self.t_min:
                rs = rs + ["T_low"]
            if self.t_max is not None and T > self.t_max:
                rs = rs + ["T_high"]

            # ④ 欠压故障状态机(基于本步末 min(V_group) 带载端电压)
            if self.v_uv_cut is not None:
                if not in_fault:
                    trig = vcmin <= self.v_uv_cut
                    if self.uv_discharge_only and not (I_a > 0.0):
                        trig = False
                    uv_cnt = uv_cnt + 1 if trig else 0
                    if uv_cnt >= self.uv_confirm:
                        in_fault = True
                        t_fault = float(out["t"])
                        uv_cnt = 0
                else:
                    hold_ok = (float(out["t"]) - t_fault) >= self.t_hold
                    rec_ok = (self.v_recover is None) or (vcmin >= self.v_recover)
                    if hold_ok and rec_ok:
                        in_fault = False

            sp_new = float(np.mean(out["SOC"]))

            # ⑤ 显示SOC
            #    a) 默认: 由物理SOC 映射; 高压触发(soc_ov)后以该步物理SOC 为锚继续演化。
            #    b) q_disp_ah 给出时(方案3): 独立安时积分 dS=-I*dt/(3600*q_disp_ah),
            #       与物理SOC 解耦; 高压触发置 soc_ov、故障置 soc_fault 仍为重锚定。
            ov_now = (self.v_ov_cut is not None and not ov_latched
                      and vcmax >= self.v_ov_cut)
            if disp_ind:
                s_disp -= I_a * self.pack.dt / (3600.0 * self.q_disp_ah)
                if ov_now:
                    ov_latched = True
                    k_ov = k
                    s_disp = self.soc_ov
                    rs = rs + ["OV_clamp"]
                elif ov_latched:
                    rs = rs + ["OV_latch"]
                if was_fault:
                    s_disp = self.soc_fault
                    sw = self.soc_fault
                else:
                    sw = float(np.clip(s_disp, 0.0, 1.0))
            else:
                if ov_now:
                    ov_latched = True
                    ov_anchor_phy = sp_new
                    k_ov = k
                    rs = rs + ["OV_clamp"]
                elif ov_latched:
                    rs = rs + ["OV_latch"]
                if was_fault:
                    sw = self.soc_fault
                elif ov_latched:
                    sw = float(np.clip(
                        self.soc_ov + (sp_new - ov_anchor_phy) / (self.soc_hi - self.soc_lo),
                        0.0, 1.0))
                else:
                    sw = float(self.to_work_soc(sp_new))

            t[k] = out["t"]
            soc_phy[k] = sp_new
            soc_work[k] = sw
            I_act[k] = I_a
            V_term[k] = out["V_term"]
            Vcmax[k] = vcmax
            Vcmin[k] = vcmin
            fault[k] = was_fault
            reasons[k] = rs

        # ⑥ 显示容量(可选, 与SOC 解耦): 以 cap_dis0_ah 为初值, 按安时变化演化;
        #    高压触发步重整定为 (cap_ov_dis_ah, cap_ov_chg_ah), 之后继续演化。
        cap_dis = None
        cap_chg = None
        if self.cap_total_ah is not None and self.cap_dis0_ah is not None:
            d_ah = (soc_phy - soc_phy[0]) * self.q_stack_ah
            cap_dis = self.cap_dis0_ah + d_ah
            cap_chg = self.cap_total_ah - cap_dis
            if k_ov is not None:
                ov_dis = (self.cap_total_ah if self.cap_ov_dis_ah is None
                          else self.cap_ov_dis_ah)
                cap_dis[k_ov:] += ov_dis - cap_dis[k_ov]
                cap_chg[k_ov:] += self.cap_ov_chg_ah - cap_chg[k_ov]
            cap_dis = np.clip(cap_dis, 0.0, self.cap_total_ah)
            cap_chg = np.clip(cap_chg, 0.0, self.cap_total_ah)

        # ⑦ 静置重估(可选, 默认关闭; 逻辑在本类内实现, 运行文件只需传参)
        #    目的: 复现"长放电后无电流期间, BMS 重新估算显示SOC 与可充/放电量"的行为。
        #    触发: |I| < i_rest_eps 连续 >= t_rest_min, 且静置前 rest_lookback_s 内
        #          净放电 >= rest_discharge_ah_min; 欠压故障封锁(FAULT)段不算静置。
        #    弛豫: v[k] = v[k-1] + (target - v[k-1]) * (1 - exp(-dt/tau_rest)),
        #          即显示SOC/容量向各自目标值一阶收敛; tau_rest<=0 时为阶跃。
        #    衔接: 段末将重估偏移传播到后续所有步, 恢复充电后从重估值连续演化(不回跳);
        #          仅 soc_fault(欠压强制锁存)允许显示状态阶跃。
        #    说明: 真实BMS的重估算算法(OCV/Kalman)未知, 此处以"目标值+时间常数"参数化。
        if self.rest_reest:
            rest = np.abs(I_act) < self.i_rest_eps
            edges = np.diff(np.r_[False, rest, False].astype(int))
            alpha = (1.0 if self.tau_rest <= 0.0
                     else 1.0 - math.exp(-self.pack.dt / self.tau_rest))
            for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
                if (e - s + 1) * self.pack.dt < self.t_rest_min:
                    continue
                if fault[s:e + 1].any():                     # 故障封锁段不算静置
                    continue
                if self.rest_discharge_ah_min is not None:
                    a = max(0, s - int(self.rest_lookback_s / self.pack.dt))
                    net_ah = float(np.sum(I_act[a:s]) * self.pack.dt / 3600.0)
                    if net_ah < self.rest_discharge_ah_min:  # 前置放电不足 -> 不重估
                        continue
                k0 = min(s + int(self.t_rest_min / self.pack.dt), e)
                # 记录静置段末的"基准值"(未重估), 用于段后偏移传播
                _soc_e0 = soc_work[e]
                _dis_e0 = None if cap_dis is None else cap_dis[e]
                _chg_e0 = None if cap_chg is None else cap_chg[e]
                for k in range(k0 + 1, e + 1):
                    if self.rest_soc_target is not None:
                        soc_work[k] = (soc_work[k - 1]
                                       + (self.rest_soc_target - soc_work[k - 1]) * alpha)
                    if cap_dis is not None and self.rest_cap_dis_target_ah is not None:
                        cap_dis[k] = (cap_dis[k - 1]
                                      + (self.rest_cap_dis_target_ah - cap_dis[k - 1]) * alpha)
                    if cap_chg is not None and self.rest_cap_chg_target_ah is not None:
                        cap_chg[k] = (cap_chg[k - 1]
                                      + (self.rest_cap_chg_target_ah - cap_chg[k - 1]) * alpha)
                # 段末衔接: 把重估偏移传播到后续步, 恢复充电时从重估值连续演化, 不回跳
                # (仅欠压强制锁存(soc_fault)才允许显示状态阶跃)
                if self.rest_soc_target is not None:
                    soc_work[e + 1:] += soc_work[e] - _soc_e0
                if cap_dis is not None and self.rest_cap_dis_target_ah is not None:
                    cap_dis[e + 1:] += cap_dis[e] - _dis_e0
                if cap_chg is not None and self.rest_cap_chg_target_ah is not None:
                    cap_chg[e + 1:] += cap_chg[e] - _chg_e0
            soc_work = np.clip(soc_work, 0.0, 1.0)
            if cap_dis is not None:
                cap_dis = np.clip(cap_dis, 0.0, self.cap_total_ah)
                cap_chg = np.clip(cap_chg, 0.0, self.cap_total_ah)

        return {"t": t, "SOC_phy": soc_phy, "SOC_work": soc_work,
                "I": I_act, "I_ref": I_ref, "V_term": V_term,
                "V_cell_max": Vcmax, "V_cell_min": Vcmin,
                "fault": fault, "reasons": reasons,
                "Cap_dis": cap_dis, "Cap_chg": cap_chg}
