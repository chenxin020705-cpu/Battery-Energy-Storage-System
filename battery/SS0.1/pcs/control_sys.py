try:
    import control as ct
except ModuleNotFoundError:  # 长时间步稳态PCS不依赖 python-control
    ct = None
import numpy as np
import math


# ================= 标幺制 <-> 实际值 换算模块 =================
class pcs_per_unit:
    """
    PCS 标幺制基值与实际物理量的双向换算。

    基值:
        S_base = P_rated / PF = 630 kW / 0.98 = 642.86 kVA  (三相视在功率)
        f_base = 50 Hz
        U_base = 400 V   (交流线电压有效值, 即模型 U_t = 1.0 pu 对应的值)
    派生基值:
        I_base = S_base / (sqrt(3) * U_base)     交流线电流有效值 [A]
        Z_base = U_base^2 / S_base               [Ohm]

    约定: 功率(有功/无功/视在)统一以 S_base 为基值, 因此
        额定有功 630 kW <-> 0.98 pu,   1.0 pu (纯有功) <-> 642.86 kW。
    """

    def __init__(self, P_rated_kW=630.0, PF=0.98, f_base_Hz=50.0,
                 U_base_V=400.0, u_dc_min_V=580.0, u_dc_max_V=850.0):
        self.P_rated_kW = P_rated_kW
        self.PF = PF
        self.f_base_Hz = f_base_Hz
        self.U_base_V = U_base_V
        self.u_dc_min_V = u_dc_min_V
        self.u_dc_max_V = u_dc_max_V
        self.S_base_kVA = P_rated_kW / PF
        self.I_base_A = 1e3 * self.S_base_kVA / (np.sqrt(3.0) * U_base_V)
        self.Z_base_Ohm = U_base_V ** 2 / (1e3 * self.S_base_kVA)

    # ---- 频率 / 频率偏差 ----
    def f_to_pu(self, f_Hz):
        return np.asarray(f_Hz, dtype=float) / self.f_base_Hz

    def pu_to_f(self, f_pu):
        return np.asarray(f_pu, dtype=float) * self.f_base_Hz

    def df_to_pu(self, df_Hz):
        return np.asarray(df_Hz, dtype=float) / self.f_base_Hz

    def pu_to_df(self, df_pu):
        return np.asarray(df_pu, dtype=float) * self.f_base_Hz

    # ---- 功率 ----
    def p_to_pu(self, P_kW):
        return np.asarray(P_kW, dtype=float) / self.S_base_kVA

    def pu_to_p(self, P_pu):
        return np.asarray(P_pu, dtype=float) * self.S_base_kVA

    # ---- 交流电压(线电压有效值) / 电流(线电流有效值) ----
    def u_ac_to_pu(self, U_V):
        return np.asarray(U_V, dtype=float) / self.U_base_V

    def pu_to_u_ac(self, U_pu):
        return np.asarray(U_pu, dtype=float) * self.U_base_V

    def i_to_pu(self, I_A):
        return np.asarray(I_A, dtype=float) / self.I_base_A

    def pu_to_i(self, I_pu):
        return np.asarray(I_pu, dtype=float) * self.I_base_A

    # ---- 下垂/惯量增益: K [pu功率 / pu频率] <-> [kW/Hz] ----
    def gain_pu_to_kW_per_Hz(self, K_pu):
        return np.asarray(K_pu, dtype=float) * self.S_base_kVA / self.f_base_Hz

    def gain_kW_per_Hz_to_pu(self, K):
        return np.asarray(K, dtype=float) * self.f_base_Hz / self.S_base_kVA

    # ---- 直流母线约束 (两电平 VSI + SVPWM 线性调制区) ----
    def u_ac_max_pu(self, u_dc_V):
        """给定直流电压下可输出的最大交流线电压(pu)"""
        return np.asarray(u_dc_V, dtype=float) / np.sqrt(2.0) / self.U_base_V

    def u_dc_min_V_for(self, u_ac_pu):
        """维持给定交流电压(pu)所需的最低直流电压(V)"""
        return np.asarray(u_ac_pu, dtype=float) * self.U_base_V * np.sqrt(2.0)

    def i_dc_A(self, P_kW, u_dc_V):
        return np.asarray(P_kW, dtype=float) * 1e3 / np.asarray(u_dc_V, dtype=float)

    # ---- 仿真接口: 实际量 -> 标幺输入 / 标幺输出 -> 实际量 ----
    def make_inputs(self, f_Hz, P_ref_kW, u_t_V, f0_Hz=None, Q_ref_kvar=0.0):
        """生成 pcs_p_total 的输入矩阵 U = [f, f0, P_ref, Q_ref, U_t] (5 x n), 全部标幺。

        Q_ref_kvar 仅在 q_mode='PQ' 时有效; 'PF' 模式下 Q_ref 通道被忽略
        (无功参考由 P_gen*tan(acos(pf)) 生成)。
        """
        f0_Hz = self.f_base_Hz if f0_Hz is None else f0_Hz
        f, p, q, u = [np.atleast_1d(np.asarray(v, dtype=float))
                      for v in (f_Hz, P_ref_kW, Q_ref_kvar, u_t_V)]
        n = max(f.size, p.size, q.size, u.size)
        f, p, q, u = [np.broadcast_to(v, n) for v in (f, p, q, u)]
        return np.vstack([f / self.f_base_Hz,                 # f
                          np.full(n, f0_Hz / self.f_base_Hz),  # f0
                          p / self.S_base_kVA,                # P_ref
                          q / self.S_base_kVA,                # Q_ref
                          u / self.U_base_V])                 # U_t

    def outputs_to_physical(self, y_out):
        """把 pcs_p_total 的标幺输出换算为实际量。

        y = [I_d, P_gen] (旧口径) 或 [I_d, P_gen, I_q, Q_gen](当前口径),
        电流以 I_base 为基值(交流线电流有效值), 功率以 S_base 为基值。
        """
        y = np.atleast_2d(np.asarray(y_out, dtype=float))
        out = {'I_d_A': y[0] * self.I_base_A,
               'P_gen_kW': y[1] * self.S_base_kVA}
        if y.shape[0] >= 4:
            out['I_q_A'] = y[2] * self.I_base_A
            out['Q_gen_kvar'] = y[3] * self.S_base_kVA
            out['S_gen_kVA'] = np.hypot(y[1], y[3]) * self.S_base_kVA
        return out

    def summary(self):
        return {'S_base_kVA': self.S_base_kVA,
                'P_rated_kW': self.P_rated_kW,
                'P_rated_pu': self.p_to_pu(self.P_rated_kW),
                'f_base_Hz': self.f_base_Hz,
                'U_base_V': self.U_base_V,
                'I_base_A': self.I_base_A,
                'Z_base_Ohm': self.Z_base_Ohm,
                'pu1.0_power_kW': self.S_base_kVA,
                'pu1.0_current_A': self.I_base_A,
                'pu1.1_limit_kW': 1.1 * self.S_base_kVA,
                'pu1.1_limit_A': 1.1 * self.I_base_A,
                'U_ac_max_pu@850V': self.u_ac_max_pu(self.u_dc_max_V),
                'U_ac_max_pu@580V': self.u_ac_max_pu(self.u_dc_min_V),
                'I_dc_A@630kW/850V': self.i_dc_A(self.P_rated_kW, self.u_dc_max_V),
                'I_dc_A@630kW/580V': self.i_dc_A(self.P_rated_kW, self.u_dc_min_V)}


# parameter
class pcs_control_sys:

    def __init__(self, q_mode='PQ', pf=0.98, build_dynamic=True):
        self.T1 = 0.01
        self.T2 = 0.01
        self.Kp =0.2
        self.Ki = 2.0
        self.u_eps = 1e-3

        # ---- 无功通道 ----
        # q_mode: 'PQ' 直接给定无功参考; 'PF' 恒功率因数 Q_ref = P_gen*tan(acos(pf))
        self.q_mode = str(q_mode).upper()
        if self.q_mode not in ('PQ', 'PF'):
            raise ValueError("q_mode 只能是 'PQ' 或 'PF'")
        self.pf = float(pf)
        self.tanphi = math.sqrt(max(1.0 - self.pf ** 2, 0.0)) / self.pf
        self.T_qref = 0.01          # 无功参考/测量滤波 (s)
        self.K_p_iq = 0.1           # 无功 PI 比例
        self.K_i_iq = 2.0           # 无功 PI 积分
        self.I_max = 1.1            # 交流电流幅值上限 (pu)
        self.T_aw = 0.05            # 抗饱和回算时间常数 (s)

        self.dynamic_available = ct is not None
        self.p_frc = None
        self.pcs_pq_ctrl = None
        self.pcs_p_gen = None
        self.pcs_q_gen = None
        self.pcs_p_total = None
        if build_dynamic:
            if ct is None:
                raise ModuleNotFoundError(
                    "完整PCS动态模型需要 python-control；只使用调度级稳态模型时"
                    "请传 build_dynamic=False。"
                )
            self.p_frc = self._p_frc()
            self.pcs_pq_ctrl = self._pcs_pq_ctrl()
            self.pcs_p_gen = self._pcs_p_gen()
            self.pcs_q_gen = self._pcs_q_gen()
            self.pcs_p_total = self._pcs_p_total()

    def _div_outfun(self,t,x,u,params):
        P_ref,U_t1 = u[0],u[1]
        U_eps = params['u_eps']
        if abs(U_t1) < U_eps:
            U_t1 = U_eps if U_t1 >= 0 else -U_eps
        return [P_ref/ U_t1]
    
    def _product_outfun(self,t,x,u,params):
        P_ref,U_t1 = u[0],u[1]
        return [P_ref * U_t1]

    def _make_sat(self,name, in_sig,out_sig,lb,ub):
        def __sat(t,x,u,params):
            u0 = u[0]
            if u0 > ub:
                u0 = ub
            elif u0 < lb:
                u0 = lb
            return [u0]
        return ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=__sat,
            inputs=[in_sig],
            outputs=[out_sig],
            params={'lb':lb,'ub':ub},
            name=name
        )

    def _deadzone_outfcn(self,t,x,u,params):
        """
        deadzone fcn |u|<=d :0,else u-sign(u)*d
        """
        d = params['d']
        u0 = u[0]
        if abs(u0) <= d:
            return [0.0]
        elif u0 > d:
            return [u0 - d]
        else:
            return [u0 + d]

    # active
    # 单通道有功 PI(旧版; pcs_p_total 现已改用 _pcs_pq_ctrl 的双通道+抗饱和结构)
    def _pcs_active_PI(self):
        blk1 = ct.tf(
            [1],[self.T1,1],
            inputs='P_gen',
            outputs='P_1',
            name='blk1'
        )

        blk2 = ct.tf(
            [1],[self.T1,1],
            inputs='P_ref',
            outputs='P_2',
            name='blk2'
        )

        sum_blk1 = ct.summing_junction(
            inputs=['P_2','P_frc'],
            outputs=['P_3'],
            name='sum_blk1'
        )

        P_sat_blk1 = self._make_sat('P_sat_blk1','P_3','P_sat1',-1.1,1.1)

        sum_blk2 = ct.summing_junction(
            inputs=['-P_1','P_sat1'],
            outputs=['P_4'],
            name='sum_blk2'
        )

        PI_blk = ct.tf(
            [self.Kp,self.Ki],
            [1.0,0],
            inputs='P_4',
            outputs='I_1',
            name='PI_blk'
        )

        I_set_blk = self._make_sat('I_sat_blk','I_1','I_pcmd',-1.1,1.1)

        return ct.interconnect(
            [blk1, blk2, sum_blk1, P_sat_blk1,sum_blk2, PI_blk,I_set_blk],
            inplist=['P_gen','P_ref','P_frc'],
            outlist=['I_pcmd'],
            inputs=['P_gen','P_ref','P_frc'],
            outputs=['I_pcmd'],
            name='pcs_active_PI'
        )

    def _p_frc(self):
        T_dw1 = 0.1
        T_dw2 = 0.1

        P_frc_inertia_1 = ct.tf(
            [1],[self.T2,1],
            inputs = 'f',
            outputs='f1',
            name='P_frc_inertia_1'
        )

        P_frc_sum_1 = ct.summing_junction(
            inputs=['-f1','f0'],
            outputs=['delta_f'],
            name='P_ref_sum_1'
        )

        deadzone_df_1 = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._deadzone_outfcn,
            inputs='delta_f',
            outputs='delta_f_deadzone1',
            name='deadzone_df_1',
            params={'d':0.0}
        )

        sat_df_up =  self._make_sat('sat_df_up', 'delta_f_deadzone1','delta_f_sat_up',0.0,0.1)

        gain_f_up = ct.tf(
            [10.0],[1],
            inputs='delta_f_sat_up',
            outputs='P_frc_up',
            name='gain_f_up'
        )

        sat_df_down = self._make_sat('sat_df_down','delta_f_deadzone1','delta_f_sat_down', -0.1,0.0)

        gain_f_down = ct.tf(
            [10.0],[1],
            inputs='delta_f_sat_down',
            outputs='P_frc_down',
            name='gain_f_down'  
        )

        sum_P_frc_ref1 = ct.summing_junction(
            inputs=['P_frc_up','P_frc_down'],
            outputs=['P_frc_ref1'],
            name='sum_P_frc_ref1'
        )

        deadzone_df_2 = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._deadzone_outfcn,
            inputs='delta_f',
            outputs='delta_f_deadzone2',
            name='deadzone_df_2',
            params={'d':0.0}
        )

        Inertia_sup1 =ct.tf(
            [1],[self.T2,1],
            inputs=['delta_f_deadzone2'],
            outputs=['delta_f_sup1'],
            name='Inertia_sup1'
        )

        Inertia_sup2 = ct.tf(
            [T_dw1, 0],[T_dw2, 1],
            inputs=['delta_f_sup1'],
            outputs=['delta_f_sup2'],
            name='Inertia_sup2'
        )

        Inertia_sup3 = ct.tf(
            [10.0],[1],
            inputs=['delta_f_sup2'],
            outputs=['delta_f_sup3'],
            name='Inertia_sup3'
        )

        sat_df_sup = self._make_sat('sat_df_sup', 'delta_f_sup3','P_frc_ref2', -0.1,0.1)

        sum_P_frc_total = ct.summing_junction(
            inputs=['P_frc_ref1','P_frc_ref2'],
            outputs=['P_frc'],
            name='sum_P_frc_total'
        )
        return ct.interconnect(
            [P_frc_inertia_1,P_frc_sum_1,deadzone_df_1,
            sat_df_up,gain_f_up,sat_df_down,gain_f_down,
            sum_P_frc_ref1,deadzone_df_2,Inertia_sup1,
            Inertia_sup2,Inertia_sup3,sat_df_sup,sum_P_frc_total],
            inplist=['f','f0'],
            outlist=['P_frc','P_frc_ref1','P_frc_ref2','P_frc_up','P_frc_down'],
            inputs=['f','f0'],
            outputs=['P_frc','P_frc_ref1','P_frc_ref2','P_frc_up','P_frc_down'],
            name='p_frc_mod'
        )
    
    def _pcs_p_gen(self):
        pcs_p_gen_blk = ct.tf(
            [1],[self.T1,1],
            inputs = ['I_pcmd'],
            outputs = 'I_d',
            name='pcs_p_gen_blk'
        )

        p_gen = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._product_outfun,
            inputs=['I_d','U_t'],
            outputs='P_gen',
            name='p_gen'
        )
        return ct.interconnect(
            [pcs_p_gen_blk,p_gen],
            inplist=['I_pcmd','U_t'],
            outlist=['I_d','P_gen'],
            inputs=['I_pcmd','U_t'],
            outputs=['I_d','P_gen'],
            name='pcs_p_gen'
        )

    # ---- 无功参考: PQ 给定 / 恒功率因数 ----
    def _q_ref_outfun(self, t, x, u, params):
        """u = [Q_ref, P_gen]; mode=0 -> PQ(Q_ref), mode=1 -> P_gen*tan(phi)"""
        if params['mode'] == 1:
            return [u[1] * params['tanphi']]
        return [u[0]]

    # ---- 双 PI + 视在电流圆限幅(P 优先) + 抗饱和(back-calculation) ----
    @staticmethod
    def _pq_limit(x, u, p):
        """返回 (e_p, e_q, i_d_raw, i_q_raw, i_d, i_q)。

        P 优先: I_d 先限到 +-i_max, I_q 再限到 sqrt(i_max^2 - I_d^2)。
        """
        e_p = u[0] - u[1]
        e_q = u[2] - u[3]
        i_d_raw = p['Kp'] * e_p + x[0]
        i_q_raw = p['Kp_q'] * e_q + x[1]
        i_d = min(max(i_d_raw, -p['i_max']), p['i_max'])
        i_q_hi = math.sqrt(max(p['i_max'] ** 2 - i_d ** 2, 0.0))
        i_q = min(max(i_q_raw, -i_q_hi), i_q_hi)
        return e_p, e_q, i_d_raw, i_q_raw, i_d, i_q

    def _pq_current_outfcn(self, t, x, u, params):
        _, _, _, _, i_d, i_q = self._pq_limit(x, u, params)
        return [i_d, i_q]

    def _pq_current_updfcn(self, t, x, u, params):
        e_p, e_q, i_d_raw, i_q_raw, i_d, i_q = self._pq_limit(x, u, params)
        # 积分 + 抗饱和回算: 饱和时把 (限幅值 - 未限幅值)/T_aw 反馈到积分器
        d_p = params['Ki'] * e_p + (i_d - i_d_raw) / params['T_aw']
        d_q = params['Ki_q'] * e_q + (i_q - i_q_raw) / params['T_aw']
        return [d_p, d_q]

    def _pcs_pq_ctrl(self):
        """有功/无功双通道电流指令生成: 参考滤波 -> PI -> 视在电流圆限幅。"""
        p_ref_lpf = ct.tf([1], [self.T1, 1], inputs='P_ref', outputs='P_2',
                          name='pq_p_ref_lpf')
        p_fb_lpf = ct.tf([1], [self.T1, 1], inputs='P_gen', outputs='P_1',
                         name='pq_p_fb_lpf')
        p_sum = ct.summing_junction(inputs=['P_2', 'P_frc'], outputs='P_3',
                                    name='pq_p_sum')
        p_sat = self._make_sat('pq_p_sat', 'P_3', 'P_sat', -1.1, 1.1)

        q_sel = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._q_ref_outfun,
            inputs=['Q_ref', 'P_gen'],
            outputs='Q_ref_sel',
            params={'mode': (1 if self.q_mode == 'PF' else 0),
                    'tanphi': self.tanphi},
            name='pq_q_select'
        )
        q_ref_lpf = ct.tf([1], [self.T_qref, 1], inputs='Q_ref_sel',
                          outputs='Q_1', name='pq_q_ref_lpf')
        q_sat = self._make_sat('pq_q_sat', 'Q_1', 'Q_1_sat', -1.1, 1.1)
        q_fb_lpf = ct.tf([1], [self.T_qref, 1], inputs='Q_gen',
                         outputs='Q_gen_1', name='pq_q_fb_lpf')

        pi_lim = ct.NonlinearIOSystem(
            updfcn=self._pq_current_updfcn,
            outfcn=self._pq_current_outfcn,
            inputs=['P_sat', 'P_1', 'Q_1_sat', 'Q_gen_1'],
            outputs=['I_pcmd', 'I_qcmd'],
            states=['x_p', 'x_q'],
            params={'Kp': self.Kp, 'Ki': self.Ki,
                    'Kp_q': self.K_p_iq, 'Ki_q': self.K_i_iq,
                    'i_max': self.I_max, 'T_aw': self.T_aw},
            name='pq_pi_limit'
        )

        return ct.interconnect(
            [p_ref_lpf, p_fb_lpf, p_sum, p_sat, q_sel, q_ref_lpf, q_sat,
             q_fb_lpf, pi_lim],
            inplist=['P_ref', 'P_frc', 'P_gen', 'Q_ref', 'Q_gen'],
            outlist=['I_pcmd', 'I_qcmd'],
            inputs=['P_ref', 'P_frc', 'P_gen', 'Q_ref', 'Q_gen'],
            outputs=['I_pcmd', 'I_qcmd'],
            name='pcs_pq_ctrl'
        )

    def _pcs_q_gen(self):
        q_lag = ct.tf(
            [1], [self.T1, 1],
            inputs='I_qcmd',
            outputs='I_q',
            name='pcs_q_gen_blk'
        )
        q_gen = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._product_outfun,
            inputs=['I_q', 'U_t'],
            outputs='Q_gen',
            name='q_gen'
        )
        return ct.interconnect(
            [q_lag, q_gen],
            inplist=['I_qcmd', 'U_t'],
            outlist=['I_q', 'Q_gen'],
            inputs=['I_qcmd', 'U_t'],
            outputs=['I_q', 'Q_gen'],
            name='pcs_q_gen'
        )

    def _pcs_p_total(self):
        """顶层模型: 输入 [f, f0, P_ref, Q_ref, U_t], 输出 [I_d, P_gen, I_q, Q_gen]。"""
        return ct.interconnect(
            [self.pcs_pq_ctrl, self.pcs_p_gen, self.pcs_q_gen, self.p_frc],
            inplist=['f', 'f0', 'P_ref', 'Q_ref', 'U_t'],
            outlist=['I_d', 'P_gen', 'I_q', 'Q_gen'],
            inputs=['f', 'f0', 'P_ref', 'Q_ref', 'U_t'],
            outputs=['I_d', 'P_gen', 'I_q', 'Q_gen'],
            name='pcs_p_total'
        )

    def steady_power_pu(self, f_pu, f0_pu, p_ref_pu):
        """稳态有功出力（标幺）：一次调频 + 参考限幅，供长时窗联合仿真使用。

        等价于 `pcs_p_total` 在恒定输入下的稳态：
            P_frc = 10 * clip(f0 - f, -0.1, 0.1)      # 10% 下垂
            P_gen = clip(P_ref + P_frc, -1.1, 1.1)
        忽略 0.01 s 电流/参考滤波与 0.1 s 高通暂态（相对 5 s 宏步可忽略）。

        Returns
        -------
        (P_gen_pu, P_frc_pu)
        """
        df = float(f0_pu) - float(f_pu)
        p_frc = 10.0 * min(max(df, -0.1), 0.1)
        p_gen = min(max(float(p_ref_pu) + p_frc, -1.1), 1.1)
        return p_gen, p_frc

    # ---- 旧版单通道无功模块(保留作参考; 当前 pcs_p_total 使用 _pcs_pq_ctrl) ----
    # reactive
    def _pcs_reactive_PI(self):
        self.T_qref = 0.01
        self.K_p_iq = 0.1
        self.K_i_iq = 2.0
        blk1 = ct.tf(
            [1],[self.T_qref,1],
            inputs='Q_ref',
            outputs='Q_1',
            name='blk1'
        )

        sat_blk1 = self._make_sat('sat_blk1','Q_1','Q_1_sat',-1.1,1.1)

        blk2 = ct.tf(
            [1],[self.T_qref,1],
            inputs='Q_gen',
            outputs='Q_gen_1',
            name='blk2'
        )

        sum_blk1 = ct.summing_junction(
            inputs=['Q_1_sat','-Q_gen_1'],
            outputs=['Q_2'],
            name='sum_blk1'
        )

        PI_blk1 = ct.tf(
            [self.K_p_iq,self.K_i_iq],[1.0,0.0],
            inputs='Q_2',
            outputs='I_q1',
            name='PI_blk1'
        )
        sat_blk2 = self._make_sat('sat_blk2','I_q1','I_qcmd',-1.1,1.1)
        
        return ct.interconnect(
            [blk1,sat_blk1,blk2,sum_blk1,PI_blk1,sat_blk2],
            inplist=['Q_ref','Q_gen'],
            outlist=['I_qcmd'],
            inputs=['Q_ref','Q_gen'],
            outputs=['I_qcmd'],
            name='pcs_reactive_PI'
        )
    
    def _pcs_reactive_const_factor(self):

        self.T_rP = 0.01
        self.K_p_iq = 0.1
        self.K_i_iq = 2.0
        self.const_factor = 0.98
        blk1 = ct.tf(
            [1],[self.T_rP,1],
            inputs='P_gen',
            outputs='P_gen_1',
            name='blk1'
        )
        blk2 = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._product_outfun,
            inputs=['P_gen_1','factor'],
            outputs='Q_gen_ref',
            name='blk2_const_factor'
        )
        sat_blk1 = self._make_sat('sat_blk1','Q_gen_ref','Q_sat',-1.1,1.1)

        blk3 = ct.tf(
            [1],[self.T1,1],
            inputs='Q_gen',
            outputs='Q_1',
            name='blk3'
        )

        sum_blk1 = ct.summing_junction(
            inputs=['Q_sat','-Q_1'],
            outputs=['Q_2'],
            name = 'sum_blk1'
        )
        
        PI_blk = ct.tf(
            [self.K_p_iq,self.K_i_iq],[1.0,0.0],
            inputs='Q_2',
            outputs='I_q1',
            name='PI_blk'
        )
        sat_blk2 = self._make_sat('sat_blk2','I_q1','I_qcmd',-1.1,1.1)

        return ct.interconnect(
            [blk1,blk2,sat_blk1,blk3,sum_blk1,PI_blk,sat_blk2],
            inplist=['P_gen','factor','Q_gen'],
            outlist=['I_qcmd'],
            inputs=['P_gen','factor','Q_gen'],
            outputs=['I_qcmd'],
            name='P_reactive_const_factor'

        )

    def _pcs_const_voltage(self):
        self.T_Qref = 0.01
        self.T_rU = 0.01
        self.K_p_lv = 0.1
        self.K_i_lv = 2.0

        blk1 = ct.tf(
            [1],[self.T_rU,1],
            inputs='U_t',
            outputs='U_t_1',
            name='blk1'
        )

        sum_blk = ct.summing_junction(
            inputs=['-U_t_1','U_t_ref'],
            outputs=['delta_U'],
            name='sum_blk'
        )

        deadzone_blk = ct.NonlinearIOSystem(
            updfcn=None,
            outfcn=self._deadzone_outfcn,
            inputs='delta_U',
            outputs='delta_u_deadzone',
            params={'d': 0.001},
            name='deadzone_blk'
        )

        PI_blk = ct.tf(
            [self.K_p_lv,self.K_i_lv],[1.0,0.0],
            inputs='delta_u_deadzone',
            outputs='I_q1',
            name='PI_blk'
        )

        sat_blk2 = self._make_sat('sat_blk2','I_q1','I_qcmd',-1.1,1.1)

        return ct.interconnect(
            [blk1,sum_blk,deadzone_blk,PI_blk,sat_blk2],
            inplist=['U_t','U_t_ref'],
            outlist=['I_qcmd'],
            inputs=['U_t','U_t_ref'],
            outputs=['I_qcmd'],
            name='pcs_const_voltage'
        )

# ================= 示例: 频率阶跃响应 =================
def run_demo(t_end=20.0, n=2000, show=True):
    """运行一次频率阶跃示例(49.8 -> 50.2 -> 50 -> 49.8 Hz)并绘图。

    只在 `python control_sys.py` 或显式调用时执行; **导入本模块不会触发仿真**。
    matplotlib 只在示例内部导入, 模块本体不依赖绘图库。

    Parameters
    ----------
    t_end : float
        仿真时长 (s)。
    n : int
        采样点数。
    show : bool
        是否调用 plt.show() 弹窗(批处理时置 False)。

    Returns
    -------
    (t_out, y_out, y_phys)
        t_out  : (n,)     时间 (s)
        y_out  : (4, n)   标幺输出 [I_d, P_gen, I_q, Q_gen]
        y_phys : dict     实际量 {'I_d_A', 'P_gen_kW', 'I_q_A', 'Q_gen_kvar', ...}
    """
    import matplotlib.pyplot as plt

    sys_pcs = pcs_control_sys()
    pu = pcs_per_unit()
    print(sys_pcs.pcs_p_total)

    # 输入顺序: [f, f0, P_ref, Q_ref, U_t] (全部标幺)
    t = np.linspace(0.0, t_end, n)
    U = np.zeros((5, t.size))
    f, f0, f1 = 49.8, 50.0, 50.2
    U[0, :] = f / f0
    U[0, n // 4:] = f1 / f0
    U[0, n // 2:] = 1.0
    U[0, 3 * n // 4:] = f / f0
    U[1, :] = 1.0            # f0
    U[2, :] = 0.0            # P_ref
    U[3, :] = 0.0            # Q_ref
    U[4, :] = 1.0            # U_t

    t_out, y_out = ct.input_output_response(
        sys_pcs.pcs_p_total, t, U, solve_ivp_kwargs={'method': 'Radau'})
    y_phys = pu.outputs_to_physical(y_out)

    fig, ax = plt.subplots(2, 1, figsize=(18, 11), sharex=True)

    ax[0].plot(t_out, y_phys['I_d_A'], 'r', lw=2, label='I_d')
    ax[0].set_ylabel('I_d (A)')
    ax[0].grid(True); ax[0].legend()
    ax[0].set_xlabel('t (s)')

    ax[1].plot(t_out, y_phys['P_gen_kW'], 'b', lw=2, label='P_gen')
    ax[1].set_ylabel('P_gen (kW)')
    ax[1].grid(True); ax[1].legend()
    ax[1].set_xlabel('t (s)')

    fig.tight_layout()
    if show:
        plt.show()
    print(t_out.shape, y_out.shape)
    return t_out, y_out, y_phys


if __name__ == '__main__':
    run_demo()
