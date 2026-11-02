"""
battery_pack.py
===============
支持"单体不一致"的电池包模型 (N_s 串 × N_p 并)。

核心思想
--------
每个单体拥有独立状态 x_j = [SOC_j, V1_j, V2_j] 与独立参数
(R0, R1, C1, R2, C2, Q, soc0, eta, ocv_off)。分数阶阶数 alpha 视为材料
属性, 全包共用(故 GL 权重只需一套)。

  并联组 (N_p 个单体): 端电压必须相等  V_cell_j = V_g  (代数约束),
                        电流按各自等效阻抗分配, Σ_j I_j = I_group。
  串联串 (N_s 组):      所有组流过同一电流  I_group = I_pack(已知)。

求解方法 (每步)
---------------
对每个单体, 支路状态更新对"当前步电流"是仿射的:
      V1_j[k] = a1_j * I_j + b1_j ,  V2_j[k] = a2_j * I_j + b2_j
      a_i_j = 1/(C_i_j * P_i_j) ,  b_i_j = -F_i * hist_i_j / P_i_j
      P_i_j = 1/dt^a_i + 1/(R_i_j C_i_j) ,  F_i = 1/dt^a_i
其中 b 只依赖**历史**(与当前电流无关)。于是单体端电压
      V_cell_j = OCV_j(SOC_j[k]) - I_j * (R0_j + a1_j + a2_j) - b1_j - b2_j
又 SOC_j[k] = SOC_j[k-1] - cQ_j * I_j (cQ_j = eta_j*dt/(3600*Q_j)), 即 OCV
也经由本步电流耦合进来。给定组电流 I_g 与 Σ I_j = I_g, 每步按以下**固定点
迭代**求解(次数很少即可收敛):

      E_j = OCV_j( SOC_j[k-1] - cQ_j * I_j ) - b1_j - b2_j
      R_j = R0_j + a1_j + a2_j
      V_g = ( Σ_j E_j/R_j - I_g ) / ( Σ_j 1/R_j )
      I_j = ( E_j - V_g ) / R_j

第一次迭代用 SOC_j[k-1] 代 OCV, 随后用上一步得到的 I_j 更新 OCV 再迭代。
该式对 N_p=1 一次即收敛到精确解, 并与 FractionalOrderModelSOC 口径一致
(OCV 取"本步更新后"的 SOC)。

状态空间
--------
    状态   x = { (SOC_j, V1_j, V2_j) , j = 0..N_cell-1 },  N_cell = N_s*N_p
    输入   u = [I_pack, T]   (I_pack: 电堆(并联组)总电流, 放电为正; T: ℃标量)
    输出   y = V_term = Σ_g V_g  (电堆端电压 = 各组端电压之和)

约定与注意
----------
  * current 为**电堆总电流**; N_p=1 时即单体电流。
  * SOC 为比例 [0,1]; 电流符号: 放电为正、充电为负。
  * OCV(SOC,T) 来自 CSV(单体), 另加逐单体偏置 ocv_off_j。
  * memory_limit 单位为**步**; 全记忆(None)时环形缓冲为 n×N_cell, 内存很大,
    电池包仿真建议显式设置(≈ 数倍 max(τ_i)/dt)。
  * store_history=True 时, 逐单体时间序列占 n×N_cell×8 字节 ×(SOC,V1,V2,I_cell),
    648 单体×14400 步约需数百 MB; 可置 False 只保留末态。
  * N_p=1 且单体一致时, 结果与 FractionalOrderModelSOC(num_series=N_s) 一致。
"""

import numpy as np
from numba import njit
from scipy.signal import lfilter

from RC2_SOC import load_ocv_table, _ocv_bilinear, DEFAULT_OCV_CSV
from RC2_Numba import compute_gl_weights_numba

# 组内固定点迭代次数(把 OCV 对本步 SOC 的依赖迭代进去); 2~3 次已足够
_OCV_ITER = 2


# --------------------------------------------------------------------------- #
# Numba 内核
# --------------------------------------------------------------------------- #
@njit(fastmath=True)
def core_pack_numba(n_steps, dt, current, T_arr,
                    N_s, N_p, N_cell,
                    R0, a1, a2, P1, P2, F1, F2, cQ,
                    soc0, ocv_off, h0,
                    w1, w2, memory_limit,
                    soc_grid, t_grid, ocv_table,
                    soc_grid_b, t_grid_b, ocv_br, br_on, kh, i_eps, h_ref, kh_rest,
                    store_history, n_iter):
    """逐单体状态 + 组内等电压固定点迭代求解的内核(含可选 OCV 双支路)。"""
    Lmax = n_steps
    if memory_limit > 0 and memory_limit < n_steps:
        Lmax = memory_limit

    buf1 = np.zeros((Lmax, N_cell))     # 最近 Lmax 步的 V1(逐单体)
    buf2 = np.zeros((Lmax, N_cell))

    V_term = np.zeros(n_steps)
    V_group = np.zeros((n_steps, N_s))  # 各组端电压(= 组内单体端电压)

    if store_history:
        SOC_h = np.zeros((n_steps, N_cell))
        V1_h = np.zeros((n_steps, N_cell))
        V2_h = np.zeros((n_steps, N_cell))
        I_h = np.zeros((n_steps, N_cell))
    else:
        SOC_h = np.zeros((1, 1))
        V1_h = np.zeros((1, 1))
        V2_h = np.zeros((1, 1))
        I_h = np.zeros((1, 1))

    soc = soc0.copy()
    hh = h0.copy()                      # 迟滞状态 h ∈ [0,1] (0=放电支路, 1=充电支路)
    v1 = np.zeros(N_cell)
    v2 = np.zeros(N_cell)

    Earr = np.zeros(N_cell)
    Rarr = np.zeros(N_cell)
    B1arr = np.zeros(N_cell)
    B2arr = np.zeros(N_cell)
    Iarr = np.zeros(N_cell)
    HVarr = np.zeros(N_cell)            # 本步迟滞电压 h*(OCV_chg - OCV_dis)
    Vg = np.zeros(N_s)

    # ---------------- k = 0: 初始时刻, V1=V2=0, 仅欧姆项, SOC 不更新 ----------------
    I_g = current[0]
    for g in range(N_s):
        sER = 0.0
        sIR = 0.0
        for c in range(N_p):
            j = g * N_p + c
            o_d = _ocv_bilinear(soc[j], T_arr[0], soc_grid, t_grid, ocv_table)
            HVarr[j] = 0.0
            if br_on == 1:                    # 迟滞: 放电基准 + h*(充电支路 - 放电支路)
                o_c = _ocv_bilinear(soc[j], T_arr[0], soc_grid_b, t_grid_b, ocv_br)
                HVarr[j] = hh[j] * (o_c - o_d)
            ocv = o_d + ocv_off[j] + HVarr[j]
            Earr[j] = ocv
            Rarr[j] = R0[j]
            sER += ocv / R0[j]
            sIR += 1.0 / R0[j]
        vg = (sER - I_g) / sIR
        Vg[g] = vg
        for c in range(N_p):
            j = g * N_p + c
            Iarr[j] = (Earr[j] - vg) / Rarr[j]

    vt0 = 0.0
    for g in range(N_s):
        V_group[0, g] = Vg[g]
        vt0 += Vg[g]
    V_term[0] = vt0
    if store_history:
        for j in range(N_cell):
            SOC_h[0, j] = soc[j]
            V1_h[0, j] = 0.0
            V2_h[0, j] = 0.0
            I_h[0, j] = Iarr[j]

    # ---------------- k >= 1 ----------------
    for k in range(1, n_steps):
        I_g = current[k]

        L = k
        if memory_limit > 0 and k > memory_limit:
            L = memory_limit
        r_start = (k - 1) % Lmax

        for g in range(N_s):
            # --- (a) 卷积只算一次: 得到 b1,b2 与系数 R ---
            for c in range(N_p):
                j = g * N_p + c
                h1 = 0.0
                h2 = 0.0
                r = r_start
                for l in range(1, L + 1):
                    h1 += w1[l] * buf1[r, j]
                    h2 += w2[l] * buf2[r, j]
                    r -= 1
                    if r < 0:
                        r = Lmax - 1
                B1arr[j] = -F1 * h1 / P1[j]
                B2arr[j] = -F2 * h2 / P2[j]
                Rarr[j] = R0[j] + a1[j] + a2[j]
                # 初值: OCV 用步初 SOC (放电基准 + 迟滞项)
                o_d = _ocv_bilinear(soc[j], T_arr[k], soc_grid, t_grid, ocv_table)
                HVarr[j] = 0.0
                if br_on == 1:
                    o_c = _ocv_bilinear(soc[j], T_arr[k], soc_grid_b, t_grid_b, ocv_br)
                    HVarr[j] = hh[j] * (o_c - o_d)
                Earr[j] = o_d + ocv_off[j] + HVarr[j] - B1arr[j] - B2arr[j]

            # --- (b) 固定点迭代: 把 OCV 对本步 SOC 的依赖迭代进去 ---
            sER = 0.0
            sIR = 0.0
            for c in range(N_p):
                j = g * N_p + c
                sER += Earr[j] / Rarr[j]
                sIR += 1.0 / Rarr[j]
            vg = (sER - I_g) / sIR
            for _it in range(n_iter):
                for c in range(N_p):
                    j = g * N_p + c
                    Ij = (Earr[j] - vg) / Rarr[j]
                    sk = soc[j] - cQ[j] * Ij
                    Earr[j] = (_ocv_bilinear(sk, T_arr[k], soc_grid, t_grid, ocv_table)
                               + ocv_off[j] + HVarr[j] - B1arr[j] - B2arr[j])
                sER = 0.0
                sIR = 0.0
                for c in range(N_p):
                    j = g * N_p + c
                    sER += Earr[j] / Rarr[j]
                    sIR += 1.0 / Rarr[j]
                vg = (sER - I_g) / sIR

            Vg[g] = vg
            for c in range(N_p):
                j = g * N_p + c
                Iarr[j] = (Earr[j] - vg) / Rarr[j]

        # --- (c) 更新逐单体状态并写入缓冲/历史 ---
        row = k % Lmax
        for g in range(N_s):
            for c in range(N_p):
                j = g * N_p + c
                Ij = Iarr[j]
                v1[j] = a1[j] * Ij + B1arr[j]
                v2[j] = a2[j] * Ij + B2arr[j]
                soc[j] = soc[j] - cQ[j] * Ij
                if br_on == 1:                      # 迟滞: 充电->1, 放电->0, 静置->h_ref
                    if Ij < -i_eps:
                        tgt = 1.0
                        kk = kh
                    elif Ij > i_eps:
                        tgt = 0.0
                        kk = kh
                    else:
                        tgt = h_ref
                        kk = kh_rest                # =0 时静置保持(旧行为)
                    hh[j] += (tgt - hh[j]) * kk
                buf1[row, j] = v1[j]
                buf2[row, j] = v2[j]
                if store_history:
                    SOC_h[k, j] = soc[j]
                    V1_h[k, j] = v1[j]
                    V2_h[k, j] = v2[j]
                    I_h[k, j] = Ij

        vt = 0.0
        for g in range(N_s):
            V_group[k, g] = Vg[g]
            vt += Vg[g]
        V_term[k] = vt

    return V_term, V_group, SOC_h, V1_h, V2_h, I_h, soc, v1, v2, hh


# --------------------------------------------------------------------------- #
# Numba 内核(单步版, 供在线/闭环逐步推进使用)
# --------------------------------------------------------------------------- #
@njit(fastmath=True)
def core_pack_step_numba(k, dt, I_g, T,
                         N_s, N_p, N_cell,
                         R0, a1, a2, P1, P2, F1, F2, cQ, ocv_off,
                         w1, w2, memory_limit,
                         soc_grid, t_grid, ocv_table,
                         soc_grid_b, t_grid_b, ocv_br, br_on, kh, i_eps, h_ref, kh_rest,
                         soc, v1, v2, hh, buf1, buf2,
                         n_iter,
                         Vg, Iarr, Earr, Rarr, B1arr, B2arr, HVarr):
    """单步推进一个 dt: 用指令电流 I_g 把状态从 k-1 推进到 k。

    与批处理内核 core_pack_numba 的 k>=1 分支**逐式一致**, 只是把状态
    (soc/v1/v2/hh 与环形缓冲 buf1/buf2) 由外部持有并在原地更新, 因而调用方
    可在两次调用之间读取状态(见 battery_pack.state())。

    就地更新: soc, v1, v2, hh, buf1, buf2
    输出写入: Vg(N_s), Iarr(N_cell); 工作数组 HVarr(N_cell) 承载迟滞电压
    """
    Lmax = buf1.shape[0]
    L = k
    if memory_limit > 0 and k > memory_limit:
        L = memory_limit
    r_start = (k - 1) % Lmax          # 上一步在环形缓冲中的行

    for g in range(N_s):
        # --- (a) 卷积只算一次: 历史 -> b, 以及系数 R; 并用步初 SOC 初始化 OCV ---
        for c in range(N_p):
            j = g * N_p + c
            h1 = 0.0
            h2 = 0.0
            r = r_start
            for l in range(1, L + 1):
                h1 += w1[l] * buf1[r, j]
                h2 += w2[l] * buf2[r, j]
                r -= 1
                if r < 0:
                    r = Lmax - 1
            B1arr[j] = -F1 * h1 / P1[j]
            B2arr[j] = -F2 * h2 / P2[j]
            Rarr[j] = R0[j] + a1[j] + a2[j]
            o_d = _ocv_bilinear(soc[j], T, soc_grid, t_grid, ocv_table)
            HVarr[j] = 0.0
            if br_on == 1:                    # 迟滞: 放电基准 + h*(充电支路 - 放电支路)
                o_c = _ocv_bilinear(soc[j], T, soc_grid_b, t_grid_b, ocv_br)
                HVarr[j] = hh[j] * (o_c - o_d)
            Earr[j] = o_d + ocv_off[j] + HVarr[j] - B1arr[j] - B2arr[j]

        # --- (b) 组内固定点迭代: 把 OCV 对本步 SOC 的依赖迭代进去 ---
        sER = 0.0
        sIR = 0.0
        for c in range(N_p):
            j = g * N_p + c
            sER += Earr[j] / Rarr[j]
            sIR += 1.0 / Rarr[j]
        vg = (sER - I_g) / sIR
        for _it in range(n_iter):
            for c in range(N_p):
                j = g * N_p + c
                Ij = (Earr[j] - vg) / Rarr[j]
                sk = soc[j] - cQ[j] * Ij
                Earr[j] = (_ocv_bilinear(sk, T, soc_grid, t_grid, ocv_table)
                           + ocv_off[j] + HVarr[j] - B1arr[j] - B2arr[j])
            sER = 0.0
            sIR = 0.0
            for c in range(N_p):
                j = g * N_p + c
                sER += Earr[j] / Rarr[j]
                sIR += 1.0 / Rarr[j]
            vg = (sER - I_g) / sIR

        Vg[g] = vg
        for c in range(N_p):
            j = g * N_p + c
            Iarr[j] = (Earr[j] - vg) / Rarr[j]

    # --- (c) 原地更新逐单体状态并写入环形缓冲 ---
    row = k % Lmax
    for g in range(N_s):
        for c in range(N_p):
            j = g * N_p + c
            Ij = Iarr[j]
            v1[j] = a1[j] * Ij + B1arr[j]
            v2[j] = a2[j] * Ij + B2arr[j]
            soc[j] = soc[j] - cQ[j] * Ij
            if br_on == 1:                          # 迟滞: 充电->1, 放电->0, 静置->h_ref
                if Ij < -i_eps:
                    tgt = 1.0
                    kk = kh
                elif Ij > i_eps:
                    tgt = 0.0
                    kk = kh
                else:
                    tgt = h_ref
                    kk = kh_rest                  # =0 时静置保持(旧行为)
                hh[j] += (tgt - hh[j]) * kk
            buf1[row, j] = v1[j]
            buf2[row, j] = v2[j]


# --------------------------------------------------------------------------- #
# 模型类
# --------------------------------------------------------------------------- #
class battery_pack:
    """支持"单体不一致"的电池包分数阶等效电路模型 (N_s 串 × N_p 并)。

    本类把电池包展开到**逐个单体**建模: 每个单体是一个分数阶二阶 RC 单元
    (欧姆内阻 R0 + 两条分数阶 RC 支路), 并带有各自的 SOC 状态、容量与 OCV
    偏置。因此可以复现并联组内的**电流不均衡**、单体/组间的 **SOC 分化**,
    以及充放电截止时的"木桶效应"等现象。

    拓扑与约束
    ----------
    串联 : N_s 个并联组首尾相接, 共享同一电流 I_pack (即输入的 current)。
    并联 : 每组 N_p 个单体并联, 端电压被强制相等(等电压代数约束), 各自的
           电流按等效阻抗分配; 静置时若单体 OCV 不同则产生环流。
    编号 : 单体下标 j = g * N_p + c
           (g = 0..N_s-1 为组号, c = 0..N_p-1 为组内位置)。

    状态空间
    --------
    状态  x_j = [SOC_j, V1_j, V2_j],  j = 0..N_cell-1, N_cell = N_s*N_p
          SOC_j : 单体荷电状态, 比例 [0, 1]
          V1_j  : 支路 1 极化电压 (V), 分数阶, 含历史记忆
          V2_j  : 支路 2 极化电压 (V), 分数阶, 含历史记忆
    输入  u = [I_pack, T]
          I_pack : 电堆(每组)总电流 (A), 约定放电为正、充电为负
          T      : 温度 (℃), 标量常数
    输出  y = V_term = Σ_g V_g  (电堆端电压, V); V_g 为第 g 组端电压

    数学模型 (逐单体, k 为时间步)
    ----------------------------
        SOC      SOC_j[k] = SOC_j[k-1] - eta_j * I_j[k] * dt / (3600 * Q_j)
        支路电压 V_i,j[k] = ( I_j[k]/C_i,j - (1/dt^a_i) * Σ_{l=1..L} w_i[l] * V_i,j[k-l] )
                            / ( 1/dt^a_i + 1/(R_i,j * C_i,j) )        (i = 1, 2)
        端电压   V_cell_j[k] = OCV_j(SOC_j[k], T) + ocv_off_j
                               - I_j[k]*R0_j - V1_j[k] - V2_j[k]
        组/包电压 V_g = V_cell_j[k] (组内相等),  V_term = Σ_g V_g
        记忆窗口 L = min(k, memory_limit)  (memory_limit<=0 表示全记忆)

    OCV 由 CSV 表 OCV(SOC, T) 双线性插值给出(单体值), 再叠加逐单体偏置 ocv_off_j。

    并联电流分配
    ------------
    支路状态对"本步电流"是仿射的, 故组内可用闭式式 + 少量固定点迭代求解
    (因 OCV 随本步 SOC 变化, 迭代 2 次即收敛):
        R_j = R0_j + a1_j + a2_j,   a_i,j = 1/(C_i,j * (1/dt^a_i + 1/(R_i,j*C_i,j)))
        E_j = OCV_j(SOC_j[k]) + ocv_off_j - b1_j - b2_j
        V_g = ( Σ_j E_j/R_j - I_g ) / ( Σ_j 1/R_j )
        I_j = ( E_j - V_g ) / R_j
    当 N_p=1 且单体一致时, 本类退化为单包模型, 结果与
    FractionalOrderModelSOC(num_series=N_s) 在机器精度内一致。

    两种调用方式
    ------------
    ① 批处理(离线, 一次跑完):   run(current_profile, T) -> dict
    ② 单步/在线(闭环节拍):      reset(soc0); 然后 state() <-> step(I, T)
       对应"① 读状态(不更新) → ② 控制器调度计算 → ③ 下发单步电流指令并推进"
       的循环。step() 单步耗时为 O(L*N_cell)(L=min(k,memory_limit)),
       **与已推进步数无关**, 满足固定控制节拍。
       注意: 单步方式要求 **有限的 memory_limit**(单位: 步), 全记忆(None)
       会使历史缓冲随步数无限增长, 不适合在线运行。

    Parameters
    ----------
    N_s : int
        串联组数(如 216)。
    N_p : int, 默认 1
        每组并联单体数(如 3)。
    dt : float, 默认 0.1
        仿真步长 (s)。
    memory_limit : int or None, 默认 None
        短记忆窗口长度, 单位**步**; None 表示全记忆(不截断)。
        全记忆时环形缓冲为 n_steps × N_cell, 电池包仿真内存开销大,
        建议显式设置(≈ 数倍 max(tau_i)/dt 即可)。
        **使用单步接口 reset()/state()/step() 时必须为有限整数。**
    ocv_csv_path : str, 默认 DEFAULT_OCV_CSV
        OCV(SOC, T) 表路径: 首列为 DOD(如 "0%"), 其余各列为温度(℃),
        单元格为**单体** OCV (V); SOC = 1 - DOD。
    store_history : bool, 默认 True
        是否保存逐单体时间序列(SOC/V1/V2/I_cell, 形状 (n_steps, N_cell))。
        内存约 n_steps × N_cell × 8 × 4 字节, 648 单体 × 14400 步约数百 MB;
        置 False 时只保留末态与 V_term。

    Attributes
    ----------
    N_cell : int
        单体总数 = N_s * N_p。
    soc_grid : (n_soc,) ndarray
        OCV 表的 SOC 采样点(比例, 升序)。
    t_grid : (n_temp,) ndarray
        OCV 表的温度采样点(℃, 升序)。
    ocv_table : (n_soc, n_temp) ndarray
        单体 OCV 表 (V)。
    params : dict or None
        set_cell_parameters 写入的逐单体参数字典(未设置时为 None)。
    F1, F2 : float
        分数阶系数 1/dt^alpha1, 1/dt^alpha2。
    P1, P2 : (N_cell,) ndarray
        支路递推分母 1/dt^a_i + 1/(R_i*C_i)。
    a1, a2 : (N_cell,) ndarray
        支路电压对本步电流的系数 1/(C_i * P_i)。
    cQ : (N_cell,) ndarray
        每安培每秒的 SOC 变化 eta*dt/(3600*Q)。
    weights_cache : dict
        GL 权重缓存, 键为 (alpha, n_points)。

    See Also
    --------
    RC2_SOC.FractionalOrderModelSOC : 单体(单串电流)模型。

    Examples
    --------
    >>> import numpy as np
    >>> from battery_pack import battery_pack
    >>> pk = battery_pack(216, N_p=3, dt=1.0, memory_limit=5000)
    >>> pk.set_cell_parameters(R0=0.001, R1=0.0008, C1=50000.0, alpha1=0.9,
    ...                        R2=0.0012, C2=250000.0, alpha2=0.7,
    ...                        Q=50.0, soc0=1.0, eta=1.0)
    >>> res = pk.run(np.full(3600, 150.0), T=25.0)   # 电堆 1C (3 并)
    >>> V_term, SOC = res["V_term"], res["SOC"]      # (n,) 与 (n, 648)
    >>>
    >>> # ② 单步/在线闭环(读状态 -> 控制器 -> 单步推进)
    >>> pk2 = battery_pack(216, N_p=12, dt=5.0, memory_limit=1500)
    >>> pk2.set_cell_parameters(R0=1.4e-3, R1=0.8e-3, C1=50000.0, alpha1=0.9,
    ...                         R2=1.2e-3, C2=250000.0, alpha2=0.7,
    ...                         Q=40.0, soc0=1.0, eta=1.0)
    >>> pk2.reset(soc0=1.0)
    >>> st = pk2.state()                 # ① 采集状态(不更新)
    >>> I_cmd = controller(st)           # ② 上位控制器调度(外部)
    >>> out = pk2.step(I_cmd, T=25.0)    # ③ 单步推进并返回本步输出

    Notes
    -----
    * run 的 current 是**电堆总电流**; N_p=1 时即单体电流。
    * 分数阶阶数 alpha1/alpha2 为全包共用的标量(视为材料属性), 而 R/C/Q/soc0/
      eta/ocv_off 可逐单体不同。
    * 温度 T 可为标量(全程恒定)或长度 n_steps 的数组(逐步温度, 逐步查 OCV 表)。
    * 本模型为电特性模型, 不含热模型、老化模型与均衡电路。
    """

    def __init__(self, N_s, N_p=1, dt=0.1, memory_limit=None,
                 ocv_csv_path=DEFAULT_OCV_CSV, store_history=True,
                 ocv_dis_csv=None, ocv_chg_csv=None, tau_h=120.0, hys_i_eps=0.1,
                 h_ref=0.5, tau_rest=0.0):
        """构造电池包模型(各参数的含义与单位见类文档的 Parameters 一节)。

        初始化时即读取 OCV(SOC, T) 表, 但**不设置任何单体参数**; 需在调用
        run() 之前先调用 set_cell_parameters()。

        ocv_dis_csv, ocv_chg_csv : str or None, 默认 None
            放电/充电**两条 OCV 支路**表(与原表同格式: 首列 DOD, 其余为温度列)。
            两者都给出时启用迟滞:
                OCV_eff = OCV_dis(SOC,T) + ocv_off
                          + h * (OCV_chg(SOC,T) - OCV_dis(SOC,T))
            h∈[0,1] 为迟滞状态(充电->1, 放电->0, 静置见 h_ref/tau_rest)。
            此时 ocv_csv_path 不再使用(单表被两条支路替换)。都不给时退回单表无迟滞。
        tau_h : float, 默认 120.0
            迟滞状态追随电流方向的时间常数 (s); 仅在有电流时生效。
        hys_i_eps : float, 默认 0.1
            判定"有电流"的逐单体电流阈值 (A); |I_j| 小于该值视为静置。
        h_ref : float, 默认 0.5
            静置期 h 的弛豫目标(0=放电支路, 1=充电支路, 0.5=中间)。
        tau_rest : float, 默认 0.0
            静置期 h 向 h_ref 弛豫的时间常数 (s); **0 表示静置保持 h 不变**
            (单状态迟滞模型的经典假设, 与旧行为一致)。
        """
        self.N_s = int(N_s)
        self.N_p = int(N_p)
        self.N_cell = self.N_s * self.N_p
        self.dt = dt
        self.memory_limit = -1 if memory_limit is None else int(memory_limit)
        self.store_history = bool(store_history)
        self.params = None
        self.weights_cache = {}
        self.tau_h = float(tau_h)
        self.hys_i_eps = float(hys_i_eps)
        self.h_ref = float(h_ref)
        self.tau_rest = float(tau_rest)
        # --- OCV: 单表(无迟滞) 或 双支路(迟滞) ---
        if ocv_dis_csv is not None and ocv_chg_csv is not None:
            self.soc_grid, self.t_grid, self.ocv_table = load_ocv_table(ocv_dis_csv)
            self.soc_grid_b, self.t_grid_b, self.ocv_br = load_ocv_table(ocv_chg_csv)
            self.br_on = 1
        else:
            self.soc_grid, self.t_grid, self.ocv_table = load_ocv_table(ocv_csv_path)
            self.soc_grid_b = np.zeros(0)
            self.t_grid_b = np.zeros(0)
            self.ocv_br = np.zeros((0, 0))
            self.br_on = 0
        # 单步/在线接口的内部状态(惰性初始化; 见 reset()/state()/step())
        self._k = None                    # None = 尚未初始化; 否则为已推进步数
        self._t = 0.0                     # 累计仿真时间 (s) = _k * dt
        self._Lmax = 0                    # 环形缓冲行数 = memory_limit
        self._soc = None                  # (N_cell,) 逐单体 SOC
        self._v1 = None                   # (N_cell,) 支路1 极化电压
        self._v2 = None                   # (N_cell,) 支路2 极化电压
        self._h = None                    # (N_cell,) 迟滞状态 (0=放电/1=充电支路)
        self._buf1 = None                 # (Lmax, N_cell) V1 环形缓冲
        self._buf2 = None                 # (Lmax, N_cell) V2 环形缓冲
        self._w1 = None                   # (Lmax+1,) 支路1 GL 权重
        self._w2 = None                   # (Lmax+1,) 支路2 GL 权重
        self._Vg = None                   # (N_s,) 本步各组端电压
        self._Iarr = None                 # (N_cell,) 本步逐单体电流
        self._Earr = None                 # 逐步核工作数组
        self._Rarr = None
        self._B1 = None
        self._B2 = None

    # ------------------------------------------------------------------ #
    def _bcast(self, v, name):
        """标量广播为逐单体数组; 也可直接传长度 N_cell 的数组。"""
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(self.N_cell, float(arr))
        arr = np.ascontiguousarray(arr.ravel(), dtype=np.float64)
        if arr.size != self.N_cell:
            raise ValueError(f"{name} 应为标量或长度 {self.N_cell} 的数组, "
                             f"实际长度 {arr.size}")
        return arr

    @staticmethod
    def _as_scalar(v, name):
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size != 1:
            raise ValueError(f"{name} 必须为标量(全包共用)")
        return float(a[0])

    # ------------------------------------------------------------------ #
    def set_cell_parameters(self, R0, R1, C1, alpha1, R2, C2, alpha2, Q,
                            soc0=1.0, eta=1.0, ocv_off=0.0, h0=0.0,
                            R3=0.0, tau3=0.0):
        """设置逐单体的电化学/电路参数(标量自动广播)。

        所有参数既可为标量(广播到全部 N_cell 个单体), 也可为长度 N_cell 的
        数组(逐单体指定)。数组按 j = g * N_p + c 排布: 前 N_p 个属于第 0 组,
        其后 N_p 个属于第 1 组, 依此类推。

        Parameters
        ----------
        R0 : float or (N_cell,) ndarray
            单体欧姆内阻 (Ω)。
        R1 : float or (N_cell,) ndarray
            支路 1 的电阻 (Ω)。
        C1 : float or (N_cell,) ndarray
            支路 1 的分数阶电容。时间常数 tau1 = (R1*C1)**(1/alpha1) (s)。
        alpha1 : float
            支路 1 的分数阶阶数(标量, 全包共用), 典型 0.5 ~ 1.0。
        R2 : float or (N_cell,) ndarray
            支路 2 的电阻 (Ω)。
        C2 : float or (N_cell,) ndarray
            支路 2 的分数阶电容。时间常数 tau2 = (R2*C2)**(1/alpha2) (s)。
        alpha2 : float
            支路 2 的分数阶阶数(标量, 全包共用)。
        Q : float or (N_cell,) ndarray
            单体容量 (Ah); 其 1C 电流即等于 Q。
        soc0 : float or (N_cell,) ndarray, 默认 1.0
            初始 SOC, 比例 [0, 1]。
        eta : float or (N_cell,) ndarray, 默认 1.0
            库仑效率, 取值 (0, 1]。
        ocv_off : float or (N_cell,) ndarray, 默认 0.0
            逐单体 OCV 偏置 (V), 用于模拟单体间电压一致性差异。
        h0 : float or (N_cell,) ndarray, 默认 0.0
            迟滞状态初值 (仅构造时给了 ocv_hys_csv 才有意义), 取值 [0,1]:
            0 = 放电支路(基准), 1 = 充电支路(OCV 抬高 dV_hys)。
        R3 : float, 默认 0.0
            第三条**慢支路**的电阻 (Ω, 标量)。0 = 关闭该支路(保持原二阶行为)。
            该支路用整数阶一阶 RC 近似(V3 以 tau3 指数衰减), 用于补足长时弛豫尾部。
        tau3 : float, 默认 0.0
            第三条慢支路的时间常数 (s); 与 R3 一起启用, tau3<=0 视为关闭。

        Returns
        -------
        None
            结果写入 self.params, 并调用 _precompute() 重算派生系数,
            同时清空 self.weights_cache。

        Raises
        ------
        ValueError
            当某参数既非标量、长度又不等于 N_cell 时; 或 alpha1/alpha2 非标量时。

        Notes
        -----
        * alpha1/alpha2 只接受标量(分数阶阶数视为材料属性, 全包共用);
          若需要逐单体不同阶数, 需扩展为每单体一套 GL 权重。
        * 每次调用都会清空权重缓存(self.weights_cache)。
        """
        self.params = {
            'R0': self._bcast(R0, 'R0'),
            'R1': self._bcast(R1, 'R1'),
            'C1': self._bcast(C1, 'C1'),
            'R2': self._bcast(R2, 'R2'),
            'C2': self._bcast(C2, 'C2'),
            'Q': self._bcast(Q, 'Q'),
            'soc0': self._bcast(soc0, 'soc0'),
            'eta': self._bcast(eta, 'eta'),
            'ocv_off': self._bcast(ocv_off, 'ocv_off'),
            'h0': self._bcast(h0, 'h0'),
            'alpha1': self._as_scalar(alpha1, 'alpha1'),
            'alpha2': self._as_scalar(alpha2, 'alpha2'),
            'R3': float(R3),
            'tau3': float(tau3),
        }
        self.weights_cache = {}
        self._precompute()
        # 参数变更 -> 单步/在线状态作废(下次 state()/step() 会按新参数重建)
        self._clear_step_state()

    # ------------------------------------------------------------------ #
    def _clear_step_state(self):
        """释放单步/在线接口的状态与缓冲(参数变更时调用)。"""
        self._k = None
        self._t = 0.0
        self._soc = self._v1 = self._v2 = self._v3 = self._h = None
        self._buf1 = self._buf2 = None
        self._w1 = self._w2 = None
        self._Vg = self._Iarr = self._Earr = self._Rarr = self._B1 = self._B2 = None
        self._HVarr = None

    # ------------------------------------------------------------------ #
    def _precompute(self):
        """预计算与"当前电流无关"的逐单体系数。"""
        p = self.params
        dt = self.dt
        self.F1 = 1.0 / (dt ** p['alpha1'])
        self.F2 = 1.0 / (dt ** p['alpha2'])
        self.P1 = self.F1 + 1.0 / (p['R1'] * p['C1'])     # term1_den
        self.P2 = self.F2 + 1.0 / (p['R2'] * p['C2'])     # term2_den
        self.a1 = 1.0 / (p['C1'] * self.P1)               # V1 对 I 的当前步系数
        self.a2 = 1.0 / (p['C2'] * self.P2)
        self.cQ = p['eta'] * dt / (3600.0 * p['Q'])       # 每安培每秒的 SOC 变化
        # 迟滞状态每步追随系数(<=1); 未启用双支路(br_on=0)时不生效
        self.kh = min(1.0, dt / self.tau_h) if self.tau_h > 0 else 1.0
        # 静置期 h 向 h_ref 的弛豫系数; tau_rest=0 -> 0 (静置保持, 旧行为)
        self.kh_rest = min(1.0, dt / self.tau_rest) if self.tau_rest > 0 else 0.0
        # 第三条慢支路(整数阶一阶 RC): V3[k]=a3*V3[k-1]+b3*I_cell[k]
        tau3 = float(p['tau3'])
        R3 = float(p['R3'])
        if R3 > 0.0 and tau3 > 0.0:
            self.a3 = float(np.exp(-dt / tau3))
            self.b3 = R3 * (1.0 - self.a3)
            self.R3_on = True
        else:
            self.a3 = self.b3 = 0.0
            self.R3_on = False

    # ------------------------------------------------------------------ #
    def _get_weights(self, alpha, n_points):
        """获取(并缓存) Grünwald-Letnikov 权重 w_l = (-1)^l * C(alpha, l)。

        Parameters
        ----------
        alpha : float
            分数阶阶数。
        n_points : int
            权重数组长度(需 >= 最大记忆窗口 L + 1)。

        Returns
        -------
        (n_points,) ndarray
            权重数组; 缓存键为 (alpha, n_points)。
        """
        key = (alpha, n_points)
        if key not in self.weights_cache:
            self.weights_cache[key] = compute_gl_weights_numba(alpha, n_points)
        return self.weights_cache[key]

    # ------------------------------------------------------------------ #
    def run(self, current_profile, T=25.0, store_cell_history=None):
        """执行一次电流工况仿真(每次调用都从 soc0 重新起算, 无内部状态残留)。

        Parameters
        ----------
        current_profile : array_like
            电堆(每个并联组)总电流时间序列 (A), 长度 n_steps。
            约定**放电为正、充电为负**; 内部会转为连续 float64 数组。
        T : float or array_like, 默认 25.0
            温度 (℃)。标量 -> 全程恒定; 也可传长度 n_steps 的数组(逐步温度),
            用于 OCV(SOC, T) 查表; 越界按表边界截断。
        store_cell_history : bool or None, 默认 None
            覆盖构造时的 self.store_history; None 表示沿用实例设置。

        Returns
        -------
        dict
            "V_term"    : (n_steps,)          电堆端电压 (V) = Σ_g V_group
            "V_group"   : (n_steps, N_s)      各组端电压 (V)(= 组内单体端电压)
            "t"         : (n_steps,)          时间轴 (s)
            "soc_final" : (N_cell,)           末态 SOC 比例
            "V1_final"  : (N_cell,)           末态支路 1 极化电压 (V)
            "V2_final"  : (N_cell,)           末态支路 2 极化电压 (V)
            —— 当 store_cell_history 为真时, 额外包含 ——
            "SOC"       : (n_steps, N_cell)   逐单体 SOC 比例
            "V1"        : (n_steps, N_cell)   逐单体支路 1 极化电压 (V)
            "V2"        : (n_steps, N_cell)   逐单体支路 2 极化电压 (V)
            "I_cell"    : (n_steps, N_cell)   逐单体电流 (A), 组内 Σ_c I_cell = I_pack

        Raises
        ------
        ValueError
            尚未调用 set_cell_parameters 设置参数时。

        Notes
        -----
        * 逐单体量按 j = g * N_p + c 排布, 便于用 reshape 还原为 (n, N_s, N_p)。
        * n_steps = 0 时返回长度为 0 的 V_term/t 以及零末态。
        """
        if self.params is None:
            raise ValueError("Parameters not set!")

        current = np.ascontiguousarray(current_profile, dtype=np.float64)
        n_steps = current.shape[0]
        # 温度: 标量 -> 广播为全程恒定; 数组 -> 逐步温度(长度须为 n_steps)
        T_in = np.asarray(T, dtype=np.float64)
        if T_in.ndim == 0:
            T_arr = np.full(n_steps, float(T_in), dtype=np.float64)
        else:
            T_arr = np.ascontiguousarray(T_in.ravel(), dtype=np.float64)
            if T_arr.size != n_steps:
                raise ValueError(
                    f"T 为数组时长度须等于 n_steps={n_steps}, 实际 {T_arr.size}")
        if n_steps == 0:
            empty = np.zeros(0, dtype=np.float64)
            return {"V_term": empty, "t": empty,
                    "soc_final": np.zeros(self.N_cell),
                    "V1_final": np.zeros(self.N_cell),
                    "V2_final": np.zeros(self.N_cell),
                    "H_final": np.zeros(self.N_cell)}

        Lmax = n_steps if self.memory_limit <= 0 else min(self.memory_limit, n_steps)
        max_mem = Lmax + 1
        p = self.params
        w1 = self._get_weights(p['alpha1'], max_mem)
        w2 = self._get_weights(p['alpha2'], max_mem)

        sh = self.store_history if store_cell_history is None else bool(store_cell_history)

        (V_term, V_group, SOC_h, V1_h, V2_h, I_h,
         soc_f, v1_f, v2_f, h_f) = core_pack_numba(
            n_steps, self.dt, current, T_arr,
            self.N_s, self.N_p, self.N_cell,
            p['R0'], self.a1, self.a2, self.P1, self.P2, self.F1, self.F2, self.cQ,
            p['soc0'], p['ocv_off'], p['h0'],
            w1, w2, self.memory_limit,
            self.soc_grid, self.t_grid, self.ocv_table,
            self.soc_grid_b, self.t_grid_b, self.ocv_br, self.br_on, self.kh,
            self.hys_i_eps, self.h_ref, self.kh_rest, sh, _OCV_ITER)

        res = {
            "V_term": V_term,
            "V_group": V_group,
            "t": np.arange(n_steps) * self.dt,
            "soc_final": soc_f,
            "V1_final": v1_f,
            "V2_final": v2_f,
            "H_final": h_f,
        }
        if self.R3_on:
            # 第三条慢支路(逐单体一阶 RC), 用 lfilter 高效递推, 叠加到端电压
            i_cell_run = current / float(self.N_p)
            v3 = lfilter([self.b3], [1.0, -self.a3], i_cell_run)
            res["V_term"] = V_term + self.N_s * v3
            res["V3"] = v3
        if sh:
            res["SOC"] = SOC_h
            res["V1"] = V1_h
            res["V2"] = V2_h
            res["I_cell"] = I_h
        return res

    # ================================================================== #
    # 单步 / 在线(闭环)接口 —— 供"①读状态 → ②控制器决策 → ③单步推进"循环使用
    # ================================================================== #
    def _ensure_step_state(self):
        """惰性初始化单步状态(首次调用时按 soc0 建立, 之后不再自动重置)。"""
        if self.params is None:
            raise ValueError("Parameters not set! 请先调用 set_cell_parameters()。")
        if self._k is not None:
            return
        if self.memory_limit <= 0:
            raise ValueError(
                "单步接口 state()/step() 需要**有限的 memory_limit**(单位: 步); "
                "全记忆(None)会使历史缓冲随步数无限增长, 不适合在线运行。"
                "请用 battery_pack(..., memory_limit=<int>) 构造实例。")
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, soc0=None):
        """复位到初始状态(t=0, k=0, V1=V2=0), 供重新开始一段在线运行。

        Parameters
        ----------
        soc0 : float or (N_cell,) ndarray, optional
            新的初始 SOC(比例 [0,1]); 省略则沿用 set_cell_parameters 的 soc0。

        Returns
        -------
        self : battery_pack
            便于链式调用。
        """
        if self.params is None:
            raise ValueError("Parameters not set! 请先调用 set_cell_parameters()。")
        if self.memory_limit <= 0:
            raise ValueError(
                "单步接口需要有限的 memory_limit(单位: 步); 请用 "
                "battery_pack(..., memory_limit=<int>) 构造实例。")
        Lmax = int(self.memory_limit)
        N = self.N_cell
        self._Lmax = Lmax
        self._soc = (self.params['soc0'].copy() if soc0 is None
                     else self._bcast(soc0, 'soc0').copy())
        self._v1 = np.zeros(N, dtype=np.float64)
        self._v2 = np.zeros(N, dtype=np.float64)
        self._v3 = 0.0          # 第三条慢支路(标量, 全包共用)
        self._h = self.params['h0'].copy()
        # buf 的第 0 行即 V1[0]=V2[0]=0(初始时刻), 其余行存放后续历史
        self._buf1 = np.zeros((Lmax, N), dtype=np.float64)
        self._buf2 = np.zeros((Lmax, N), dtype=np.float64)
        self._k = 0
        self._t = 0.0
        # 预分配逐步核的工作/输出数组(避免每步分配)
        self._Vg = np.zeros(self.N_s, dtype=np.float64)
        self._Iarr = np.zeros(N, dtype=np.float64)
        self._Earr = np.zeros(N, dtype=np.float64)
        self._Rarr = np.zeros(N, dtype=np.float64)
        self._B1 = np.zeros(N, dtype=np.float64)
        self._B2 = np.zeros(N, dtype=np.float64)
        self._HVarr = np.zeros(N, dtype=np.float64)
        p = self.params
        self._w1 = self._get_weights(p['alpha1'], Lmax + 1)
        self._w2 = self._get_weights(p['alpha2'], Lmax + 1)
        return self

    # ------------------------------------------------------------------ #
    def state(self):
        """**只读**采集当前状态(不推进、不修改内部状态), 对应闭环流程第 ① 步。

        返回的数组均为**副本**, 调用方修改不会影响模型内部状态。
        若尚未 reset(), 首次调用会按 soc0 自动初始化(仅此一次; 之后不再重置)。

        Returns
        -------
        dict
            "t"      : float      当前累计仿真时间 (s)
            "k"      : int        已推进步数
            "SOC"    : (N_cell,)  逐单体 SOC 比例
            "V1","V2": (N_cell,)  逐单体支路极化电压 (V)
            "H"      : (N_cell,)  逐单体迟滞状态 (0=放电支路, 1=充电支路)
            "N_s","N_p","N_cell" : 拓扑参数
        """
        self._ensure_step_state()
        return {"t": self._t, "k": self._k,
                "SOC": self._soc.copy(), "V1": self._v1.copy(), "V2": self._v2.copy(),
                "V3": float(self._v3), "H": self._h.copy(),
                "N_s": self.N_s, "N_p": self.N_p, "N_cell": self.N_cell}

    # ------------------------------------------------------------------ #
    def predict_step(self, I, T=25.0):
        """预测下一步输出，但不修改在线模型的任何状态。

        BMS在搜索最大允许电流和由功率反求电流时会多次调用本接口。若预测
        过程推进了真实状态，一个EMS周期内SOC和分数阶历史会被重复更新。
        因此这里对全部在线状态、历史环形缓冲和工作数组做快照，并在调用
        ``step`` 后无条件恢复。返回值本身由 ``step`` 创建，和恢复后的数组
        不共享可写内存。
        """
        self._ensure_step_state()
        scalar_names = ("_k", "_t", "_Lmax", "_v3")
        array_names = (
            "_soc", "_v1", "_v2", "_h", "_buf1", "_buf2", "_w1", "_w2",
            "_Vg", "_Iarr", "_Earr", "_Rarr", "_B1", "_B2", "_HVarr",
        )
        scalar_snapshot = {name: getattr(self, name) for name in scalar_names}
        array_snapshot = {
            name: None if getattr(self, name) is None else getattr(self, name).copy()
            for name in array_names
        }
        try:
            output = self.step(I, T=T)
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in output.items()
            }
        finally:
            for name, value in scalar_snapshot.items():
                setattr(self, name, value)
            for name, value in array_snapshot.items():
                setattr(self, name, value)

    # ------------------------------------------------------------------ #
    def step(self, I, T=25.0):
        """**单步推进**一个 dt: 接受电流指令 I, 更新状态并返回本步输出(流程第 ③ 步)。

        状态由 x_k 推进到 x_{k+1}(标准前向欧拉)。
        `I` 为**电堆(每组)总电流**(A), 放电为正、充电为负; `T` 为温度(℃),
        可逐步变化。单步耗时为 O(L*N_cell), L = min(k, memory_limit),
        **与已推进步数无关**(前 L 步因窗口未满更快), 适合固定控制节拍。

        与批处理 run() 的关系
        ---------------------
        run(P) 的第 k 个采样(k >= 1)等于依次 step(P[1]) ... step(P[k]) 后的输出。
        run 的第 0 个采样是"初始时刻仅含欧姆项"的约定值(不消耗电流推进状态),
        故没有对应的 step 调用。

        Returns
        -------
        dict
            "t"       : float      本步结束时刻 (s)
            "k"       : int        推进后的步数
            "V_term"  : float      电堆端电压 (V) = Σ_g V_group
            "V_group" : (N_s,)     各组端电压 (V)
            "I_cell"  : (N_cell,)  逐单体电流 (A), 组内 Σ_c I_cell = I
            "SOC"     : (N_cell,)  逐单体 SOC 比例
            "V1","V2" : (N_cell,)  逐单体支路极化电压 (V)
            "H"       : (N_cell,)  逐单体迟滞状态 (0=放电支路, 1=充电支路)
        """
        self._ensure_step_state()
        self._k += 1
        p = self.params
        core_pack_step_numba(
            self._k, self.dt, float(I), float(T),
            self.N_s, self.N_p, self.N_cell,
            p['R0'], self.a1, self.a2, self.P1, self.P2, self.F1, self.F2, self.cQ,
            p['ocv_off'], self._w1, self._w2, self.memory_limit,
            self.soc_grid, self.t_grid, self.ocv_table,
            self.soc_grid_b, self.t_grid_b, self.ocv_br, self.br_on, self.kh,
            self.hys_i_eps, self.h_ref, self.kh_rest,
            self._soc, self._v1, self._v2, self._h, self._buf1, self._buf2,
            _OCV_ITER,
            self._Vg, self._Iarr, self._Earr, self._Rarr, self._B1, self._B2,
            self._HVarr)
        self._t = self._k * self.dt
        if self.R3_on:
            # 第三条慢支路: 用单体电流递推, 叠加到各组端电压
            self._v3 = self.a3 * self._v3 + self.b3 * (float(I) / float(self.N_p))
            self._Vg += self._v3
        return {"t": self._t, "k": self._k,
                "V_term": float(self._Vg.sum()), "V_group": self._Vg.copy(),
                "I_cell": self._Iarr.copy(),
                "SOC": self._soc.copy(), "V1": self._v1.copy(), "V2": self._v2.copy(),
                "V3": float(self._v3), "H": self._h.copy()}
