"""
RC2_SOC.py
==========
分数阶二阶 RC 等效电路模型 —— 带 SOC 状态与温度相关 OCV 的版本。

与 RC2_Numba.py 中 FractionalOrderModelNumba 的关系
---------------------------------------------------
本模块 **新增** FractionalOrderModelSOC 类, 原类与原接口不做任何修改。
两者的分数阶核心(Grünwald-Letnikov 权重递推 + 短记忆截断)完全一致,
区别在于本类:

  1. 新增 SOC 状态(安时积分), 构成状态空间 {SOC, V1, V2};
  2. OCV 不再为常数, 而由 CSV 表格 OCV(SOC, T) 双线性插值给出;
  3. 支持串联节数 num_series, 堆端电压 = num_series × 单体端电压。

CSV 表约定
----------
lfp_ocv_soc.csv: 首列为 **DOD**(0%..100%), 其余各列为不同温度(℃)下的
**单体** OCV(V)。SOC 与 DOD 互补:  SOC = 1 - DOD。


状态空间定义
------------
    状态   x = [SOC, V1, V2]
             SOC : 荷电状态, 比例 [0, 1]
             V1  : 支路 1 极化电压(单体), 分数阶, 含历史记忆
             V2  : 支路 2 极化电压(单体), 分数阶, 含历史记忆
    输入   u = [I, T]
             I   : 电流(A), 约定放电为正、充电为负
             T   : 温度(℃), 标量常数
    输出   y = V_term, 堆端电压(V) = num_series × 单体端电压

逐时刻方程(k = 0,1,...):
    单体端电压  v_cell[k] = OCV_cell(SOC[k], T) - I[k]*R0 - V1[k] - V2[k]
    堆端电压    V_term[k] = num_series * v_cell[k]
    SOC 递推    SOC[k]    = SOC[k-1] - eta * I[k] * dt / (3600 * Q)
    支路电压    V_i[k]    = (I[k]/C_i - (1/dt^a_i) * Σ_{j=1..L} w_i[j] * V_i[k-j])
                            / (1/dt^a_i + 1/(R_i*C_i))
    记忆窗口    L = min(k, memory_limit)   (memory_limit<=0 表示全记忆)
"""

import os

import numpy as np
from numba import njit

from RC2_Numba import compute_gl_weights_numba

# 默认 OCV(SOC, T) 表路径(不含任何绝对路径), 按顺序尝试:
#   1) 与本模块同目录的 lfp_ocv_soc.csv  —— 交付包内的布局
#   2) ../parameter/lfp_ocv_soc.csv      —— 仓库原始布局
_HERE = os.path.dirname(os.path.abspath(__file__))
_OCV_CANDIDATES = [
    os.path.join(_HERE, "lfp_ocv_soc.csv"),
    os.path.join(_HERE, os.pardir, "parameter", "lfp_ocv_soc.csv"),
]
DEFAULT_OCV_CSV = next((p for p in _OCV_CANDIDATES if os.path.exists(p)),
                       _OCV_CANDIDATES[0])


# --------------------------------------------------------------------------- #
# OCV 表读取
# --------------------------------------------------------------------------- #
def load_ocv_table(csv_path=DEFAULT_OCV_CSV):
    """读取 OCV(SOC, T) 表。

    参数
    ----
    csv_path : str
        首列为 DOD(如 "0%"), 其余各列为温度(℃), 单元格为单体 OCV(V)。

    返回
    ----
    soc_grid  : (n_soc,)  SOC 比例 [0,1], 升序
    t_grid    : (n_temp,) 温度 ℃, 升序
    ocv_table : (n_soc, n_temp) 单体 OCV(V), C 连续

    说明: 0% DOD 对应满电(高电压), 故 SOC = 100 - DOD。
    """
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")          # 去掉可能存在的空列

    dod = (df.iloc[:, 0].astype(str)
           .str.replace("%", "", regex=False).astype(float).to_numpy())
    soc_pct = 100.0 - dod                       # DOD -> SOC(%)

    temps = np.array([float(str(c).replace("℃", "")) for c in df.columns[1:]])
    ocv = df.iloc[:, 1:].astype(float).to_numpy()   # (n_dod, n_temp)

    # 行列分别按 SOC、温度升序排列, 便于内核做有序区间定位
    order = np.argsort(soc_pct)
    soc_grid = soc_pct[order] / 100.0
    ocv = ocv[order, :]

    torder = np.argsort(temps)
    t_grid = temps[torder]
    ocv = ocv[:, torder]

    return (np.ascontiguousarray(soc_grid, dtype=np.float64),
            np.ascontiguousarray(t_grid, dtype=np.float64),
            np.ascontiguousarray(ocv, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Numba 内核
# --------------------------------------------------------------------------- #
@njit(fastmath=True)
def _ocv_bilinear(soc, T, soc_grid, t_grid, ocv_table):
    """单体 OCV 双线性插值; SOC/温度越界时按边界值截断(不外推)。"""
    n_soc = soc_grid.shape[0]
    n_t = t_grid.shape[0]

    # --- SOC 定位 ---
    if soc <= soc_grid[0]:
        i0 = 0
        i1 = 0
        ws = 0.0
    elif soc >= soc_grid[n_soc - 1]:
        i0 = n_soc - 1
        i1 = n_soc - 1
        ws = 0.0
    else:
        i0 = 0
        for j in range(n_soc - 1):
            if soc >= soc_grid[j] and soc < soc_grid[j + 1]:
                i0 = j
                break
        i1 = i0 + 1
        ws = (soc - soc_grid[i0]) / (soc_grid[i1] - soc_grid[i0])

    # --- 温度定位 ---
    if T <= t_grid[0]:
        j0 = 0
        j1 = 0
        wt = 0.0
    elif T >= t_grid[n_t - 1]:
        j0 = n_t - 1
        j1 = n_t - 1
        wt = 0.0
    else:
        j0 = 0
        for j in range(n_t - 1):
            if T >= t_grid[j] and T < t_grid[j + 1]:
                j0 = j
                break
        j1 = j0 + 1
        wt = (T - t_grid[j0]) / (t_grid[j1] - t_grid[j0])

    v00 = ocv_table[i0, j0]
    v01 = ocv_table[i0, j1]
    v10 = ocv_table[i1, j0]
    v11 = ocv_table[i1, j1]
    v0 = v00 + (v01 - v00) * wt
    v1 = v10 + (v11 - v10) * wt
    return v0 + (v1 - v0) * ws


@njit(fastmath=True)
def core_solver_soc_numba(n_steps, dt, current, T,
                          R0, R1, C1, alpha1,
                          R2, C2, alpha2,
                          Q, soc0, eta, num_series,
                          w1, w2, memory_limit,
                          soc_grid, t_grid, ocv_table):
    """SOC + 分数阶二阶 RC 求解内核(全 C 级原生速度)。

    返回 (V_term, SOC, V1, V2); 其中 V1/V2 为单体支路极化电压。
    """
    # 1. 预计算常数
    dt_pow_a1 = dt ** alpha1
    dt_pow_a2 = dt ** alpha2

    factor1 = 1.0 / dt_pow_a1
    term1_den = factor1 + (1.0 / (R1 * C1))

    factor2 = 1.0 / dt_pow_a2
    term2_den = factor2 + (1.0 / (R2 * C2))

    # 2. 初始化状态/输出数组
    SOC = np.zeros(n_steps, dtype=np.float64)
    V1 = np.zeros(n_steps, dtype=np.float64)
    V2 = np.zeros(n_steps, dtype=np.float64)
    V_term = np.zeros(n_steps, dtype=np.float64)

    # 初始状态: soc0, V1=V2=0
    SOC[0] = soc0
    ocv0 = _ocv_bilinear(soc0, T, soc_grid, t_grid, ocv_table)
    V_term[0] = num_series * (ocv0 - current[0] * R0)

    # 3. 逐步递推
    soc_prev = soc0
    for k in range(1, n_steps):
        I_k = current[k]

        # --- SOC 安时积分 ---
        soc_k = soc_prev - eta * I_k * dt / (3600.0 * Q)
        SOC[k] = soc_k

        # --- 记忆窗口 ---
        L = k
        if memory_limit > 0 and k > memory_limit:
            L = memory_limit

        # --- 历史卷积(手写循环, w 正序 × V 倒序) ---
        v1_hist = V1[k - L:k]
        v2_hist = V2[k - L:k]
        w1_segment = w1[1:L + 1]
        w2_segment = w2[1:L + 1]

        sum1 = 0.0
        for j in range(L):
            sum1 += w1_segment[j] * v1_hist[L - 1 - j]
        sum2 = 0.0
        for j in range(L):
            sum2 += w2_segment[j] * v2_hist[L - 1 - j]

        # --- 分数阶支路状态更新(单体) ---
        V1[k] = (I_k / C1 - factor1 * sum1) / term1_den
        V2[k] = (I_k / C2 - factor2 * sum2) / term2_den

        # --- 端电压(堆级 = num_series × 单体) ---
        ocv_k = _ocv_bilinear(soc_k, T, soc_grid, t_grid, ocv_table)
        V_term[k] = num_series * (ocv_k - I_k * R0 - V1[k] - V2[k])

        soc_prev = soc_k

    return V_term, SOC, V1, V2


# --------------------------------------------------------------------------- #
# 模型类
# --------------------------------------------------------------------------- #
class FractionalOrderModelSOC:
    """分数阶二阶 RC 模型(含 SOC 状态与温度相关 OCV)。

    Parameters
    ----------
    dt : float
        仿真步长(s)。
    memory_limit : int or None
        短记忆窗口长度(步); None 表示全记忆。
    num_series : int
        串联节数 N, 堆端电压 = N × 单体端电压。
    ocv_csv_path : str
        OCV(SOC, T) 表路径; 见 load_ocv_table。
    """

    def __init__(self, dt=0.1, memory_limit=None, num_series=1,
                 ocv_csv_path=DEFAULT_OCV_CSV):
        self.dt = dt
        self.memory_limit = -1 if memory_limit is None else int(memory_limit)
        self.num_series = int(num_series)
        self.params = {}
        self.weights_cache = {}
        # 预读 OCV 表(与仿真步数、参数无关)
        self.soc_grid, self.t_grid, self.ocv_table = load_ocv_table(ocv_csv_path)

    # ------------------------------------------------------------------ #
    def set_parameters(self, R0, R1, C1, alpha1, R2, C2, alpha2,
                       Q, soc0=1.0, eta=1.0):
        """设置电路参数与 SOC 参数。

        R0          欧姆内阻(Ω, 单体)
        R1,C1,alpha1 支路 1 的电阻(Ω)、分数阶电容、阶数
        R2,C2,alpha2 支路 2 的电阻(Ω)、分数阶电容、阶数
        Q           容量(Ah)
        soc0        初始 SOC, 比例 [0, 1]
        eta         库仑效率(0~1]
        """
        self.params = {
            'R0': R0, 'R1': R1, 'C1': C1, 'alpha1': alpha1,
            'R2': R2, 'C2': C2, 'alpha2': alpha2,
            'Q': Q, 'soc0': soc0, 'eta': eta,
        }
        self.weights_cache = {}     # 参数变更 -> 权重缓存失效

    # ------------------------------------------------------------------ #
    def _get_weights(self, alpha, n_points):
        key = (alpha, n_points)
        if key not in self.weights_cache:
            self.weights_cache[key] = compute_gl_weights_numba(alpha, n_points)
        return self.weights_cache[key]

    # ------------------------------------------------------------------ #
    def ocv(self, soc, T=25.0):
        """外部查询 OCV(堆级)。

        soc : 标量或数组, SOC 比例 [0, 1]; 越界按边界截断。
        T   : 温度(℃), 标量。
        返回: 与 soc 同形; soc 为标量时返回 float。
        """
        scalar = np.isscalar(soc)
        s = np.clip(np.atleast_1d(np.asarray(soc, dtype=float)),
                    self.soc_grid[0], self.soc_grid[-1])
        # 先在温度维插值(每行一个 SOC 采样点), 再在 SOC 维插值
        row = np.array([np.interp(T, self.t_grid, self.ocv_table[i, :])
                        for i in range(self.ocv_table.shape[0])])
        v = self.num_series * np.interp(s, self.soc_grid, row)
        return float(v[0]) if scalar else v

    # ------------------------------------------------------------------ #
    def run(self, current_profile, T=25.0):
        """执行仿真。

        current_profile : 电流序列(A), 放电为正、充电为负。
        T               : 温度(℃), 标量常数。

        返回 (V_term, SOC, V1, V2):
            V_term : 堆端电压(V)
            SOC    : 荷电状态比例 [0, 1]
            V1, V2 : 单体支路极化电压(V)
        """
        if not self.params:
            raise ValueError("Parameters not set!")

        current = np.ascontiguousarray(current_profile, dtype=np.float64)
        n_steps = current.shape[0]
        if n_steps == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty, empty

        # 所需权重长度
        limit_val = n_steps if self.memory_limit == -1 else self.memory_limit
        max_mem = limit_val + 1
        w1 = self._get_weights(self.params['alpha1'], max_mem)
        w2 = self._get_weights(self.params['alpha2'], max_mem)

        p = self.params
        return core_solver_soc_numba(
            n_steps, self.dt, current, float(T),
            p['R0'], p['R1'], p['C1'], p['alpha1'],
            p['R2'], p['C2'], p['alpha2'],
            p['Q'], p['soc0'], p['eta'], self.num_series,
            w1, w2, self.memory_limit,
            self.soc_grid, self.t_grid, self.ocv_table)
