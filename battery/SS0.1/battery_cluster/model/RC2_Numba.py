import numpy as np
from numba import njit, float64
import time


# 使用 @njit(nopython=True) 强制编译，fastmath=True 允许进行激进的数学优化
@njit(fastmath=True)
def compute_gl_weights_numba(alpha, n_points):
    """
    Numba 加速版的权重计算
    """
    weights = np.zeros(n_points, dtype=np.float64)
    weights[0] = 1.0
    for k in range(1, n_points):
        weights[k] = (1.0 -(alpha +1.0) / k) * weights[k-1]
    return weights

@njit(fastmath=True)
def core_solver_numba(n_steps, dt, current,
                      R0, R1, C1, alpha1,
                      R2, C2, alpha2, OCV,
                      w1,w2,memory_limit):
    """
    Numba 加速的核心求解器(kernel)
    这里没有任何python对象开销,全是C级别的原生速度
    """
    # 1.预计算常数(Pre-calculation)
    dt_pow_a1 = dt ** alpha1
    dt_pow_a2 = dt ** alpha2

    # 公式: V_k = (I/C - sum(w*V_hist)/dt^a) / (1/dt^a + 1/RC)
    factor1 = 1.0 / dt_pow_a1
    term1_den = factor1 + (1.0 / (R1 * C1))

    factor2 = 1.0 / dt_pow_a2
    term2_den = factor2 + (1.0 / (R2 * C2))

    # 2. 初始化数组
    V1 = np.zeros(n_steps, dtype=np.float64)
    V2 = np.zeros(n_steps, dtype=np.float64)
    V_term = np.zeros(n_steps, dtype=np.float64)

    # 3.循环迭代
    for k in range(1, n_steps):
        I_k = current[k]

        # 确定记忆窗口长度
        # 注意：Numba 中min 必须处理相同类型， memory_limit 需要传入整数
        L = k
        if memory_limit > 0 and k > memory_limit:
            L = memory_limit

        # --- 核心化的优点：手动卷积 vs np.dot ---
        # Numba 对切片支持，但为了极致速度，需要确保切片是连续内存
        # 历史电压切片：V1[k-L: k](正序)
        # 权重切片：w1[1: L+1](正序)
        # 卷积逻辑是: sum(w[j] * V[k-j]), 对应 w 正序与 V 倒序的点积

        # 提取切片(Views)
        v1_hist = V1[k-L:k]
        v2_hist = V2[k-L:k]

        # 倒序切片在 Numba 中可能产生拷贝，但这里利用 numpy 的 stride trick 通常很快
        # dot(权重，倒序电压)
        # 注意：w数组我们从索引1开始取，因为w[0]是当前时刻的系数(已移项处理)
        w1_segment= w1[1: L+1]
        w2_segment= w2[1: L+1]

        # 手动点积或者np.dot 均可， Numba 对 np.dot 有指令集优化
        #修改点

        #history1 = np.dot(w1_segment, v1_hist[::-1])
        #history2 = np.dot(w2_segment, v2_hist[::-1])

        # 改进加速：手写循环
        # 1. 计算 history1
        sum1 = 0.0
        # 这里的逻辑是：w 从前往后，v 从后往前
        # w[0]*v[L-1] + w[1]*v[L-2] + ...
        for j in range(L):
            sum1 += w1_segment[j] * v1_hist[L - 1 - j]
        history1 = sum1

        # 2. 计算 history2
        sum2 = 0.0
        for j in range(L):
            sum2 += w2_segment[j] * v2_hist[L - 1 - j]
        history2 = sum2
        # 更新状态
        V1[k] = (I_k / C1 - factor1 * history1) / term1_den
        V2[k] = (I_k / C2 - factor2 * history2) / term2_den

        # 计算端电压
        V_term[k] = OCV - I_k * R0 - V1[k] - V2[k]

    return V_term, V1, V2

# FractionalOrderModel_Numba:
class FractionalOrderModelNumba:
    def __init__(self, dt=0.1, memory_limit=None):
        self.dt = dt
        # 如果 memory_limit 为 None，传入 -1 给 Numba 处理
        self.memory_limit = -1 if memory_limit is None else int(memory_limit)
        self.params = {}
        self.weights_cache = {}

    def set_parameters(self, R0, R1, C1, alpha1, R2, C2, alpha2, OCV):
        self.params = {
            'R0': R0, 'R1': R1, 'C1': C1, 'alpha1': alpha1,
            'R2': R2, 'C2': C2, 'alpha2': alpha2, 'OCV': OCV
        }
        # 清空权重缓存
        self.weights_cache = {}

    def _get_weights(self, alpha, n_points):
        # 缓存机制：避免每次 run 都重新计算权重
        key = (alpha, n_points)
        if key not in self.weights_cache:
            # 调用 Numba 函数计算权重
            self.weights_cache[key] = compute_gl_weights_numba(alpha, n_points)
        return self.weights_cache[key]

    def run(self, current_profile):
        if not self.params:
            raise ValueError("Parameters not set!")


        n_steps = len(current_profile)
        # 确定最大需要的权重长度
        limit_val = n_steps if self.memory_limit == -1 else self.memory_limit
        #修复点
        max_mem = limit_val + 1

        # 获取权重 (Numpy 数组)
        w1 = self._get_weights(self.params['alpha1'], max_mem)
        w2 = self._get_weights(self.params['alpha2'], max_mem)

        # 调用 Numba 核心求解器
        # 注意：这里需要把字典解包成一个个 float 传进去，因为 Numba 处理字典比较慢
        return core_solver_numba(
            n_steps, self.dt, current_profile,
            self.params['R0'],
            self.params['R1'], self.params['C1'], self.params['alpha1'],
            self.params['R2'], self.params['C2'], self.params['alpha2'],
            self.params['OCV'],
            w1, w2,
            self.memory_limit
        )
