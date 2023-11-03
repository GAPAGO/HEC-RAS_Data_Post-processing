import numpy as np
from scipy.special import erfc


class BoussinesqEq1d:
    def __init__(self,
                 mu_d=0.1,
                 k=3e-5 * 3600,
                 h_s=16.0):
        """
        记 H 为地下水面高程，Z 为隔水底板高程，H-Z 为潜水层厚度
        参数 k 比较敏感，可参考以下公式取值
            k = - κ / μ
            κ = (d_p^2 / 180) * (ε_p^3 / (1 - ε_p)^2)
            ε_p = V_pore / V_tot
        其中
            κ：多孔介质渗透率
            μ：动力黏度
            d_p：有效粒径/球体直径
            ε_p：孔隙率，孔隙体积与总体积的比率
        :param mu_d: 潜水含水层的单位给水度 μ_d，默认值为简化为常数 0.1
        :param k: 水力传导系数 (hydraulic conductivity)/ 渗透系数，默认值简化为常数 (m/h)
        :param h_s: 含水层厚度 (m)
        h_mean: 时段始末 H-Z 的平均值：h_s - (地表高程 - 桑干河年均水面高程)
        将地表高程近似为河道最高水位：h_s - (河道最高水位 - 桑干河年均水面高程)
        """
        self.mu_d = mu_d
        self.k = k
        self.h_s = h_s

    def h(self,
          y: np.ndarray,
          t: int,
          wse_series: np.ndarray,
          gse_series: np.ndarray,
          ) -> np.ndarray:
        """
        顶托渗漏平均入渗一维地下水控制方程：
        \[h\left( {y,t} \right) =
        \sum\limits_0^m {\left[ {\left( {{H_m} - {H_{m - 1}} - {\varepsilon ^ * }_m} \right) \cdot
        {\rm{erfc}}\left( {\frac{y}{{2\sqrt {a\left( {t - {t_m}} \right)} }}} \right) +
        {\varepsilon ^ * }_m} \right]} \]
        求 t 时刻地下水高程
        :param y: 设定的距离，形状为 (m, n)，单位: m
        :param t: 目标时刻在序列 xss_wse_time_series 中的位置
        :param wse_series: 水位时间序列，形状为 (m, n, p)，单位: (m, 小时)
        :param gse_series: 地下水位（无穷）时间序列，形状为 (m, n, p)，单位: (m, 小时)
        :return: 形状为 (m, n) 的数组
        """
        H_init = gse_series[..., 0]  # (m, n)
        gsc_series = gse_series - H_init[..., np.newaxis]  # (m, n, p)
        cal_gsc_series = np.insert(gsc_series, 0, gsc_series[..., 0], axis=-1)  # (m, n, p + 1)
        gsc_diff = np.diff(cal_gsc_series, axis=-1)  # (m, n, p)
        H_max = np.max(wse_series, axis=-1)  # (m, n)
        H_mean = np.mean(wse_series, axis=-1)  # (m, n)
        h_mean = self.h_s - (H_max - H_mean)  # (m, n)
        cal_wse_series = np.insert(wse_series, 0, H_init, axis=-1)  # (m, n, p + 1)
        diff = np.diff(cal_wse_series, axis=-1)  # (m, n, p)
        a = (self.k * h_mean / self.mu_d)[..., np.newaxis]  # (m, n), 单位: m^2/h
        t_m = np.arange(t)  # (t,)
        t_i = np.full(t_m.shape[0], t)
        sqrt_term = np.sqrt(a * (t_i - t_m)[np.newaxis, np.newaxis, ...].astype(np.float32))  # (m, n, t)
        denominator = np.float32(2) * sqrt_term  # (m, n, t)
        erfc_arg = y[..., np.newaxis] / denominator  # (m, n, t)
        erfc_term = erfc(erfc_arg).astype(np.float32)  # (m, n, t)
        value = (diff[..., t_m] - gsc_diff[..., t_m]) * erfc_term + gsc_diff[..., t_m]  # (m, n, t)
        H_y_t = np.sum(value, axis=-1) + H_init  # (m, n)

        return H_y_t


if __name__ == "__main__":
    ...
