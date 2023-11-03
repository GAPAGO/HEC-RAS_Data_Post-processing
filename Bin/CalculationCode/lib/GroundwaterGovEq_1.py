import numpy as np
from scipy.special import erfc
from scipy.special.cython_special import erfc


class BoussinesqEq1d:
    def __init__(self,
                 mu_d=0.16,
                 k=0.042,
                 h_s=10.0):
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
        :param mu_d: 潜水含水层的单位给水度 μ_d，默认值为简化为常数，粉土：0.16
        :param k: 水力传导系数 (hydraulic conductivity)/ 渗透系数，默认值简化为常数，粉土：0.042 (m/h)
        :param h_s: 土层厚度，大同桑干河段粉土层厚度：10 (m)
        h_mean: 时段始末 H-Z 的平均值：h_s - (地表高程 - 桑干河年均水面高程)
        将地表高程近似为河道最高水位：h_s - (河道最高水位 - 桑干河年均水面高程)
        """
        self.mu_d = mu_d
        self.k = k
        self.h_s = h_s
        self.a = None  # 单位: m^2/h

        self.H_y_t = None  # (m, n)

    def h(self,
          y: np.ndarray,
          t: int,
          wse_series: np.ndarray) -> np.ndarray:
        """
        根据顶托渗漏条件下的一维地下水控制方程：
        h(x,t)=\sum_{0}^{m}(H_{m}-H_{m-1})\cdot e r f c\left({\frac{x}{2{\sqrt{a(t-t_{m})}}}}\right)
        求 t 时刻地下水绝对深度
        :param y: 设定的距离，形状为 (m, n)，单位: m
        :param t: 目标时刻在序列 xss_wse_time_series 中的位置
        :param wse_series: 水位时间序列，形状为 (m, n, p)，单位: (m, 小时)
        :return: 形状为 (m, n) 的数组
        """
        H_min = np.min(wse_series, axis=2)
        H_max = np.max(wse_series, axis=2)
        H_mean = np.mean(wse_series, axis=2)
        h_mean = self.h_s - (H_max - H_mean)
        self.a = self.k * h_mean / self.mu_d  # 单位: m^2/h
        cal_wse_series = np.insert(wse_series, 0, H_min, axis=2)  # (m, n, p + 1)
        diff = np.diff(cal_wse_series, axis=2)  # (m, n, p)
        t_m = np.arange(t)  # t

        sqrt_term = np.sqrt(self.a[..., np.newaxis] * (t - t_m))
        denominator = 2 * sqrt_term
        erfc_arg = y[..., np.newaxis] / denominator
        erfc_term = erfc(erfc_arg)

        value = diff[..., t_m] * erfc_term
        value = value.astype(np.float32)

        self.H_y_t = np.sum(value, axis=2) + H_min
        return self.H_y_t


if __name__ == "__main__":
    ...
