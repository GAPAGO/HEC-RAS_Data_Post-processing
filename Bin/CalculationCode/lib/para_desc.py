class ParameterDescription:
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
