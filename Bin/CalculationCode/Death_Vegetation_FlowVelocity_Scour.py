# -*- coding: utf-8 -*-
"""
########################################################################################################################
流速
    植被受冲刷死亡
        返回月最大流速
########################################################################################################################
"""
from lib.utils import load_cache_2, cache_2
from lib.utils import monthly_max


class DeathVegetationFlowVelocityScour:
    def __init__(self, file_name="Cells"):
        print("\nVeg Death due to Scour -- -- Surface Water Velocity\n")
        self.fn = file_name

        self.tsgm = load_cache_2(f"TSGM_{self.fn}")  # time series grouped by month (t,) 值为分月每日索引列表 不等长
        self.cv = load_cache_2(f"CV_{self.fn}")  # (m, n, p)

    def calculate(self):
        result = monthly_max(self.cv, self.tsgm)  # (m, n, t)
        cache_2(f"DVFVS_{self.fn}", result)  # TODO: save_data


if __name__ == "__main__":
    DVFVS = DeathVegetationFlowVelocityScour()
    DVFVS.calculate()
    fn = DVFVS.fn

    print(load_cache_2(f"DVFVS_{fn}"))
    print(load_cache_2(f"DVFVS_{fn}").shape)
