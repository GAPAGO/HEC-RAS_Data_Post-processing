# -*- coding: utf-8 -*-
"""
########################################################################################################################
流速
    水体携氧植被生长
        返回有水情况下的月平均流速
########################################################################################################################
"""
from lib.utils import load_cache_2, cache_2
from lib.utils import threshold_breaking_array, mask_based_remove_array, monthly_screen_average_array


class GrowthVegetationFlowVelocityRespiration:
    def __init__(self, file="Cells"):
        print("\nVeg Growth due to DO -- -- Surface Water Velocity\n")
        self.file = file

        self.tsgm = load_cache_2(f"TSGM_{self.file}")  # (t,)
        self.chd = load_cache_2(f"CHD_{self.file}")  # (m, n, p)
        self.cv = load_cache_2(f"CV_{self.file}")  # (m, n, p)

    def calculate(self):
        mask = threshold_breaking_array(self.chd, 0, 0)  # (m, n)[p'] --edit--> (m, n, p)
        bool_array = threshold_breaking_array(self.cv, -1, 0)  # (m, n)[p''] 仅格式转化 --edit--> (m, n, p)
        bool_array = mask_based_remove_array(bool_array, mask)  # (m, n)[p'''] --edit--> (m, n, p)

        result = monthly_screen_average_array(self.cv, self.tsgm, bool_array)  # (m, n, t)
        cache_2(f"GVFVR_{self.file}", result)  # TODO: save_data


if __name__ == "__main__":
    GVFVR = GrowthVegetationFlowVelocityRespiration()
    GVFVR.calculate()
    file_name = GVFVR.file

    print(load_cache_2(f"GVFVR_{file_name}"))
    print(load_cache_2(f"GVFVR_{file_name}").shape)
