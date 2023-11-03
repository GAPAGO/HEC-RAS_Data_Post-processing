# -*- coding: utf-8 -*-
"""
########################################################################################################################
剪应力
    种子着床
        返回月峰值剪应力
########################################################################################################################
"""
from lib.utils import load_cache_2, cache_2
from lib.utils import monthly_max


class GerminationSeedShearStressSeedbed:
    def __init__(self, file="Cells"):
        print("\nSeed Germination due to implantation -- -- Surface Water Shear Stress\n")
        self.file = file
        self.tsgm = load_cache_2(f"TSGM_{self.file}")  # (t,)
        self.css = load_cache_2(f"CSS_{self.file}")  # (m, n, p)

    def calculate(self):
        result = monthly_max(self.css, self.tsgm)  # (m, n, t)
        cache_2(f"GSSSS_{self.file}", result)  # TODO: save_data


if __name__ == "__main__":
    GSSSS = GerminationSeedShearStressSeedbed()
    GSSSS.calculate()
    file_name = GSSSS.file

    print(load_cache_2(f"GSSSS_{file_name}"))
    print(load_cache_2(f"GSSSS_{file_name}").shape)
