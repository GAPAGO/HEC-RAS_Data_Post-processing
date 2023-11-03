# -*- coding: utf-8 -*-
"""
########################################################################################################################
水位
    植被受涝死亡
        设定 cells 淹没深度 ddepth (m) 和淹没时长阈值 dtime (day) 淹没百分率阈值 dpct 设为定值 90% 计算每月超阈值天数百分比
########################################################################################################################
"""
import numpy as np
from tqdm.auto import tqdm
from lib.utils import cache_2, cache_json, load_cache_2
from lib.utils import moving_average, threshold_breaking_array, monthly_percentage_array


class DeathVegetationWaterLevelWaterlogging:
    def __init__(self,
                 file_name="Cells",
                 thresholds=((0.05, 0.2, 0.05), (5, 30, 5), (0.9, 0.9, 0.1))):
        """
        精简版本
        :param file_name: file name
        :param thresholds: ((depth: float), (time: int), (pct: float))
        """
        print("\nVeg Death due to Waterlogging -- -- Surface Water Flooding Depth\n")
        self.fn = file_name
        self.thresholds = thresholds

        self.tsgm = load_cache_2(f"TSGM_{self.fn}")  # time series grouped by month (t,) 值为分月每日索引列表 不等长
        self.chd = load_cache_2(f"CHD_{self.fn}")  # (m, n, p)

    def get_pct_s(self, depth, time, pct):
        avg = moving_average(
            (self.chd > depth).astype(np.float32),
            time)  # (m, n, p - time + 1)  self.chd > depth: (m, n, p)
        bool_array = threshold_breaking_array(avg, pct, time - 1)  # (m, n) --edit--> (m, n, p)
        pct = monthly_percentage_array(bool_array, self.tsgm)  # (m, n, t)
        return pct

    def calculate(self):
        th_vals = [np.arange(start, end + step, step) for start, end, step in self.thresholds]  # [depth, time, pct]
        shape = tuple(len(th_val) for th_val in th_vals) + self.chd.shape[:-1] + self.tsgm.shape  # (depth, time, pct, m, n, t)
        result = np.zeros(shape, dtype=np.float32)
        label = {}

        with tqdm(total=np.prod([len(th_val) for th_val in th_vals]),
                  ncols=100) as pbar:
            for indices in np.ndindex(*shape[:3]):
                meter = th_vals[0][indices[0]]
                hour = th_vals[1][indices[1]] * 24  # TODO: day --> hour
                percent = th_vals[2][indices[2]]
                result[indices] = self.get_pct_s(meter, hour, percent)
                label[str(indices)] = (meter, hour, percent)
                pbar.update()

        result = np.transpose(result, (3, 4, 5, 0, 1, 2))  # (m, n, t, depth, time, pct)
        cache_2(f"DVWLW_{self.fn}", result)  # TODO: save_data
        cache_json(f"DVWLW_{self.fn}", label)  # TODO: save_label


if __name__ == "__main__":
    DVWLW = DeathVegetationWaterLevelWaterlogging()
    DVWLW.calculate()
    fn = DVWLW.fn

    print(load_cache_2(f"DVWLW_{fn}"))
    print(load_cache_2(f"DVWLW_{fn}").shape)
