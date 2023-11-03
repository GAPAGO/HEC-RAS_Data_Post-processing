# -*- coding: utf-8 -*-
"""
########################################################################################################################
水位
    植被苗期受旱死亡
        设定 cells 的 stime (day) 平均退水率阈值 srate (cm/day) 统计水位为 0 时间序列 计算每月超阈值天数百分比
########################################################################################################################
"""
import numpy as np
from tqdm.auto import tqdm
from lib.utils import cache_2, cache_json, load_cache_2
from lib.utils import moving_average, threshold_breaking_array, mask_based_remove_array, monthly_percentage_array


class DeathSeedlingWaterLevelDrought:
    def __init__(self,
                 file_name="Cells",
                 thresholds=((5, 30, 5), (0.5, 5.0, 0.5))):
        """
        精简版本
        :param file_name: file name
        :param thresholds: ((time: int), (pct: float))
        """
        print("\nSeedling Death due to Drought -- -- Decline Rate of Underground Water Level\n")
        self.fn = file_name
        self.thresholds = thresholds  # (time, rate)

        self.tsgm = load_cache_2(f"TSGM_{self.fn}")  # time series grouped by month (t,) 值为分月每日索引列表 不等长
        self.cubd = load_cache_2(f"CUBD_{self.fn}")  # (m, n, p)
        self.chd = load_cache_2(f"CHD_{self.fn}")  # (m, n, p)

    def get_pct_s(self, time, rate):
        mask = threshold_breaking_array(self.chd, 0, 0)  # (m, n)[p'] --edit--> (m, n, p)

        cubd = np.insert(self.cubd, 0, 0, axis=-1)
        diff = np.diff(cubd, axis=-1)  # (m, n, p)
        avg = moving_average(diff, time)  # (m, n, p - time + 1)
        bool_array = threshold_breaking_array(avg, rate, time - 1)  # (m, n)[p''] --edit--> (m, n, p)
        bool_array = mask_based_remove_array(bool_array, mask)  # (m, n)[p'''] --edit--> (m, n, p)
        pct = monthly_percentage_array(bool_array, self.tsgm)  # (m, n, t)
        return pct

    def calculate(self):
        th_vals = [np.arange(start, end + step, step) for start, end, step in self.thresholds]  # [time, rate]
        shape = tuple(len(th_val) for th_val in th_vals) + self.chd.shape[:-1] + self.tsgm.shape  # (time, rate, m, n, t)
        result = np.zeros(shape, dtype=np.float32)
        label = {}

        with tqdm(total=np.prod([len(th_val) for th_val in th_vals]),
                  ncols=100) as pbar:
            for indices in np.ndindex(*shape[:2]):
                hour = th_vals[0][indices[0]] * 24  # TODO: day --> hour
                m_per_hour = th_vals[1][indices[1]] / 2400  # TODO: day --> hour
                result[indices] = self.get_pct_s(hour, m_per_hour)
                label[str(indices)] = (hour, m_per_hour)
                pbar.update()

        result = np.transpose(result, (2, 3, 4, 0, 1))  # (m, n, t, time, rate)
        cache_2(f"DSWLD_{self.fn}", result)  # TODO: save_data
        cache_json(f"DSWLD_{self.fn}", label)  # TODO: save_label


if __name__ == "__main__":
    DSWLD = DeathSeedlingWaterLevelDrought()
    DSWLD.calculate()
    fn = DSWLD.fn

    print(load_cache_2(f"DSWLD_{fn}"))
    print(load_cache_2(f"DSWLD_{fn}").shape)
