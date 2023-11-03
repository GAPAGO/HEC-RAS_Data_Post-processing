# -*- coding: utf-8 -*-
"""
########################################################################################################################
流速
    种子依靠启动流速传播
        设定启动流速阈值 v(m/s)，返回月传播时间百分比
########################################################################################################################
"""
import numpy as np
from tqdm import tqdm
from lib.utils import load_cache_2, cache_2, cache_json
from lib.utils import threshold_breaking_array, monthly_percentage_array


class DistributionSeedFlowVelocityResuspension:
    def __init__(self,
                 file_name="Cells",
                 thresholds=((0.0, 1.0, 0.2),)):
        """
        精简版本
        :param file_name: file name
        :param thresholds: ((velocity: float))
        """
        print("\nSeed Distribution due to Resuspension -- -- Surface Water Velocity\n")
        self.fn = file_name
        self.thresholds = thresholds

        self.tsgm = load_cache_2(f"TSGM_{self.fn}")  # (t,)
        self.cv = load_cache_2(f"CV_{self.fn}")  # (m, n, p)

    def get_pct_s(self, vel):
        bool_array = threshold_breaking_array(self.cv, vel, 0)  # (m, n)[p'] --edit--> (m, n, p)
        pct = monthly_percentage_array(bool_array, self.tsgm)  # (m, n, t)
        return pct

    def calculate(self):
        th_vals = [np.arange(start, end + step, step) for start, end, step in self.thresholds]  # [velocity]
        shape = tuple(len(th_val) for th_val in th_vals) + self.cv.shape[:-1] + self.tsgm.shape  # (velocity, m, n, t)
        result = np.zeros(shape, dtype=np.float32)
        label = {}

        with tqdm(total=np.prod([len(th_val) for th_val in th_vals]),
                  ncols=100) as pbar:
            for indices in np.ndindex(*shape[:1]):
                m_per_s = th_vals[0][indices[0]]
                result[indices] = self.get_pct_s(m_per_s)
                label[str(indices)] = (m_per_s,)
                pbar.update()

        result = np.transpose(result, (1, 2, 3, 0))  # (m, n, t, velocity)
        cache_2(f"DSFVR_{self.fn}", result)  # TODO: save_data
        cache_json(f"DSFVR_{self.fn}", label)  # TODO: save_label


if __name__ == "__main__":
    DSFVR = DistributionSeedFlowVelocityResuspension()
    DSFVR.calculate()
    fn = DSFVR.fn

    print(load_cache_2(f"DSFVR_{fn}"))
    print(load_cache_2(f"DSFVR_{fn}").shape)
