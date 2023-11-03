"""
########################################################################################################################
水位
    植被受旱死亡
            cells地下水位主要与河道的距离有关，求解一维地下水控制方程
            设定 cells 干旱深度 yheight (m) 和干旱时长阈值 ytime (day) 干旱百分率阈值 ypct 设为定值 90% 计算每月超阈值天数百分比
########################################################################################################################
"""
import os
import sys
lib_dir = os.path.join(os.getcwd(), 'lib')
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
n = os.cpu_count() - 1
import numpy as np
import pandas as pd
from lib.utils import root_dir
from lib.utils import cache_2, cache_json, load_cache_2
from lib.utils import moving_average, threshold_breaking_array, monthly_percentage_array
from lib.para_desc import ParameterDescription
from boussinesq_eq1d import h  # C
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor


class DeathVegetationWaterLevelDrought(ParameterDescription):
    def __init__(self,
                 file_name="Cells",
                 thresholds=((0.5, 3.5, 0.5), (5, 30, 5), (0.9, 0.9, 0.1)),
                 **kwargs):
        """
        精简版本
        :param file_name: file name
        :param thresholds: ((height: float), (time: int), (pct: float))
        """
        super().__init__(**kwargs)
        print("\nVeg Death due to Drought -- -- Underground Water Burial Depth\n")
        self.fn = file_name
        self.thresholds = thresholds

        self.xsse = load_cache_2(f"XSSE_{self.fn}")  # cross section surface elevation 河道水位 (m, n, p)
        self.ptts = load_cache_2(f"PTTS_{self.fn}")  # phreatic thickness time series 潜水含水层厚度边界条件 (m, n, p)
        self.cbd = load_cache_2(f"CBD_{self.fn}")  # cell bank distance 网格河道距离 (m, n)
        self.cme = load_cache_2(f"CME_{file_name}")  # cell minimum elevation 网格地面高程 (m, n)
        self.cie = load_cache_2(f"CIE_{file_name}")  # cell impervious elevation 网格不透水面高程 (m, n)
        self.ccc = load_cache_2(f"CCC_{file_name}")  # cell center coordination 网格中心坐标 (m, n, 2)
        self.tsgm = load_cache_2(f"TSGM_{self.fn}")  # time series grouped by month (t,) 值为分月每日索引列表 不等长

        # 初始化地下水位和埋深
        self.cuse = np.zeros(self.xsse.shape, dtype=np.float32)  # cell underground water surface elevation (m, n, p)
        self.cubd = np.zeros(self.xsse.shape, dtype=np.float32)  # cell underground water burial depth (m, n, p)
        # 目标时间索引
        self.time_idx_series = np.arange(0, self.xsse.shape[-1], dtype=np.int32)  # (p,)
        # 参考 MODFLOW 代码 回归地下倾斜不透水平面高程
        self.gse_series = self.cie[..., np.newaxis] + self.ptts  # groundwater surface elevation 生成边界条件

    @staticmethod
    def np_3d_export_csv(result: np.ndarray, string: str):
        for j in range(result.shape[1]):  # 对每个漫滩循环
            df = pd.DataFrame(result[:, j, :])
            df.to_csv(os.path.join(root_dir, f"Bin\\CalculationCode\\cache\\{string}_{j + 1}.csv"), index=False)

    @staticmethod
    def worker(args):
        cbd, t, rgse_diff, a, diff = args
        if t == 0:
            return 0
        else:
            return h(cbd, t, rgse_diff, a, diff)  # 核心计算代码 C

    def preprocessing(self):
        H_init = self.gse_series[..., 0]  # (m, n)
        rgse_series = self.gse_series - H_init[..., np.newaxis]  # (m, n, p)
        cal_rgse_series = np.insert(rgse_series, 0, rgse_series[..., 0], axis=-1)  # (m, n, p + 1)
        rgse_diff = np.diff(cal_rgse_series, axis=-1)  # (m, n, p)
        H_max = np.max(self.xsse, axis=-1)  # (m, n)
        H_mean = np.mean(self.xsse, axis=-1)  # (m, n)
        h_mean = self.h_s - (H_max - H_mean)  # (m, n)
        cal_wse_series = np.insert(self.xsse, 0, H_init, axis=-1)  # (m, n, p + 1)
        diff = np.diff(cal_wse_series, axis=-1)  # (m, n, p)
        a = (self.k * h_mean / self.mu_d)[..., np.newaxis]  # (m, n), m^2 / h
        return rgse_diff, a, diff

    def postprocessing(self, value):
        return value + self.gse_series[..., 0][..., np.newaxis]  # (m, n)

    def cal_CellUndergroundWaterSurfaceElevation(self):
        rgse_diff, a, diff = self.preprocessing()
        with ThreadPoolExecutor(max_workers=n) as executor:
            args = [(self.cbd, t, rgse_diff, a, diff) for t in self.time_idx_series]
            results = list(tqdm(executor.map(self.worker, args), total=len(args), ncols=100))  # 使用线程池
            for t, result in zip(self.time_idx_series, results):
                self.cuse[..., t] = result
        self.cuse = self.postprocessing(self.cuse)
        cache_2(f"CUSE_{self.fn}", self.cuse)
        self.np_3d_export_csv(self.cuse, f"CUSE_{self.fn}")  # TODO: SAVE

    def cal_CellUndergroundWaterBurialDepth(self):
        self.cubd = self.cme[..., np.newaxis] - self.cuse  # (m, n, p)
        self.cubd = np.clip(self.cubd, 0, None)
        cache_2(f"CUBD_{self.fn}", self.cubd)
        DeathVegetationWaterLevelDrought.np_3d_export_csv(self.cubd, f"CUBD_{self.fn}")  # TODO: SAVE

    def get_pct_s(self, height, time, pct, enable_load_cubd: bool = False):
        if enable_load_cubd:
            self.cubd = load_cache_2(f"CUBD_{self.fn}")
        desiccation = self.cubd > height  # (m, n, p)
        avg = moving_average(desiccation, time)  # (m, n, p - time + 1)
        bool_array = threshold_breaking_array(avg, pct, time - 1)  # (m, n) 值为 不定长超过阈值的一维索引数组
        pct = monthly_percentage_array(bool_array, self.tsgm)  # (m, n, t)
        return pct

    def calculation(self):
        th_vals = [np.arange(start, end + step, step) for start, end, step in self.thresholds]  # [height, time, pct]
        shape = tuple(len(th_val) for th_val in th_vals) + self.cme.shape + self.tsgm.shape  # (height, time, pct, m, n, t)
        result = np.zeros(shape, dtype=np.float32)
        label = {}

        def update_result(indices):
            meter = th_vals[0][indices[0]]
            hour = th_vals[1][indices[1]] * 24  # TODO: day --> hour
            percent = th_vals[2][indices[2]]
            result[indices] = self.get_pct_s(meter, hour, percent)
            label[str(indices)] = (meter, hour, percent)

        with ThreadPoolExecutor(n) as executor:
            list(tqdm(executor.map(update_result, np.ndindex(*shape[:3])),
                      total=np.prod([len(th_val) for th_val in th_vals]),
                      ncols=100))

        result = np.transpose(result, (3, 4, 5, 0, 1, 2))  # (m, n, t, height, time, pct)
        cache_2(f"DVWLD_{self.fn}", result)  # TODO: save_data
        cache_json(f"DVWLD_{self.fn}", label)  # TODO: save_label


if __name__ == "__main__":
    ...
