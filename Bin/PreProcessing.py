# -*- coding: utf-8 -*-
"""
提取与拼接
"""
import numpy as np
import pandas as pd
from datetime import datetime
from CalculationCode.conf import var
from CalculationCode.conf.var import *
from CalculationCode.lib.utils import get_file_name, cache_1, load_cache_1, open_HDF_file, cache_2, load_cache_2
from CalculationCode.lib.utils import cell_to_face, filter_facepoints
from CalculationCode.lib.utils import polyline_segment
from CalculationCode.lib.utils import get_LocationOnCPolyline, get_NearestSectionData
from CalculationCode.lib.utils import get_DistanceToBEPolyline
from CalculationCode.lib.Decorators import timer
from scipy.linalg import lstsq
from pathlib import Path


def get_location_df(cp: str) -> pd.DataFrame:
    """
    读取 csv 文件并返回一个 DataFrame
    :param cp: csv文件名
    :return: 修改全局变量 mask，返回数据框 df
    """
    df = pd.read_csv(cp)
    var.MASK = df.isna().values
    df = df.fillna(0).astype("int32")
    print(f"\nCells DataFrame Shape: {df.shape}, Mask Generated!\n")
    return df


def get_face_fp_np(rhfp: str, lcs: pd.DataFrame) -> list[np.ndarray, np.ndarray]:
    cf = np.zeros(lcs.shape, dtype=list)
    cfp = np.zeros(lcs.shape, dtype=list)
    for j, (l, cs) in enumerate(lcs.items()):
        cell_face_map = cell_to_face(rhfp, l)
        cell_facepoint_map = filter_facepoints(rhfp, l)
        for i, c in cs.items():
            face_indices = cell_face_map[c]
            facepoint_indices = cell_facepoint_map[c]
            cf[i, j] = face_indices
            cfp[i, j] = facepoint_indices
    return [cf, cfp]


@timer
def cal_CellHydraulicDepth(hfp: str, lcs: pd.DataFrame, file):
    data_file = open_HDF_file(hfp)
    r = None
    for j, (l, cs) in enumerate(lcs.items()):
        data_path = result_2d + l + "/Cell Hydraulic Depth"
        data = data_file[data_path]
        dnp = np.array(data)
        if r is None:
            r = np.zeros(lcs.shape + (dnp.shape[0],), dtype=np.float32)
        for i, c in cs.items():
            ts = dnp[:, c]
            r[i, j, :] = ts
    print(f"Cell Hydraulic Depth Shape of {hfp}: {r.shape}")
    cache_1(hfp, f"CHD_{file}", r)


@timer
def cal_CellShearStress(hfp: str, lcs: pd.DataFrame, lfs: np.ndarray, file):
    manning = 0.04  # TODO: edit
    density = 998.2  # TODO: edit
    gravity = 9.801  # TODO: edit
    data_file = open_HDF_file(hfp)
    r = None
    for j, l in enumerate(lcs.columns):
        chdp = result_2d + l + "/Cell Hydraulic Depth"
        R_cell = np.array(data_file[chdp]).T

        fcip = geometry_2d + l + "/Faces Cell Indexes"
        fci = np.array(data_file[fcip])

        fnvp = result_2d + l + "/Face Velocity"
        ftvp = result_2d + l + "/Face Tangential Velocity"
        fnv = np.array(data_file[fnvp])
        ftv = np.array(data_file[ftvp])

        nv_length = geometry_2d + l + "/Faces NormalUnitVector and Length"
        normal_vector = data_file[nv_length][:, :2]
        length = data_file[nv_length][:, -1]

        tangent_vector = np.zeros_like(normal_vector)
        tangent_vector[:, 0] = -normal_vector[:, 1]
        tangent_vector[:, 1] = normal_vector[:, 0]

        od_normal = fnv[..., np.newaxis] * normal_vector[np.newaxis, ...]
        od_tangent = ftv[..., np.newaxis] * tangent_vector[np.newaxis, ...]
        od_vel = od_normal + od_tangent

        R_face = ((R_cell[fci[:, 0], :] + R_cell[fci[:, 1], :]) / 2).T
        R_face_power = np.where(R_face != 0, R_face ** (1 / 3), 0.00001)
        C_D = np.where(R_face != 0, manning ** 2 * gravity / R_face_power, 0)
        od_fss = density * C_D[..., np.newaxis] * np.abs(od_vel) * od_vel

        if r is None:
            r = np.zeros(lcs.shape + (od_fss.shape[0],), dtype=np.float32)
        for i, fs in enumerate(lfs[:, j]):
            t = np.sqrt(
                (np.sum(od_fss[:, fs, 0] * length[fs][np.newaxis, :], axis=1) ** 2 +
                 np.sum(od_fss[:, fs, 1] * length[fs][np.newaxis, :], axis=1) ** 2)) / np.sum(length[fs])
            r[i, j, :] = t

    print(f"Cell Shear Stress Shape of {hfp}: {r.shape}")
    cache_1(hfp, f"CSS_{file}", r)


@timer
def cal_CellVelocity(hfp: str, lcs: pd.DataFrame, lfs: np.ndarray, file):
    data_file = open_HDF_file(hfp)
    r = None
    for j, l in enumerate(lcs.columns):
        fnvp = result_2d + l + "/Face Velocity"
        ftvp = result_2d + l + "/Face Tangential Velocity"
        fnv = np.array(data_file[fnvp])
        ftv = np.array(data_file[ftvp])

        nv_length = geometry_2d + l + "/Faces NormalUnitVector and Length"
        normal_vector = data_file[nv_length][:, :2]
        length = data_file[nv_length][:, -1]

        tangent_vector = np.zeros_like(normal_vector)
        tangent_vector[:, 0] = -normal_vector[:, 1]
        tangent_vector[:, 1] = normal_vector[:, 0]

        od_normal = fnv[..., np.newaxis] * normal_vector[np.newaxis, ...]
        od_tangent = ftv[..., np.newaxis] * tangent_vector[np.newaxis, ...]
        od_vel = od_normal + od_tangent

        if r is None:
            r = np.zeros(lcs.shape + (od_vel.shape[0],), dtype=np.float32)
        for i, fs in enumerate(lfs[:, j]):
            t = np.sqrt(
                (np.sum(od_vel[:, fs, 0] * length[fs][np.newaxis, :], axis=1) ** 2 +
                 np.sum(od_vel[:, fs, 1] * length[fs][np.newaxis, :], axis=1) ** 2)) / np.sum(length[fs])
            r[i, j, :] = t

    print(f"Cell Velocity Shape of {hfp}: {r.shape}")
    cache_1(hfp, f"CV_{file}", r)


@timer
def cal_CellMinimumElevation(rhfp: str, lcs: pd.DataFrame, file):
    result = np.zeros(lcs.shape, dtype=np.float32)
    for j, (l, cs) in enumerate(lcs.items()):
        data_path = geometry_2d + l + "/Cells Minimum Elevation"
        dnp = np.array(open_HDF_file(rhfp)[data_path])
        for i, c in enumerate(cs):
            result[i, j] = dnp[c]
    print(f"Cells Minimum Elevation Shape: {result.shape}")
    cache_2(f"CME_{file}", result)


@timer
def cal_CellCenterCoordinate(rhfp: str, lcs: pd.DataFrame, file):
    result = np.zeros(lcs.shape + (2,), dtype=np.float32)
    for j, (l, cs) in enumerate(lcs.items()):
        data_path = geometry_2d + l + "/Cells Center Coordinate"
        dnp = np.array(open_HDF_file(rhfp)[data_path])
        for i, c in enumerate(cs):
            result[i, j] = dnp[c, :]
    print(f"Cells Center Coordinate Shape: {result.shape}")
    cache_2(f"CCC_{file}", result)


@timer
def cal_CellToBankDistance(rhfp: str, file):
    ccc = load_cache_2(f"CCC_{file}")
    river_bank_lines_coord_datas = polyline_segment(rhfp, river_bank_lines_coord_path, river_bank_lines_seg_path)
    dist_array = get_DistanceToBEPolyline(river_bank_lines_coord_datas, ccc)
    print(f"Cells Bank Distance Shape: {dist_array.shape}")
    cache_2(f"CBD_{file}", dist_array)


@timer
def cal_CellImperviousElevation(lcs: pd.DataFrame, file, depth=16.0):
    cme = load_cache_2(f"CME_{file}")  # (m, n)
    ccc = load_cache_2(f'CCC_{file}')  # (m, n, 2)
    botms = np.zeros_like(lcs, dtype=np.float32)
    for j in range(lcs.shape[1]):
        cme_j = cme[~var.MASK[:, j], j]  # (p,)
        ccc_j = ccc[~var.MASK[:, j], j]  # (p, 2)
        A = np.c_[ccc_j[:, 0], ccc_j[:, 1], np.ones(cme_j.size)]
        C, _, _, _ = lstsq(A, cme_j)
        botm = C[0] * ccc_j[:, 0] + C[1] * ccc_j[:, 1] + C[2] - depth
        botms[~var.MASK[:, j], j] = botm
    print(f"Cells Impervious Elevation Shape: {botms.shape}")
    cache_2(f"CIE_{file}", botms)


@timer
def cal_XSWaterSurfaceElevation(hfp: str, rhfp: str, file):
    data_wse_np = np.array(open_HDF_file(hfp)[wse_xss_path])
    ccc = load_cache_2(f"CCC_{file}")
    river_centerlines_coord_datas = polyline_segment(rhfp, river_centerlines_coord_path, river_centerlines_seg_path)

    xss_attributes = pd.DataFrame(pd.read_hdf(rhfp, xss_path))
    rr_attributes = pd.DataFrame(pd.read_hdf(rhfp, rr_path))
    reach_index, proj_location = get_LocationOnCPolyline(rr_attributes, river_centerlines_coord_datas, ccc)

    C = get_NearestSectionData(xss_attributes, reach_index, proj_location)
    A = C[..., 0].astype(int)
    B = C[..., 1].astype(int)
    wA = C[..., 2]  # (m, n)
    wB = C[..., 3]  # (m, n)
    wse_A = data_wse_np[:, A].transpose(1, 2, 0)  # (m, n, p)
    wse_B = data_wse_np[:, B].transpose(1, 2, 0)  # (m, n, p)
    wse = wA[..., np.newaxis] * wse_A + wB[..., np.newaxis] * wse_B  # (m, n, p)
    print(f"Water Surface Elevation Shape of {hfp}: {wse.shape}")
    cache_1(hfp, f"XSSE_{file}", wse)


def DataAcquisitionSingle(rhfp, input_file_path):
    lcs = get_location_df(input_file_path)  # 计算 1 次
    input_file_name = get_file_name(input_file_path)
    var.LCS = lcs

    lfs, lfps = get_face_fp_np(rhfp, lcs)  # 计算 1 次

    cal_CellMinimumElevation(rhfp, lcs, input_file_name)  # 计算 1 次
    cal_CellCenterCoordinate(rhfp, lcs, input_file_name)  # 计算 1 次
    cal_CellToBankDistance(rhfp, input_file_name)  # 计算 1 次
    cal_CellImperviousElevation(lcs, input_file_name)  # 计算 1 次

    return lcs, lfs, lfps, input_file_name


def DataAcquisitionMultiple(hfp: str, lcs, lfs, lfps, rhfp, file):
    cal_CellHydraulicDepth(hfp, lcs, file)  # 需要拼接
    cal_CellShearStress(hfp, lcs, lfs, file)  # 需要拼接
    cal_CellVelocity(hfp, lcs, lfps, file)  # 需要拼接
    cal_XSWaterSurfaceElevation(hfp, rhfp, file)  # 需要拼接


def DataAcquisition(input_file_path):
    rhfp = next((plan.plan_path for plan in
                 my_plans if plan.plan_path.endswith(ref_hdf_file)), None)
    """
    lcs: locational cells
    lfs: locational faces
    lfps: locational facepoints
    fn: file name
    hfp: hdf file path
    """
    lcs, lfs, lfps, fn = DataAcquisitionSingle(rhfp, input_file_path)
    for hf in cal_hdf_files:
        hfp = next((plan.plan_path for plan in
                    my_plans if plan.plan_path.endswith(hf)), None)
        DataAcquisitionMultiple(hfp, lcs, lfs, lfps, rhfp, fn)


def DataSplicingAndInterpolation(input_file_path, depth=16.0):
    """
    时间序列数据预处理与拼接
    """
    fn = get_file_name(input_file_path)
    start_times = [datetime.strptime(plan.plan_start_time, date_format)
                   for plan in my_plans
                   if any(plan.plan_path.endswith(hf)
                          for hf in cal_hdf_files)]
    end_times = [datetime.strptime(plan.plan_end_time, date_format)
                 for plan in my_plans
                 if any(plan.plan_path.endswith(hf)
                        for hf in cal_hdf_files)]
    var.ST = min(start_times)
    var.ET = max(end_times)

    dt = pd.date_range(start=var.ST, end=var.ET, freq="H")
    # 按月分组
    df = pd.DataFrame({'date': dt, 'index': range(len(dt))})
    tsgm = np.array(
        [g['index'].tolist() for _, g
         in df.groupby(
            pd.Grouper(key='date', freq='M'))],
        dtype=object)
    cache_2(f"TSGM_{fn}", tsgm)  # (t,) 值为分月每日索引列表 不等长
    # len(dt) 时间戳的数量 包含首尾
    var.MAXIDX = len(dt) - 1

    """拼接 "CHD", "CSS", "CV" 水深 剪力 流速 时间序列 无数据用 0 填充"""
    r1 = np.zeros(
        (len(cal_hdf_files), len(acquired_data_1)),
        dtype=np.ndarray)
    for j, dn in enumerate(acquired_data_1):
        result1 = None
        for i, pn in enumerate(cal_hdf_files):
            r1[i, j] = load_cache_1(pn, dn + f"_{fn}")
            cursor = dt.get_loc(start_times[i])
            if result1 is None:
                result1 = np.zeros(
                    var.LCS.shape + (var.MAXIDX + 1,),
                    dtype=np.float32)  # (m, n, p)
            result1[..., cursor: cursor + r1[i, j].shape[-1]] = r1[i, j]  # (m, n, p')
        cache_2(dn + f"_{fn}", result1)  # (m, n, p)

    """拼接 "XSSE" 断面水面高程 时间序列 无数据用最小值填充"""
    r2 = []
    for pn in cal_hdf_files:
        r2.append(
            load_cache_1(
                pn, acquired_data_2 + f"_{fn}"))
    """
    arr (m, n, p')
    [...] [(m, n), (m, n), (m, n)]
    np.stack(...) (m, n, num)
    np.minimum.reduce(...) (m, n)
    """
    min_values = np.minimum.reduce(
        np.stack(
            [np.min(arr, axis=-1) for arr in r2],
            axis=-1),
        axis=-1)  # (m, n)

    result2 = None
    for i in range(len(cal_hdf_files)):
        cursor = dt.get_loc(start_times[i])
        if result2 is None:
            result2 = np.zeros(
                var.LCS.shape + (var.MAXIDX + 1,),
                dtype=np.float32)
            result2 += min_values[..., np.newaxis]
        result2[..., cursor: cursor + r2[i].shape[-1]] = r2[i]
    cache_2(acquired_data_2 + f"_{fn}", result2)  # (m, n, p)

    """对率定的地下水埋深进行后处理 上采样 高于地表部分设为 0 序列拼接 无数据用均值填充"""
    csv_documents = Path(r"..\..\HECRAS_DataProcessing\CSVDocuments")
    pt_list = []
    for i, pn in enumerate(pt_csv_files):
        date_range = pd.date_range(start=start_times[i], end=end_times[i], freq='D')
        pt_list.append(
            pd.Series(
                depth - np.clip(
                    np.genfromtxt(
                        csv_documents / pn,
                        delimiter=','), 0, None),
                index=date_range
            ).resample('H').interpolate(method='cubic').values)
    mean_pt = np.mean([np.mean(pt) for pt in pt_list])

    phreatic_thickness = None
    for i in range(len(pt_list)):
        cursor = dt.get_loc(start_times[i])
        if phreatic_thickness is None:
            phreatic_thickness = np.zeros(
                var.LCS.shape + (var.MAXIDX + 1,),
                dtype=np.float32)  # (m, n, p)
            phreatic_thickness += mean_pt
        # pt_list[i] (p',) 在赋值操作中 广播规则并不适用
        phreatic_thickness[..., cursor: cursor + pt_list[i].shape[0]] = pt_list[i][np.newaxis, np.newaxis, ...]
    cache_2(f"PTTS_{fn}", phreatic_thickness)  # (m, n, p)


def PreProcessing(input_file_path):
    title = f"THE CURRENT CALCULATION FILE: {input_file_path}"
    print(f"{title:{'-'}^100}")
    # 数据获取
    DataAcquisition(input_file_path)
    # 数据拼接
    DataSplicingAndInterpolation(input_file_path)


if __name__ == "__main__":
    ...
