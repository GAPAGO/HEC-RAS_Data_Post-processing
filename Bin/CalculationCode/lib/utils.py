# -*- coding: utf-8 -*-
from typing import Tuple
from h5py import File
import os
import json
import h5py
import numpy as np
import pandas as pd

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
cache_1_relative_path = 'Bin\\CachePreProcessing'
cache_2_relative_path = 'Bin\\CachePostProcessing'
cache_1_path = os.path.join(root_dir, cache_1_relative_path)
cache_2_path = os.path.join(root_dir, cache_2_relative_path)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)


def get_file_name(file_path: str) -> str:
    parts = file_path.split('.')
    return parts[-2]


def cache_1(hfp: str, fn: str, dnp):
    folder = cache_1_path
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(os.path.join(folder, f'{get_file_name(hfp)}{fn}'), dnp)


def cache_2(file_name: str, np_data):
    folder = cache_2_path
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(os.path.join(folder, f'{file_name}'), np_data)


def cache_json(file_name: str, data: dict):
    folder = cache_2_path
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, f'{file_name}.json'), 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_cache_1(hfp: str, fn: str):
    return np.load(f'{cache_1_path}\\{get_file_name(hfp)}{fn}.npy', allow_pickle=True)


def load_cache_2(fn: str):
    return np.load(f'{cache_2_path}\\{fn}.npy', allow_pickle=True)


def delete_files_in_folder(folder_path):
    for filename in os.listdir(os.path.join(root_dir, folder_path)):
        file_path = os.path.join(root_dir, folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def moving_average(a, n):
    """
    滑动窗口平均
    :param a: (..., m)
    :param n: int n
    :return: (..., m - n + 1)
    """
    avg = np.cumsum(a, axis=-1, dtype=np.float32)
    avg[..., n:] = avg[..., n:] - avg[..., :-n]
    return avg[..., n - 1:] / n


# def threshold_breaking_idx(avg, th, t):
#     """
#     :param avg: (..., m - n + 1)
#     :param th: float
#     :param t: int 可为 moving_average 中的 n - 1 即可使 m - n + 1 + (n - 1) = m
#     :return: (...) 值为 不定长超过阈值的一维索引 (int) 数组
#     """
#     r = np.argwhere(avg > th)  # (num, idx:len(avg.shape))
#     r[..., -1] += t
#     idx = np.zeros(avg.shape[:-1], dtype=np.ndarray)
#     for i in np.ndindex(avg.shape[:-1]):
#         idx[i] = r[np.all(r[:, :-1] == i, axis=1), -1].astype(np.int32)
#     return idx


def threshold_breaking_array(avg, th, t):
    bool_array = avg > th  # (m, n, p - time + 1)
    false_array = np.full(avg.shape[:-1] + (t,), False)
    new_bool_array = np.concatenate((false_array, bool_array), axis=-1)
    return new_bool_array


# def mask_based_remove(idx, mask):
#     r = np.zeros(idx.shape, dtype=np.ndarray)
#     for i in np.ndindex(idx.shape):
#         r[i] = np.array([x for x in idx[i] if x not in mask[i]], dtype=np.int32)
#     return r


def mask_based_remove_array(array, mask_array):
    array[mask_array] = False
    return array


# def monthly_percentage(idx, tsgm):
#     month_hours = np.array([len(month) for month in tsgm], dtype=np.int32)  # (tsgm.shape[0],) or tsgm.shape
#     month_starts = np.cumsum(np.insert(month_hours, 0, 0))
#
#     def func(a):
#         return np.diff(np.searchsorted(a, month_starts))  # (tsgm.shape[0],)
#
#     shape = idx.shape + tsgm.shape  # (..., tsgm.shape[0])
#     counts = np.zeros(shape, dtype=np.int32)
#     for index, x in np.ndenumerate(idx):
#         counts[index] = func(x)  # (tsgm.shape[0],)  值为每个月的索引数量
#
#     percentage = counts / month_hours.astype(np.float32)
#     return percentage  # (..., tsgm.shape[0])


def monthly_percentage_array(array, tsgm):
    month_hours = np.array([len(month) for month in tsgm], dtype=np.int32)  # (tsgm.shape[0],) or tsgm.shape
    month_starts = np.cumsum(np.insert(month_hours, 0, 0))
    counts = np.zeros(array.shape[:-1] + tsgm.shape, dtype=np.float32)
    for i in range(tsgm.shape[0]):
        start, end = month_starts[i], month_starts[i + 1]
        counts[..., i] = np.sum(array[..., start: end], axis=-1)
    shape = (1,) * (counts.ndim - 1) + (-1,)
    month_hours = month_hours.reshape(shape)
    return counts / month_hours  # (..., tsgm.shape[0])


# def monthly_screen_average(a, tsgm, idx):
#     shape = idx.shape + tsgm.shape
#
#     idx_screen = np.zeros(shape, dtype=list)
#     for index, x in np.ndenumerate(idx):
#         idx_screen[index] = [np.intersect1d(x, sublist) for sublist in tsgm]
#
#     average = np.zeros(shape, dtype=np.float32)
#     for i in np.ndindex(average.shape):
#         if len(idx_screen[i]) == 0:
#             average[i] = 0
#         else:
#             average[i] = np.mean(a[i[:-1]][idx_screen[i]])
#     return average


def monthly_screen_average_array(a, tsgm, array):
    result = np.zeros(a.shape[:-1] + (tsgm.shape[0],))
    month_hours = np.array([len(month) for month in tsgm], dtype=np.int32)
    month_starts = np.cumsum(np.insert(month_hours, 0, 0))
    epsilon = 1e-7
    for i in range(tsgm.shape[0]):
        start, end = month_starts[i], month_starts[i + 1]
        a_slice = a[..., start: end]
        array_slice = array[..., start: end]
        result[..., i] = np.sum(a_slice * array_slice, axis=-1) / (np.sum(array_slice, axis=-1) + epsilon)
    return result


# def monthly_average(a, tsgm):
#     shape = a.shape[:-1] + tsgm.shape
#     average = np.zeros(shape, dtype=np.float32)
#     for i in range(len(tsgm)):
#         average[..., i] = np.mean(a[..., tsgm[i]], axis=-1)
#     return average


def monthly_max(a, tsgm):
    shape = a.shape[:-1] + tsgm.shape
    average = np.zeros(shape, dtype=np.float32)
    for i in range(len(tsgm)):
        average[..., i] = np.max(a[..., tsgm[i]], axis=-1)
    return average


def open_HDF_file(file_path: str) -> File:
    hec_file = h5py.File(file_path, 'r')
    return hec_file


def polyline_segment(plan_path: str, polyline_coord_path: str, polyline_seg_path: str) -> np.ndarray[np.ndarray]:
    points: np.ndarray = open_HDF_file(plan_path)[polyline_coord_path][:]
    reaches: np.ndarray = open_HDF_file(plan_path)[polyline_seg_path][:, 1]
    results = np.zeros(len(reaches), dtype=object)
    start = 0
    for i in range(len(reaches)):
        end = start + reaches[i]
        segment: np.ndarray = points[start: end].astype(np.float32)
        results[i] = segment
        start = end
    return results


def get_PointToPolyline(polyline: np.ndarray, points: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.float32]:
    """
    计算多点到多段线的距离和投影位置
    :param polyline: (n, 2) 多段线坐标
    :param points: (m, p, 2) 多点坐标
    :return: tuple[(m, p), (m, p), np.float32] 距离、投影位置和多段线长度
    """
    distances = np.zeros(points.shape[:-1], dtype=np.float32)  # (m, p, 2)
    lengths = np.zeros(points.shape[:-1], dtype=np.float32)  # (m, p, 2)
    # 获取多段线上的线段端点坐标
    p1 = np.array(polyline[:-1])  # (n - 1, 2)
    p2 = np.array(polyline[1:])  # (n - 1, 2)
    # 计算线段长度
    segment_length = np.linalg.norm(p2 - p1, axis=-1)  # (n - 1,)  即沿着最后一个维度求其二范数
    # 计算多段线总长度
    total_length = np.cumsum(segment_length)  # (n - 1,)
    overall_length = total_length[-1]

    # 计算点到线段的垂足在线段上的投影长度
    # points[..., np.newaxis, :] (m, p, 1, 2)
    # points[..., np.newaxis, :] - p1 (m, p, n - 1, 2)
    # (p2 - p1).T (2, n - 1)
    # '...ij,ji->...i' 对最后两个维度的对应行列相乘 即普通矩阵乘法取对角线元素 (m, p, n - 1)
    # (m, p, n - 1) / (n - 1,) = (m, p, n - 1)
    # todo: alter np.sum -> None, axis=-1 -> None
    # todo: alter '...ij,jk->...ik' -> '...ij,ji->...i'
    projection_length = np.einsum(
        '...ij,ji->...i', points[..., np.newaxis, :] - p1, (p2 - p1).T,
        dtype=np.float32) / segment_length  # (m, p, n - 1)

    # 判断垂足是否在线段外侧 条件情况
    before = projection_length < 0  # (m, p, n - 1)
    after = projection_length > segment_length  # (m, p, n - 1)

    # 计算点到线段两端点的距离
    distance_before = np.linalg.norm(points[..., np.newaxis, :] - p1, axis=-1)  # (m, p, n - 1)
    distance_after = np.linalg.norm(points[..., np.newaxis, :] - p2, axis=-1)  # (m, p, n - 1)
    # 计算垂足坐标
    projection = p1 + projection_length[..., np.newaxis] * (p2 - p1) / segment_length[..., np.newaxis]  # (m, p, n - 1, 2)

    # 计算点到垂足的距离
    distance_on_segment = np.linalg.norm(points[..., np.newaxis, :] - projection, axis=-1)  # (m, p, n - 1)
    # 根据垂足位置计算点到线段的距离 考虑垂足在线段外的情况
    distance = np.where(before, distance_before,
                        np.where(after, distance_after,
                                 distance_on_segment))  # (m, p, n - 1)

    # 计算最近点在线段外侧时在线段末端到多段线末端的长度
    length_before = total_length - segment_length  # (n - 1,)
    length_after = total_length  # (n - 1,)
    # 计算最近点在线段上时在线段末端到多段线末端的长度
    length_on_segment = total_length - (segment_length - projection_length)  # (m, p, n - 1)
    # 根据最近点位置计算最近点在线段末端到多段线末端的长度
    length = overall_length - np.where(before, length_before,
                                       np.where(after, length_after,
                                                length_on_segment))  # (m, p, n - 1)

    min_index = np.argmin(distance, axis=-1)  # (m, p)
    distances[...] = distance[np.arange(distance.shape[0])[:, None], np.arange(distance.shape[1]), min_index]  # (m, p)
    lengths[...] = length[np.arange(length.shape[0])[:, None], np.arange(length.shape[1]), min_index]  # (m, p)

    return distances, lengths, overall_length


def get_DistanceToBEPolyline(polyline_coords: np.ndarray[np.ndarray], points_coord: np.ndarray) -> np.ndarray:
    """
    计算多个点到最近多段线之间距离的函数
    为了实现向量化运算，请将 points_coord 中不存在的值设置为 [0， 0]
    :param polyline_coords: 多段线一维数组嵌套二维数组
    :param points_coord: location_cells 坐标点三维数组
    :return: dist
    """
    # 获取 points_coord 数组的前两个维度的大小
    m, n = points_coord.shape[: 2]
    # 初始化 dist 数组，它用于存储每个点与每条多段线之间的距离
    dists = np.zeros((m, n, len(polyline_coords)), dtype=np.float32)
    dist = np.zeros((m, n), dtype=np.float32)

    # 遍历每条多段线
    for i, seg_polyline_coords in enumerate(polyline_coords):
        # 计算每个点到多段线的距离
        distances, _, _ = get_PointToPolyline(seg_polyline_coords, points_coord)
        # 更新 dist 数组
        dists[..., i] = distances

    min_index = np.argmin(dists, axis=-1)  # (m, n)
    dist[...] = dists[np.arange(dists.shape[0])[:, None], np.arange(dists.shape[1]), min_index]

    # 返回每个点与最近多段线之间的距离
    return dist


def get_LocationOnCPolyline(rr_attributes: pd.DataFrame, polyline_coords: np.ndarray[np.ndarray],
                            points_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算多个点到最近多段线之间投影位置的函数
    :param rr_attributes:
    :param polyline_coords: 多段线一维数组嵌套二维数组
    :param points_coord: location_cells 坐标点三维数组
    :return: loc
    """
    m, n = points_coord.shape[:2]

    dists = np.zeros((m, n, len(polyline_coords)), dtype=np.float32)
    locs = np.zeros((m, n, len(polyline_coords)), dtype=np.float32)
    loc = np.zeros((m, n), dtype=np.float32)
    total_lengths = np.zeros(len(polyline_coords), dtype=np.float32)
    reach_index = np.zeros((m, n, 2), dtype='U16')

    # 遍历每条多段线
    for i, seg_polyline_coords in enumerate(polyline_coords):
        # 计算每个点到多段线的距离和投影位置
        distances, lengths, total_length = get_PointToPolyline(seg_polyline_coords, points_coord)
        # 更新 dist 和 loc 数组
        dists[..., i] = distances
        locs[..., i] = lengths
        total_lengths[i] = total_length

    min_index = np.argmin(dists, axis=-1)  # (m, n)
    loc[...] = locs[np.arange(locs.shape[0])[:, None], np.arange(locs.shape[1]), min_index]

    reach_index[..., 0] = rr_attributes.loc[min_index.flatten(), 'River Name'].values.reshape(m, n)
    reach_index[..., 1] = rr_attributes.loc[min_index.flatten(), 'Reach Name'].values.reshape(m, n)

    # TODO: Designed for SangGan River
    def sanggan_river():
        """
        为桑干河专门写的函数，min_index == 1 代表了 YongDing_River_Reach_1，它的断面编号是从 YongDing_River_Reach_2 开始算起
        :return: 无，操作的是函数外的变量
        """
        reach_indices = min_index == 1  # (m, n) Bool
        if np.any(reach_indices):
            loc[reach_indices] += total_lengths[2]

    sanggan_river()

    # 返回每个点与最近多段线的多段线编号、之间的距离和投影位置
    return reach_index, loc


def get_NearestSectionData(xss_attributes: pd.DataFrame, reach_index: np.ndarray,
                           proj_location: np.ndarray) -> np.ndarray:
    """
    返回一个三维 np 数组，
    :param xss_attributes: 形状为 (266, 4) 桑干河一维断面数 River, Reach, Station, Name
    :param reach_index: 形状为 (m, n, 2)
    :param proj_location: 形状为 (m, n)
    :return: 形状为 (m, n, 4) 表示 m 个漫滩的 n 个位置距离最近的两个断面 A 和 B 的行号以及距离 A 和 B 的百分比
    """
    m, n = proj_location.shape
    result = np.zeros((m, n, 4), dtype=np.float32)

    # 预处理 xss_attributes 数据框
    xss_attributes['Station_float'] = xss_attributes['Station'].str.strip(" *").astype(np.float32)

    for i in range(m):
        for j in range(n):
            river = reach_index[i, j, 0]
            reach = reach_index[i, j, 1]
            location = proj_location[i, j]
            mask1 = (xss_attributes['River'].str.contains(river)) & (xss_attributes['Reach'].str.contains(reach))
            if (mask1 & (xss_attributes['Station_float'] >= location)).any():
                index1 = xss_attributes[mask1 & (xss_attributes['Station_float'] >= location)].index.max()
            else:
                index1 = xss_attributes[mask1 & (xss_attributes['Station_float'] < location)].index.min()
            value1 = xss_attributes.loc[index1, 'Station_float']

            mask2 = (xss_attributes['River'].str.contains(river)) & (xss_attributes['Reach'].str.contains(reach))
            if (mask2 & (xss_attributes['Station_float'] <= location)).any():
                index2 = xss_attributes[mask2 & (xss_attributes['Station_float'] <= location)].index.min()
            else:
                index2 = xss_attributes[mask2 & (xss_attributes['Station_float'] > location)].index.max()
            value2 = xss_attributes.loc[index2, 'Station_float']

            if index1 == index2:
                dist_pct_1 = dist_pct_2 = 0.5
            else:
                dist_pct_1 = (value1 - location) / (value1 - value2)
                dist_pct_2 = (location - value2) / (value1 - value2)

            result[i, j] = [index1, index2, dist_pct_1, dist_pct_2]

    return result


def cell_to_face(hdf_file: str, reach_path):
    """
    获取字典，包含 A# 漫滩所有 cell: face 键值对
    :param hdf_file: 计划文件地址
    :param reach_path: 漫滩编号
    :return: 返回字典
    """
    with open_HDF_file(hdf_file) as f:
        face_cell: np.ndarray = f["/Geometry/2D Flow Areas/" + reach_path + "/Faces Cell Indexes"][:]
    print(f"FP {reach_path}'s Faces Cell Indexes Shape:{face_cell.shape}")
    cell_face_map = {}
    for idx, cells in enumerate(face_cell):
        for cell in cells:
            if cell not in cell_face_map:
                # 字典元素的新增，值类型为列表
                cell_face_map[cell] = [idx]
            else:
                # 先获取对应值，再使用列表元素的增加
                cell_face_map[cell].append(idx)
    print(f"FP {reach_path}'s Cell Num:{len(cell_face_map)}\n")
    return cell_face_map


def filter_facepoints(hdf_file_path: str, reach_path):
    """
    过滤 numpy.ndarray 数组中 -1 值，并获取 cell: facepoints 字典
    包含 A# 漫滩所有 cell: facepoint 键值对
    :param hdf_file_path: 计划文件地址
    :param reach_path: 漫滩编号
    :return: 返回字典
    """
    with open_HDF_file(hdf_file_path) as f:
        cell_fp = f['/Geometry/2D Flow Areas/' + reach_path + '/Cells FacePoint Indexes'][:]
    print(f"FP {reach_path}'s Cells FacePoint Indexes Shape:{cell_fp.shape}")
    cell_fp_map = {}
    for cell, fps in enumerate(cell_fp):
        valid_fps = []
        for fp in fps:
            fp_int = int(fp)
            if fp_int != -1:
                valid_fps.append(fp_int)
        cell_fp_map[cell] = valid_fps
    print(f"FP {reach_path}'s Cell Num:{len(cell_fp_map)}\n")
    return cell_fp_map
