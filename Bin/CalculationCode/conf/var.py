# -*- coding: utf-8 -*-
"""
全局变量
"""
from collections import namedtuple

MASK = None
LCS = None
ST = None
ET = None
MAXIDX = None

input_file_path_list = ["A1.csv"]  # , "A2.csv", "A3.csv", "A4.csv", "A5.csv", "A6.csv", "A7.csv", "A8.csv", "A9.csv"
input_file_ref_path = 'Cells.csv'
cal_hdf_files = ['S1.hdf', 'S2.hdf', 'S3.hdf']
pt_csv_files = ["BuriedDepth_PhreaticSurface_S1.csv",
                "BuriedDepth_PhreaticSurface_S2.csv",
                "BuriedDepth_PhreaticSurface_S3.csv"]
ref_hdf_file = "S1.hdf"
acquired_data_1 = ["CHD", "CSS", "CV"]
acquired_data_2 = "XSSE"
Plan = namedtuple('Plan', ['plan_path', 'plan_start_time', 'plan_end_time'])
# 模型输出文件 修改 .p# 后缀为 .S# 无影响
my_plans = [
    Plan('../HDF5Documents/SGR_1D_2D_Coupling.S1.hdf', '2020:06:30:00', '2020:08:10:00'),  # S1
    Plan('../HDF5Documents/SGR_1D_2D_Coupling.S2.hdf', '2020:10:19:00', '2021:01:01:00'),  # S2
    Plan('../HDF5Documents/SGR_1D_2D_Coupling.S3.hdf', '2021:03:29:00', '2021:07:17:00'),  # S3
]
# 日期格式
date_format = "%Y:%m:%d:%H"

xss_path = \
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Cross Section Attributes"
wse_xss_path = \
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Water Surface"
river_bank_lines_coord_path = \
    "Geometry/River Bank Lines/Polyline Points"
river_bank_lines_seg_path = \
    "Geometry/River Bank Lines/Polyline Parts"
rr_path = \
    "Geometry/River Centerlines/Attributes"
river_centerlines_coord_path = \
    "Geometry/River Centerlines/Polyline Points"
river_centerlines_seg_path = \
    "Geometry/River Centerlines/Polyline Parts"
result_2d = \
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/"
geometry_2d = \
    "Geometry/2D Flow Areas/"
