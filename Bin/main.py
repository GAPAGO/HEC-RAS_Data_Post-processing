# -*- coding: utf-8 -*-
import sys
sys.path.append('CalculationCode')
sys.path.append('CalculationCode/lib')
from PreProcessing import PreProcessing
from PostProcessing import PostProcessing
from CalculationCode.conf.var import input_file_path_list
# from CalculationCode.lib.utils import delete_files_in_folder


if __name__ == "__main__":
    # delete_files_in_folder("Bin\\CachePostProcessing")
    # delete_files_in_folder("Bin\\CachePostProcessing")
    # delete_files_in_folder("Bin\\CalculationCode\\cache")

    for input_file_path in input_file_path_list:
        PreProcessing(input_file_path)
        PostProcessing(input_file_path)
