# Data Processing Program of HEC-RAS
## Based on [.p#.hdf output file](HDF5Documents) & [time series of groundwater level](CSVDocuments)
### in non-floodplain area calibrated by 1D subsurface-surface water coupling model

* _Vectorization operation based on NumPy_
* _The core calculation part uses the compiled C of cython_
* _Includes calculation of one-dimensional groundwater of channel-floodplain system_
* _Includes calculation of characteristic parameters of plant-hydrodynamic process_
* _Not include Picard iteration methods of 1-D along-the-path leakage model_
```
Project directory structure description
│  LICENSE
│  README.md
│  list.txt
│  
├─.idea
│  │  .gitignore
│  │  .name
│  │  csv-editor.xml
│  │  HECRAS_DataProcessing_SangGanHe.iml
│  │  modules.xml
│  │  misc.xml
│  │  HECRASPostProcessing_SangGanHe.iml
│  │  workspace.xml
│  │  
│  └─inspectionProfiles
│          Project_Default.xml
│          
├─Bin
│  │  main.py                               control console
│  │  PreProcessing.py                      for data calculation, concatenation, interpolation
│  │  TimeSeriesSelection.py                for time series result visualization
│  │  PostProcessing.py                     for eigenvalue calculation
│  │  A1.csv                                study object: 2D grid's cell name + grid number
│  │  ...
│  │  Cells.csv
│  │  
│  ├─CachePostProcessing
│  │      CME_A1.npy                        cell minmum elevation
│  │      CCC_A1.npy                        cell center coordinate
│  │      CBD_A1.npy                        distance from cell center to river bank
│  │      CIE_A1.npy                        cell impervious surface elevation
│  │      TSGM_A1.npy                       timestamp array contains the number of each month's timestamp
│  │      CHD_A1.npy                        cell hydraulic depth
│  │      CSS_A1.npy                        cell shear stress
│  │      CV_A1.npy                         cell velocity
│  │      XSSE_A1.npy                       cross section water surface elevation
│  │      PTTS_A1.npy                       phreatic thickness time series
│  │      CUSE_A1.npy                       cell groundwater surface elevation
│  │      CUBD_A1.npy                       cell groundwater burial depth
│  │      
│  ├─CachePreProcessing
│  │      
│  ├─CacheSelectedTimeSeries
│  └─CalculationCode
│     │  Death_Seedling_WaterLevel_Drought.py
│     │  Death_Vegetation_FlowVelocity_Scour.py
│     │  Death_Vegetation_WaterLevel_Drought.py
│     │  Death_Vegetation_WaterLevel_Waterlogging.py
│     │  Distribution_Seed_FlowVelocity_Resuspension.py
│     │  Germination_Seed_ShearStress_Seedbed.py
│     │  Growth_Vegetation_FlowVelocity_Respiration.py
│     │  
│     ├─cache
│     │      
│     ├─conf
│     │     var.py                          global variation
│     │          
│     └─lib
│        │  Decorators.py                   timer module
│        │  utils.py                        global function
│        │  setup.py                        for cython compile
│        │  boussinesq_eq1d.pyx             cython
│        │  boussinesq_eq1d.c               compiled from boussinesq_eq1d.pyx
│        │  boussinesq_eq1d.cp39-win_amd64.pyd
│        │  para_desc.py                    soil parameter setting
│        │      
│        ├─version
│        │      
│        └─build
│          
├─HDF5Documents
│      SGR_1D_2D_Coupling.S3.hdf            .p#.hdf output file
│      SGR_1D_2D_Coupling.S2.hdf            .p#.hdf output file
│      SGR_1D_2D_Coupling.S1.hdf            .p#.hdf output file
│      
├─HDF5Geography
│      SGR_1D_2D_Coupling.g01.hdf
│      
├─__pycache__
├─CSVDocuments
│      BuriedDepth_PhreaticSurface_S1.csv   time series of groundwater level
│      BuriedDepth_PhreaticSurface_S2.csv   time series of groundwater level
│      BuriedDepth_PhreaticSurface_S3.csv   time series of groundwater level
│      
├─Test
│      view_cache.py
│      test_for_vel_and_shear_stress.py     verify the calculation method in rasmapper
│      vel_node.png
│      vel_face.png
│      minus_node-face.png                  difference between the two flow rate calculation methods
│      
└─View                                      hydrodynamic simulation results of a two-dimensional coupled model
    │  SGR_1D_2D_Coupling.S1.hdf            viewed by importing HECRAS
    │  SGR_1D_2D_Coupling.S2.hdf            viewed by importing HECRAS
    │  SGR_1D_2D_Coupling.S3.hdf            viewed by importing HECRAS
    │  
    ├─Terrain                               viewed by importing HECRAS
    │      Projection.prj                   viewed by importing HECRAS
    │      Terrain.hdf                      viewed by importing HECRAS
    │      Terrain.result.tif
    │      Terrain.vrt
    │      
    ├─S1_SWE_FE_Simplify
    │      PostProcessing.hdf
    │      
    └─S3_SWE_FE_Simplify
            PostProcessing.hdf
```