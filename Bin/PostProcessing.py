from CalculationCode.lib.utils import get_file_name
from CalculationCode.Death_Vegetation_WaterLevel_Drought import DeathVegetationWaterLevelDrought
from CalculationCode.Death_Seedling_WaterLevel_Drought import DeathSeedlingWaterLevelDrought
from CalculationCode.Death_Vegetation_WaterLevel_Waterlogging import DeathVegetationWaterLevelWaterlogging
from CalculationCode.Death_Vegetation_FlowVelocity_Scour import DeathVegetationFlowVelocityScour
from CalculationCode.Growth_Vegetation_FlowVelocity_Respiration import GrowthVegetationFlowVelocityRespiration
from CalculationCode.Germination_Seed_ShearStress_Seedbed import GerminationSeedShearStressSeedbed
from CalculationCode.Distribution_Seed_FlowVelocity_Resuspension import DistributionSeedFlowVelocityResuspension


def PostProcessing(input_file):
    fn = get_file_name(input_file)

    DVWLD = DeathVegetationWaterLevelDrought(fn)
    DVWLD.cal_CellUndergroundWaterSurfaceElevation()
    DVWLD.cal_CellUndergroundWaterBurialDepth()
    DVWLD.calculation()

    DSWLD = DeathSeedlingWaterLevelDrought(fn)
    DSWLD.calculate()

    DVWLW = DeathVegetationWaterLevelWaterlogging(fn)
    DVWLW.calculate()

    DVFVS = DeathVegetationFlowVelocityScour(fn)
    DVFVS.calculate()

    GVFVR = GrowthVegetationFlowVelocityRespiration(fn)
    GVFVR.calculate()

    GSSSS = GerminationSeedShearStressSeedbed(fn)
    GSSSS.calculate()

    DSFVR = DistributionSeedFlowVelocityResuspension(fn)
    DSFVR.calculate()
