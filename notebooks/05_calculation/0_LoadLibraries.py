## Import Library
### Processing Data
import sys
import pandas as pd
import numpy as np

### Visualization
from IPython.display import display

### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format

## Import Original Modules
sys.path.append("../../src")
from initialize import initial_conditions
from calculation import weight
from visualize import rank as vr

### Initialize Conditions
ar = initial_conditions.AR
year_style = initial_conditions.YEAR_STYLE

year_start = initial_conditions.YEAR_START
year_end = initial_conditions.YEAR_END
year_range = initial_conditions.YEAR_RANGE

extract_population = initial_conditions.EXTRACT_POPULATION
top_p_or_num = initial_conditions.TOP_P_OR_NUM
region_corporation = initial_conditions.REGION_CORPORATION
applicant_weight = initial_conditions.APPLICANT_WEIGHT

classification = initial_conditions.CLASSIFICATION
class_weight = initial_conditions.CLASS_WEIGHT

## Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = "../../data/interim/internal/filtered_after_agg/"
OUTPUT_DIR = "../../data/processed/internal/"
EX_DIR = "../../data/processed/external/schmoch/"

## Initialize Input and Output Conditions
input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
output_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"

### Check the condition
print(input_condition)
print(output_condition)
