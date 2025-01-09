## Import Library
### Processing Data
import sys
import pandas as pd
import numpy as np

import networkx as nx
import networkx.algorithms.bipartite as bip


### Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import io

### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 20

## Import Original Modules
sys.path.append("../../src")
from initialize import initial_conditions
from calculation import weight
from visualize import rank as vr

### Import Initial Conditions
ar = initial_conditions.AR
year_style = initial_conditions.YEAR_STYLE

year_start = initial_conditions.YEAR_START
year_end = initial_conditions.YEAR_END
year_range = initial_conditions.YEAR_RANGE
year_range = 5

extract_population = initial_conditions.EXTRACT_POPULATION
top_p_or_num = initial_conditions.TOP_P_OR_NUM
region_corporation = initial_conditions.REGION_CORPORATION
applicant_weight = initial_conditions.APPLICANT_WEIGHT

classification = initial_conditions.CLASSIFICATION
class_weight = initial_conditions.CLASS_WEIGHT

## Initialize Global Variables
global data_dir, outputs_dir
# DATA_DIR = "../../data/interim/internal/filtered_after_agg/"
data_dir = "../../data/processed/internal"
outputs_dir = "../../outputs/figures"

## Initialize Input and Output Conditions
input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
output_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"

### Check the condition
print(input_condition)
print(output_condition)
