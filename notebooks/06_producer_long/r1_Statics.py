#! (root)/notebooks/06_producer_long/r1_Statics.py python3
# -*- coding: utf-8 -*-

#%%
# %load 0_LoadLibraries.py 
## Import Library
### Processing Data
import pandas as pd
import numpy as np
import sys

### Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import seaborn as sns
import cv2
from PIL import Image
import io

### Network Analysis
import networkx as nx
import networkx.algorithms.bipartite as bip

### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 20
pd.options.display.float_format = "{:.3f}".format

## Import Original Modules
sys.path.append("../../src")
import initial_condition
from process import weight
from visualize import rank as vr

# %%
## Set Global Variables
global DATA_DIR, EX_DIR, OUTPUT_DIR
DATA_DIR = "../../data/processed/internal/corporations/"
EX_DIR = "../../data/processed/external/schmoch/"
OUTPUT_DIR = "../../output/figures/"

## Initial Conditons
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE

extract_population = initial_condition.EXTRACT_POPULATION
top_p_or_num = ("p", 100)
region_corporation = "right_person_addr"
applicant_weight = initial_condition.APPLICANT_WEIGHT

classification = initial_condition.CLASSIFICATION
class_weight = initial_condition.CLASS_WEIGHT

color_list = initial_condition.COLOR_LIST

input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
fig_name_base = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}.png"

print(input_condition)

# %%
period_order_dict = {
    f"{period_start}-{period_start+year_range-1}": i
    for i, period_start in enumerate(range(year_start, year_end + 1, year_range))
}
period_order_dict[f"{year_start}-{year_end}"] = len(period_order_dict)
print(period_order_dict)

df = pd.read_csv(
    f"{DATA_DIR}{input_condition}.csv",
    encoding="utf-8",
    engine="python",
    sep=",",
    index_col=0,
)
display(df)

filtered_df = pd.read_csv(
    "../../data/interim/internal/filtered_before_agg/addedclassification.csv", sep=","
)
print(filtered_df)

# %%
##

