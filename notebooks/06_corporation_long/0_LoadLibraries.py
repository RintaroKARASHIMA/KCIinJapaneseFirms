#! (root)/notebooks/06_producer_long/0_Load_Libraries.py python3
# -*- coding: utf-8 -*-

## Import Library
### System
import IPython
import io
import sys

### Processing Data
import numpy as np
import pandas as pd

### Visualization
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from PIL import Image
from IPython.display import display
import seaborn as sns


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

### Initialize Conditions
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE

extract_population = initial_condition.EXTRACT_POPULATION
top_p_or_num = initial_condition.TOP_P_OR_NUM
region_corporation = initial_condition.REGION_CORPORATION
applicant_weight = initial_condition.APPLICANT_WEIGHT

classification = initial_condition.CLASSIFICATION
class_weight = initial_condition.CLASS_WEIGHT

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
