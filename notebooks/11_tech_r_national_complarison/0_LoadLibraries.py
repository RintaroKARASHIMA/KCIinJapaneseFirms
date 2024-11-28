#! (root)/notebooks/00_template/0_LoadLibraries.py python3
# -*- coding: utf-8 -*-

## Import Library
### Processing Data
import sys
import pandas as pd
import numpy as np

### Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import seaborn as sns
import cv2
from PIL import Image
import io
import networkx as nx
import networkx.algorithms.bipartite as bip


### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 20

## Import Original Modules
sys.path.append("../../src")
import initial_condition
from process import weight
from visualize import rank as vr

## Initialize Global Variables
global DATA_DIR, EX_DIR, OUTPUT_DIR
DATA_DIR = '../../data/processed/internal/tech/'
EX_DIR = '../../data/processed/external/'
OUTPUT_DIR = '../../output/figures/'

## Initialize Input and Output Conditions
### Import Initial Conditions
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

input_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
fig_name_base = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}.png'

### Check the condition
print(input_condition)
print(fig_name_base)
