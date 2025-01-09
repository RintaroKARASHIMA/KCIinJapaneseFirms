#! (root)/notebooks/05_calculation/2_Complexity.py python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../src')
from initialize.load_libraries import *

## Import Library
### Processing Data
### Visualization
### Third Party

### Set Visualization Parameters

## Import Original Modules
from initialize import initial_conditions
from calculation import weight

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
data_dir = '../../data/interim/internal/filtered_after_agg/'
outputs_dir = '../../data/processed/internal/'
ex_dir = '../../data/processed/external/'

## Initialize Input and Output Conditions
input_condition = f'{ar}_{year_style}_{year_start}_{year_end}_{year_range}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
output_condition = f'{ar}_{year_style}_{year_start}_{year_end}_{year_range}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
filter_condition = f'{ar}_{year_style}_{extract_population}_reg_num_top_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}'

### Check the condition
print(input_condition)
print(output_condition)
