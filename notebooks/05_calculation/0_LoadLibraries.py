#! (root)/notebooks/05_calculation/0_LoadLibraries.py python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../src')

## Import Library
### Systems
from importlib import reload
### Processing Data
### Visualization
### Third Party
### Set Visualization Parameters


## Import Original Modules
from initialize import initial_conditions
reload(initial_conditions)
from calculation import weight

### Initialize Conditions
ar = initial_conditions.AR
year_style = initial_conditions.YEAR_STYLE

year_start = initial_conditions.YEAR_START
year_end = initial_conditions.YEAR_END
year_range = initial_conditions.YEAR_RANGE

region_corporation = initial_conditions.REGION_CORPORATION
classification = initial_conditions.CLASSIFICATION
if 'ipc' in classification: digit = initial_conditions.DIGIT
value = initial_conditions.VALUE
class_weight = initial_conditions.CLASS_WEIGHT
applicant_weight = initial_conditions.APPLICANT_WEIGHT

extract_population = initial_conditions.EXTRACT_POPULATION
top_p_or_num = initial_conditions.TOP_P_OR_NUM


## Initialize Input and Output Conditions
condition = initial_conditions.CONDITION

### Check the condition
print(condition)