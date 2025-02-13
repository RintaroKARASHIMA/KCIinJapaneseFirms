#! (root)/src/initialize/initial_conditions.py python3
# -*- coding: utf-8 -*-


## 初期条件
AR = 'app'  # app or reg
YEAR_STYLE = 'nendo'  # nendo or year

YEAR_START = 1981  # int
YEAR_END = 2010  # int
YEAR_RANGE = 5  # int

COUNTRY = 'JP'  # JP or EU

REGION_CORPORATION = 'right_person_name' # right_person_name or right_person_addr
CLASSIFICATION = 'ipc3'  # ipc3 or ipc4 or schmoch
VALUE = 'reg_num'
CLASS_WEIGHT = 'fraction'  # fraction or duplication
APPLICANT_WEIGHT = 'fraction'  # fraction or duplication

# reg_num filter
EXTRACT_POPULATION = 'all'  # all or sep_year
TOP_P_OR_NUM = ('p', 3)  # (p or num, int)

PERIOD_ORDER_DICT = {f'{YEAR_START}-{YEAR_START+YEAR_RANGE-1}': i for i, YEAR_START in enumerate(range(YEAR_START, YEAR_END+1, YEAR_RANGE))} 
PERIOD_ORDER_DICT[f'{YEAR_START}-{YEAR_END}'] = len(PERIOD_ORDER_DICT)

# INITIAL_CONDITION = 


