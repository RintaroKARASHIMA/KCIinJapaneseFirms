#! (root)/src/initialize/initial_conditions.py python3
# -*- coding: utf-8 -*-

# Filter Condition
# COUNTRY = 'JP'  # JP or EU
AR = 'app'  # app or reg
YEAR_STYLE = 'nendo'  # nendo or year

YEAR_START = 1981  # int
YEAR_END = 2010  # int
YEAR_RANGE = 5  # int

REGION_CORPORATION = 'right_person_name'  # right_person_name or right_person_addr
CLASSIFICATION = 'schmoch35'  # ipc3 or ipc4 or schmoch35
if 'ipc' in CLASSIFICATION: DIGIT = int(CLASSIFICATION[-1])

VALUE = 'reg_num'  # reg_num or app_num
CLASS_WEIGHT = 'fraction'  # fraction or duplication
APPLICANT_WEIGHT = 'fraction'  # fraction or duplication

EXTRACT_POPULATION = 'all'  # all or sep_year
TOP_P_OR_NUM = ('p', 3)  # (p or num, int)

PERIOD_ORDER_DICT = {0: f'{YEAR_START}-{YEAR_END}'} | \
                    {i+1: f'{YEAR_START}-{YEAR_START+YEAR_RANGE-1}' for i,
                     YEAR_START in enumerate(range(YEAR_START, YEAR_END+1, YEAR_RANGE))}
PERIOD_COL= f"{AR}_{YEAR_STYLE}_period"

# Filter Conditions
FILTER_CONDITIONS = {
    'ar': AR,
    'year_style': YEAR_STYLE,
    'year_start': YEAR_START,
    'year_end': YEAR_END,
    'year_range': YEAR_RANGE,
    'region_corporation': REGION_CORPORATION,
    'classification': CLASSIFICATION,
    'extract_population': EXTRACT_POPULATION,
    'top_p_or_num': TOP_P_OR_NUM
}

# Aggregation Conditions
AGGREGATION_CONDITIONS = {
    'value': VALUE,
    'class_weight': CLASS_WEIGHT,
    'applicant_weight': APPLICANT_WEIGHT
}

# Calculation Conditions
CALCULATION_CONDITIONS = {
    'period_order_dict': PERIOD_ORDER_DICT,
    'region_corporation': REGION_CORPORATION,
    'classification': CLASSIFICATION,
    'extract_population': EXTRACT_POPULATION,
    'top_p_or_num': TOP_P_OR_NUM
}

OUTPUT_DIR = '../../outputs/'
RAW_IN_DIR = '../../data/raw/internal/'
RAW_EX_DIR = '../../data/raw/external/'
IN_IN_DIR = '../../data/interim/internal/'
IN_EX_DIR = '../../data/interim/external/'
PRO_IN_DIR = '../../data/processed/internal/'
PRO_EX_DIR = '../../data/processed/external/'