#! (root)/src/initialize/initial_conditions.py python3
# -*- coding: utf-8 -*-


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

PERIOD_ORDER_DICT = {f'{YEAR_START}-{YEAR_END}' : 0} | \
                    {f'{YEAR_START}-{YEAR_START+YEAR_RANGE-1}': i+1 for i,
                     YEAR_START in enumerate(range(YEAR_START, YEAR_END+1, YEAR_RANGE))}

CONDITION = f'{AR}_{YEAR_STYLE}_{YEAR_START}_{YEAR_END}_{YEAR_RANGE}_{EXTRACT_POPULATION}_{TOP_P_OR_NUM[0]}_{TOP_P_OR_NUM[1]}_{REGION_CORPORATION}_{APPLICANT_WEIGHT}_{CLASSIFICATION}_{CLASS_WEIGHT}'

OUTPUT_DIR = '../../outputs/'
RAW_IN_DIR = '../../data/raw/internal/'
RAW_EX_DIR = '../../data/raw/external/'
IN_IN_DIR = '../../data/interim/internal/'
IN_EX_DIR = '../../data/interim/external/'
PRO_IN_DIR = '../../data/processed/internal/'
PRO_EX_DIR = '../../data/processed/external/'