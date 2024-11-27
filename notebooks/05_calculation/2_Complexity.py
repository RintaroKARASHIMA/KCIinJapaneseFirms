# Calculate Complexity

# %%
# Import Libraries
import pandas as pd

import numpy as np
import openpyxl
import sys

sys.path.append("../../src")
from ecomplexity import ecomplexity
from ecomplexity import proximity

# 小数点以下 桁数 6
pd.options.display.float_format = "{:.3f}".format
from IPython.display import display

# %%
# Import Original Modules
import initial_condition
from visualize import rank


# %%
# Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = "../../data/interim/internal/filtered_after_agg/"
OUTPUT_DIR = "../../data/processed/internal/"
EX_DIR = "../../data/processed/external/schmoch/"

# Initial Conditons
## How to span years
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

## Range of years
year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE

## Target
extract_population = initial_condition.EXTRACT_POPULATION
# top_p_or_num = initial_condition.TOP_P_OR_NUM
top_p_or_num = ("p", 3)
region_corporation = initial_condition.REGION_CORPORATION
region_corporation = "right_person_name"
applicant_weight = initial_condition.APPLICANT_WEIGHT

## Classification
classification = initial_condition.CLASSIFICATION
# classification = 'ipc3'
class_weight = initial_condition.CLASS_WEIGHT

## Name of Input and Output Files
input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
output_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"

## Check the condition
print(input_condition)
print(output_condition)

# %%
def kh_ki(c_df, classification, n=19):
    kh1_ki1_df = pd.merge(
        c_df.copy(),
        c_df[c_df["mcp"] == 1]
        .groupby([region_corporation])[["ubiquity"]]
        .sum()
        .reset_index(drop=False)
        .copy()
        .rename(columns={"ubiquity": "kh_1"}),
        on=[region_corporation],
        how="left",
    )
    kh1_ki1_df = pd.merge(
        kh1_ki1_df.copy(),
        c_df[c_df["mcp"] == 1]
        .groupby([classification])[["diversity"]]
        .sum()
        .reset_index(drop=False)
        .copy()
        .rename(columns={"diversity": "ki_1"}),
        on=[classification],
        how="left",
    )
    kh1_ki1_df["kh_1"] = kh1_ki1_df["kh_1"] / kh1_ki1_df["diversity"]
    kh1_ki1_df["ki_1"] = kh1_ki1_df["ki_1"] / kh1_ki1_df["ubiquity"]
    kh_ki_df = kh1_ki1_df.copy()
    for i in range(n):
        kh_ki_df = pd.merge(
            kh_ki_df,
            kh_ki_df[kh_ki_df["mcp"] == 1]
            .groupby([region_corporation])[[f"ki_{i+1}"]]
            .sum()
            .reset_index(drop=False)
            .copy()
            .rename(columns={f"ki_{i+1}": f"kh_{i+2}"}),
            on=[region_corporation],
            how="left",
            copy=False
        )
        kh_ki_df = pd.merge(
            kh_ki_df,
            kh_ki_df[kh_ki_df["mcp"] == 1]
            .groupby([classification])[[f"kh_{i+1}"]]
            .sum()
            .reset_index(drop=False)
            .copy()
            .rename(columns={f"kh_{i+1}": f"ki_{i+2}"}),
            on=[classification],
            how="left", 
            copy=False
        )
        kh_ki_df[f"kh_{i+2}"] = kh_ki_df[f"kh_{i+2}"] / kh_ki_df["diversity"]
        kh_ki_df[f"ki_{i+2}"] = kh_ki_df[f"ki_{i+2}"] / kh_ki_df["ubiquity"]
    return kh_ki_df


# %%
schmoch_df = pd.read_csv(
    f"{EX_DIR}35.csv",
    encoding="utf-8",
    sep=",",
    #  usecols=['Field_number', 'Field_en']
).drop_duplicates()

reg_num_top_df = pd.read_csv(
    f"{DATA_DIR}{input_condition}.csv", encoding="utf-8", sep=","
)
print(reg_num_top_df[region_corporation].nunique())

#%%
pd.read_csv('../../data/interim/internal/filtered_before_agg/japan.csv')\
    .query('1981 <= app_nendo <= 2010')\
    .query(f'right_person_addr in {list(reg_num_top_df[region_corporation].unique())}')\
    ['reg_num'].nunique()

#%%
trade_cols = {'time':f'{ar}_{year_style}_period', 'loc':region_corporation, 'prod':classification, 'val':'reg_num'}
rename_col_dict = {'eci':'kci', 'pci':'tci'}
col_order_list = [f'{ar}_{year_style}_period', region_corporation, classification, 'reg_num', 'rca', 'mcp', 'diversity', 'ubiquity', 'kci', 'tci']

c_df = ecomplexity(reg_num_top_df,
                   cols_input = trade_cols, 
                   rca_mcp_threshold = 1)
# prox_df = proximity(c_df, trade_cols)
# c_out_df = c_df.copy()
print(c_df.columns)

c_df = c_df[c_df['reg_num'] > 0]\
           .rename(columns=rename_col_dict)\
           [col_order_list]
c_df = pd.concat([kh_ki(c_df[c_df[f'{ar}_{year_style}_period'] == period], classification) for period in c_df[f'{ar}_{year_style}_period'].unique()], 
                 axis='index', 
                 ignore_index=True)

# for segment in c_df[f'{ar}_{year_style}_period'].unique():
#     display(c_df[c_df[f'{ar}_{year_style}_period'] == segment].head())
#     display(c_df[c_df[f'{ar}_{year_style}_period'] == segment].describe())
#     print(c_df[c_df[f'{ar}_{year_style}_period'] == segment].info())
#     print('\n')

#%%
