#! (root)/notebooks/00_template/1_sample.py python3
# -*- coding: utf-8 -*-

# %%
# %load 0_LoadLibraries.py
## Import Library
### Processing Data
import sys
import pandas as pd
import numpy as np

### Visualization
from IPython.display import display

### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
pd.options.display.float_format = '{:.3f}'.format

## Import Original Modules
sys.path.append('../../src')
import initial_condition
from process import weight
from visualize import rank as vr

### Import Initial Conditions
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE
year_range = 5

extract_population = initial_condition.EXTRACT_POPULATION
top_p_or_num = initial_condition.TOP_P_OR_NUM
region_corporation = initial_condition.REGION_CORPORATION
applicant_weight = initial_condition.APPLICANT_WEIGHT

classification = initial_condition.CLASSIFICATION
class_weight = initial_condition.CLASS_WEIGHT

## Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = '../../data/interim/internal/filtered_after_agg/'
OUTPUT_DIR = '../../data/processed/internal/'
EX_DIR = '../../data/processed/external/schmoch/'

## Initialize Input and Output Conditions
input_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
output_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'

### Check the condition
print(input_condition)
print(output_condition)


# %%
# Load Data
df = pd.read_csv(
    DATA_DIR + f'{input_condition}.csv', 
    encoding='utf-8', 
    engine='python', 
    sep=','
)
display(df)

#%%
# 各期間
classification_df = pd.merge(c_df.groupby([f'{ar}_{year_style}_period', classification])[['reg_num']].sum().reset_index(drop=False), 
                        c_df.groupby([f'{ar}_{year_style}_period', classification])[[region_corporation]].nunique().reset_index(drop=False), 
                        on=[f'{ar}_{year_style}_period', classification], 
                        how='inner')
classification_df = pd.merge(classification_df, 
                      c_df[[f'{ar}_{year_style}_period', classification, 'ubiquity', 'tci']\
                          +[f'ki_{i}' for i in range(1, 20+1)]]\
                          .drop_duplicates(keep='first'), 
                      on=[f'{ar}_{year_style}_period', classification], 
                      how='inner')

# classification_df['reg_num'] = classification_df['reg_num'].astype(np.int64)
classification_df = pd.merge(classification_df, 
                            schmoch_df.rename(columns={'Field_number':classification}), 
                            on=[classification], 
                            how='inner')\
                            .drop(columns=[classification])\
                            .rename(columns={'Field_en':classification})
# display(classification_df)
# schmoch_df['ipc3'] = schmoch_df['IPC_code'].str[:3]
# schmoch_df = schmoch_df.drop_duplicates()
# schmoch_df
# classification_df = pd.merge(classification_df,
#                                 schmoch_df,
#                                 # on=['ipc3'],
#                                 on=['schmoch35'],
#                                 how='left')\
#                             .rename(columns={'Field_en':'schmoch35'})\
#                             .drop(columns=['IPC_code', 'Field_number'])\
#                             .drop_duplicates()
# for period in classification_df[f'{ar}_{year_style}_period'].unique():
#     classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'] = (classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'] - classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'].min()) / (classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'].max() - classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'].min())


fiveyears_df_dict = {
    f'{year}': classification_df[classification_df[f'{ar}_{year_style}_period']==f'{year}'][[f'{ar}_{year_style}_period', classification, 'tci']].drop_duplicates(keep='first')\
        for year in classification_df[f'{ar}_{year_style}_period'].unique() if year != f'{year_start}-{year_end}'
}

rank.rank_doubleaxis(fiveyears_df_dict, 
                     rank_num=35, 
                     member_col=classification,
                     value_col='tci',
                     prop_dict={
                                "figsize": (16, 10),
                                "xlabel": "Period",
                                "ylabel": "Technological Fields",
                                "title": "",
                                "fontsize": 15,
                                "year_range": 15,
                                "ascending": False,
                                "color": "default",
    })
