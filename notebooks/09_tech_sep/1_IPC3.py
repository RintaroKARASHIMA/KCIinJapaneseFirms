#! (root)/notebooks/00_template/1_sample.py python3
# -*- coding: utf-8 -*-

# %%
# Import Library
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py


# Processing Data
from visualize import rank as vr

# Visualization

# Third Party
from ecomplexity import ecomplexity

# Set Visualization Parameters
pd.options.display.float_format = '{:.3f}'.format

# Import Original Modules
sys.path.append('../../src')
# from process import weight

# Import Initial Conditions
# %run ../../src/initial_conditions.py

# Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = '../../data/processed/internal/05_2_4_tech/'
OUTPUT_DIR = '../../data/processed/internal/'
EX_DIR = '../../data/processed/external/'

# Check the condition
print(input_condition)
print(output_condition)


# %%
df = pd.read_csv(
                 '../../data/processed/internal/05_2_4_tech/app_nendo_1981_2010_5_all_p_3_right_person_name_fraction_schmoch35_fraction.csv',
                 encoding='utf-8',
                 sep=',',
                 )
df#['schmoch5'].unique()

# %%
# fiveyears_df_dict = {
#     f'{year}': classification_df[classification_df[f'{ar}_{year_style}_period'] == f'{year}'][[f'{ar}_{year_style}_period', classification, 'tci']].drop_duplicates(keep='first')
#     for year in classification_df[f'{ar}_{year_style}_period'].unique() if year != f'{year_start}-{year_end}'
# }
fiveyears_df_dict = {
    f'{year}': df.query(f'{ar}_{year_style}_period == "{year}"')\
                 .filter(items=[f'{ar}_{year_style}_period', 'ipc3', 'tci', 'schmoch5', 'Field_en'], axis='columns')\
                 .drop_duplicates(keep='first')
    for year in df[f'{ar}_{year_style}_period'].unique() if year != f'{year_start}-{year_end}'
}

vr.rank_doubleaxis(fiveyears_df_dict,
                     rank_num=124,
                     member_col='ipc3',
                     value_col='tci',
                     prop_dict={
                         "figsize": (32, 52),
                         "xlabel": "Period",
                         "ylabel": "Technological Fields",
                         "title": "",
                         "fontsize": 48,
                         "year_range": 5,
                         "ascending": False,
                         "color": "default",
                     })

# %%
df.query('Field_en == "Food chemistry"')
# %%
