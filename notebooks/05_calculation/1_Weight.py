#! (root)/notebooks/05_calculation/1_CreateRegNumFilter.py python3
# -*- coding: utf-8 -*-

#%%
import sys
import numpy as np
import pandas as pd
from IPython.display import display

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]) / 'src')
from initialize.config_loader import load_weight_config
cfg = load_weight_config(Path(__file__).resolve().parents[2] / 'config' / 'weight.yaml')
print(cfg.out_file_name)


#%%
japan_df = pd.read_csv(
    f'{cfg.in_dir}japan_corporations.csv', 
    encoding='utf-8', 
    sep=','
    )


# %%
unique_df = japan_df.assign(
    ipc3 = lambda x: x['ipc'].str[:3],
    ipc4 = lambda x: x['ipc'].str[:4],
    )\
    .groupby(['reg_num'], as_index=False)\
    .agg(
        # 複数の分類が紐づく場合
        class_dup = (cfg.classification, 'nunique'),
        # 複数の特許権者が紐づく場合
        applicant_dup = (cfg.region_corporation, 'nunique'),
    ).drop_duplicates(keep='first', ignore_index=True)


#%%
unique_df#.query('app_nendo == 2')
#%%
japan_df.query('reg_num == 4003835')

#%%
app_weighted_df = pd.merge(
    japan_df.filter(
                    items=['reg_num', cfg.region_corporation, cfg.classification, f'{cfg.ar}_{cfg.year_style}']
                    ),
    unique_df,
    on='reg_num',
    how='inner',
    )\
    .assign(
        applicant_fraction = lambda x: 1 / (x['applicant_dup']),
        class_fraction = lambda x: 1 / (x['class_dup']),
        both_fraction = lambda x: 1 / (x['applicant_dup'] * x['class_dup']),
    )\
    .drop(columns=['applicant_dup', 'class_dup'])
app_weighted_df.to_csv(
                      f'{cfg.out_dir}{cfg.out_file_name}.csv', 
                      encoding='utf-8', 
                      sep=',',
                      index=False
                      )
app_weighted_df
# %%
