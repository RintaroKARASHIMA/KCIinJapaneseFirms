#! (root)/notebooks/05_calculation/1_CreateRegNumFilter.py python3
# -*- coding: utf-8 -*-

#%%
import sys
import numpy as np
import pandas as pd
from IPython.display import display

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]) / 'src')
from initialize.config_loader import load_filter_config
from calculation.reg_num_filter import reg_num_filter

cfg = load_filter_config(Path(__file__).resolve().parents[2] / 'config' / 'reg_num_filter.yaml')


weighted_df = pd.read_csv(
    f'{cfg.in_dir}{cfg.in_file_name}.csv',
    encoding='utf-8',
    sep=',',
)

display(weighted_df)

#%%
print(weighted_df.columns)


reg_num_filter(weighted_df, cfg.ar, cfg.year_style, cfg.year_start, cfg.year_end, cfg.year_range, cfg.applicant_weight, cfg.class_weight, cfg.extract_population, cfg.top_p_or_num, cfg.top_p_or_num_value)\
            .query(f'period == "{cfg.year_start}-{cfg.year_end}"'
                    if cfg.extract_span == 'all'
                    else f'period != "{cfg.year_start}-{cfg.year_end}"', 
                      engine='python')\
            .to_csv(f'{cfg.out_dir}{cfg.out_file_name}.csv', 
                       encoding='utf-8', 
                       sep=',', 
                       index=False)
# %%
