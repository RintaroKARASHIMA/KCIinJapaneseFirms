#! (root)/notebooks/05_calculation/1_CreateRegNumFilter.py python3
# -*- coding: utf-8 -*-

#%%
import sys
import numpy as np
import pandas as pd
from IPython.display import display

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]) + '/src/')
from initialize.config_loader import load_weight_config, load_filter_config
from calculation.reg_num_filter import reg_num_filter, aggregate

# weight_cfg = load_weight_config(str(Path(__file__).resolve().parents[2]) + '/config/weight.yaml')
filter_cfg = load_filter_config(str(Path(__file__).resolve().parents[2]) + '/config/reg_num_filter_agg.yaml')
print(filter_cfg.out_file_name)


#%%
japan_df = pd.read_csv(
    f'{filter_cfg.in_dir}japan_corporations.csv', 
    encoding='utf-8', 
    sep=','
    ).assign(
    ipc3 = lambda x: x['ipc'].str[:3],
    ipc4 = lambda x: x['ipc'].str[:4],
    )


# %%
unique_df = japan_df\
    .groupby(['reg_num'], as_index=False)\
    .agg(
        # 複数の分類が紐づく場合
        schmoch35_dup = ('schmoch35', 'nunique'),
        ipc3_dup = ('ipc3', 'nunique'),
        ipc4_dup = ('ipc4', 'nunique'),
        # 複数の特許権者が紐づく場合
        corporation_dup = ('corporation', 'nunique'),
        prefecture_dup = ('prefecture', 'nunique'),
    ).drop_duplicates(keep='first', ignore_index=True)


#%%
unique_df#.query('app_nendo == 2')
#%%
japan_df.query('reg_num == 4003835')
#%%
weighted_df = pd.merge(
    japan_df,
    unique_df,
    on='reg_num',
    how='inner',
    )
#%%
weighted_df

#%%
reg_num_filter_df = reg_num_filter(weighted_df, filter_cfg.ar, filter_cfg.year_style, 
               filter_cfg.year_start, filter_cfg.year_end, filter_cfg.year_range,
               filter_cfg.applicant_weight, filter_cfg.class_weight, 
               filter_cfg.extract_span, filter_cfg.extract_population, 
               filter_cfg.top_p_or_num, filter_cfg.top_p_or_num_value)#\
            # .to_csv(f'{filter_cfg.out_dir}{filter_cfg.out_file_name}.csv', 
            #            encoding='utf-8', 
            #            sep=',', 
            #            index=False)
# %%
#%%
filtered_df = pd.merge(
    weighted_df,
    reg_num_filter_df,
    on=['reg_num', filter_cfg.extract_population],
    how='inner',
    )\
    .filter(items=['reg_num', 
                   filter_cfg.region_corporation, 
                   filter_cfg.classification,
                   f'{filter_cfg.ar}_{filter_cfg.year_style}',
                   f'{filter_cfg.region_corporation}_dup', 
                   f'{filter_cfg.classification}_dup',
                   ])\
    .drop_duplicates(keep='first', ignore_index=True)

#%%
filtered_df

#%%
agg_df = pd.concat(
                  [aggregate(filtered_df, filter_cfg.ar, filter_cfg.year_style, 
                             filter_cfg.year_start, filter_cfg.year_end, 
                             filter_cfg.applicant_weight, filter_cfg.class_weight, 
                             filter_cfg.region_corporation, filter_cfg.classification)]
                  +
                  [aggregate(filtered_df, filter_cfg.ar, filter_cfg.year_style, 
                             year, year+filter_cfg.year_range-1, 
                             filter_cfg.applicant_weight, filter_cfg.class_weight, 
                             filter_cfg.region_corporation, filter_cfg.classification)
                   for year in range(filter_cfg.year_start, filter_cfg.year_end+1, filter_cfg.year_range)],
                  axis='index',
                  ignore_index=True
)

#%%
agg_df
# %%
agg_df.to_csv(
    f'{filter_cfg.out_dir}{filter_cfg.out_file_name}.csv', 
    encoding='utf-8', 
    sep=',', 
    index=False
)
# %%
