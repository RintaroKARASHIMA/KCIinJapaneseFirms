#! (root)/notebooks/3_calculate/1_AggregateWeight.py python3
# -*- coding: utf-8 -*-
#%%
import sys
import numpy as np
import pandas as pd
from IPython.display import display
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]) / 'src')
from initialize.config_loader import  load_filter_config, load_agg_config, load_adj_config
from calculation import aggregate, biadjm


filter_cfg = load_filter_config(Path(__file__).resolve().parents[2] / 'config' / 'reg_num_filter.yaml')
adj_cfg = load_adj_config(Path(__file__).resolve().parent[2] / 'config' / 'adj.yaml')

#%%
agg_weight_df = pd.read_csv(
    f'{adj_cfg.in_dir}{adj_cfg.in_file_name}.csv',
    encoding='utf-8',
    sep=',',
)
filter_df = pd.read_csv(
    f'{filter_cfg.out_dir}{filter_cfg.out_file_name}.csv',
    encoding='utf-8',
    sep=',',
)

#%%
# filter
filtered_df = pd.merge(agg_weight_df, filter_df, on=filter_cfg.extract_population, how='inner')
agg_df = aggregate(filtered_df, adj_cfg.ar, adj_cfg.year_style, adj_cfg.year_start, adj_cfg.year_end, adj_cfg.year_range, adj_cfg.applicant_weight, adj_cfg.class_weight, adj_cfg.region_corporation, adj_cfg.classification)
adj_df = biadjm(
    agg_df,
    producer_col = adj_cfg.region_corporation,
    class_col = adj_cfg.classification,
    count_col = 'weight',
)
adj_df.to_csv(
    f'{adj_cfg.out_dir}{adj_cfg.out_file_name}.csv',
    encoding='utf-8',
    sep=',',
    index=False,
)

#%%
# ここから技術分野ごとの特許数分布可視化

#%%
# ここから隣接行列の可視化
