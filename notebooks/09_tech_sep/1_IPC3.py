#! (root)/notebooks/3_calculate/1_AggregateWeight.py python3
# -*- coding: utf-8 -*-
#%%
import sys
import numpy as np
import pandas as pd
from IPython.display import display
from pathlib import Path
from importlib import reload
from ecomplexity import ecomplexity

# スクリプト実行時でもノートブックでも動くように root を決定
HERE = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
ROOT = HERE.parents[2]  # 2つ上の階層

# 先頭に挿入して自作srcを優先解決
sys.path.append(str(ROOT / 'src/'))
from initialize.config_loader import load_filter_config, load_adj_config
from calculation.aggregate import aggregate
from visualize.bump_chart import *
# reload(visualize.bump_chart)

filter_cfg = load_filter_config(str(ROOT / 'config' / 'reg_num_filter.yaml'))
adj_cfg    = load_adj_config(str(ROOT / 'config' / 'adj.yaml'))

#%%
agg_weight_df = pd.read_csv(
    f'{adj_cfg.in_dir}{adj_cfg.in_file_name}.csv',
    encoding='utf-8',
    sep=',',
).query(f'{adj_cfg.year_start} <={adj_cfg.ar}_{adj_cfg.year_style} <= {adj_cfg.year_end}')
filter_df = pd.read_csv(
    f'{filter_cfg.out_dir}{filter_cfg.out_file_name}.csv',
    encoding='utf-8',
    sep=',',
)
schmoch_df = pd.read_csv(
    'C:/Users/rin/Desktop/KCIinJapaneseFirms/data/processed/external/schmoch/35.csv', 
    encoding='utf-8',
    sep=',',
    ).filter(items=['Field_number', 'Field_en', 'schmoch5'])\
    .drop_duplicates(subset=['Field_number', 'Field_en'], keep='first')

# %%
long_agg_df = aggregate(
    agg_weight_df.query('right_person_name in @filter_df.right_person_name.unique()')\
                 .rename(columns={'right_person_name': 'corporation'}),
    adj_cfg.ar,
    adj_cfg.year_style,
    adj_cfg.year_start,
    adj_cfg.year_end,
    adj_cfg.year_range,
    adj_cfg.applicant_weight,
    adj_cfg.class_weight,
    adj_cfg.region_corporation, 
    adj_cfg.classification,
    adj_cfg.top_p_or_num,
    adj_cfg.top_p_or_num_value,
 )
sep_agg_df = pd.concat(
    [aggregate(
        agg_weight_df.query(f'right_person_name in @filter_df.right_person_name.unique() \
                              & ({year} <= {adj_cfg.ar}_{adj_cfg.year_style} <= {year+adj_cfg.year_range-1})')\
              .rename(columns={'right_person_name': 'corporation'}),
    adj_cfg.ar,
    adj_cfg.year_style,
    year,
    year+adj_cfg.year_range-1,
    adj_cfg.year_range,
    adj_cfg.applicant_weight,
    adj_cfg.class_weight,
    adj_cfg.region_corporation, 
    adj_cfg.classification,
    adj_cfg.top_p_or_num,
    adj_cfg.top_p_or_num_value,)
    for year in range(adj_cfg.year_start, adj_cfg.year_end+1, adj_cfg.year_range)],
    axis='index',
    ignore_index=True
    )

#%%
sep_agg_df
# %%
trade_cols = {
    "time": "period",
    "loc": adj_cfg.region_corporation,
    "prod": adj_cfg.classification,
    "val": "weight",
}
c_df = ecomplexity(
    sep_agg_df,
    trade_cols,
    rca_mcp_threshold=1,
).rename(columns={'eci': 'kci', 'pci': 'tci'})
# %%
class_df = c_df.drop_duplicates(subset=['period', 
                             adj_cfg.classification], keep='first')\
                                 .filter(
                                     items=['period', 
                                            adj_cfg.classification,
                                            'tci']
                                 )
# %%
sep_class_df = pd.merge(
                        class_df, 
                        schmoch_df, 
                        left_on=adj_cfg.classification, 
                        right_on='Field_number', 
                        how='left'
                        )\
                        .drop(columns=['Field_number', 'schmoch35'])\
                        .rename(columns={'Field_en': 'schmoch35'})
sep_class_df

# %%
# 1) 期ごとに tci の順位を作成（大きいほど1位）
df_ranked = (
    sep_class_df.assign(
        Ranking = sep_class_df.groupby("period")["tci"].rank(ascending=False, method="dense")
    )
    .assign(Ranking=lambda d: d["Ranking"].astype(int))   # 見栄え用に整数化
)

# 2) サンプルが期待する列名にあわせてリネーム
df_income = (
    df_ranked.rename(columns={
        "period": "Year",
        "schmoch35": "District Name",
        "tci": "Income"   # ← サンプルの hover が "Income" を参照するため合わせる
    })
    .sort_values(["Year", "Ranking"], ignore_index=True)
)
# %%
custom_colors = get_custom_colors(background="light")
fig = go.Figure()

year_order = sorted(df_income["Year"].unique())
add_district_traces(fig, df_income, custom_colors, year_order)
add_ranking_annotations(fig, df_income, year_order)

add_subtitle(fig, "", subtitle_font_size=15, subtitle_color="grey", y_offset=1.05, x_offset=0.0)
add_footer(fig, "", footer_font_size=12, footer_color="grey", y_offset=-0.1, x_offset=0.35)

customize_layout(fig, year_order=year_order)
fig.show()

# %%
