#%%
## Import Library
import pandas as pd
import numpy as np
import sys

sys.path.append('../../src')

## Set Visualization Parameters 
from IPython.display import display
pd.options.display.float_format = '{:.2f}'.format

#%%
## Import Original Modules
import initial_condition

#%%
## Set Global Variables
global data_dir, output_dir
data_dir = '../../data/interim/internal/filtered_before_agg/'
output_dir = '../../data/interim/internal/filter_after_agg/'

## 初期条件
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE

extract_population = initial_condition.EXTRACT_POPULATION
top_p_or_num = initial_condition.TOP_P_OR_NUM

region_corporation = initial_condition.REGION_CORPORATION

condition = f'{ar}_{year_style}_{extract_population}_reg_num_top_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}'
print(condition)

#%%
## Load Raw Data
all_df = pd.read_csv(
    f'{data_dir}japan.csv',
    encoding='utf-8',
    sep=',',
    usecols=['reg_num', region_corporation, f'{ar}_{year_style}'],
    dtype={'reg_num': str, region_corporation: str, f'{ar}_{year_style}': np.int64},
)
all_df = all_df[
    all_df[f'{ar}_{year_style}'].isin(range(year_start, year_end + 1))
].drop_duplicates()
all_df[f'{ar}_{year_style}_period'] = f'{year_start}-{year_end}'

# sep_year_df_dict = {}
sep_year_df_list = []

for year in range(year_start, year_end + 1, year_range):
    # sep_year_df_dict[f'{year}-{year+year_range-1}'] = all_df[all_df[f'{ar}_year'].isin(range(year, year+year_range))]
    sep_year_df = all_df[
        all_df[f'{ar}_{year_style}'].isin(range(year, year + year_range))
    ].drop_duplicates()
    sep_year_df[f'{ar}_{year_style}_period'] = f'{year}-{year+year_range-1}'
    sep_year_df_list.append(sep_year_df)


#%%
## Weight each patent
all_applicant_weight_df = (
    all_df.groupby([f'{ar}_{year_style}_period', 'reg_num'])[[region_corporation]]
    .nunique()
    .reset_index(drop=False)
    .rename(columns={region_corporation: 'applicant_weight'})
    .copy()
)

all_reg_num_df = pd.merge(
    all_df.drop(columns=[f'{ar}_{year_style}']),
    all_applicant_weight_df,

    on=[f'{ar}_{year_style}_period', 'reg_num'],
    how='left',
)

all_reg_num_df['reg_num'] = round(1 / all_reg_num_df['applicant_weight'], 2)

# display(all_reg_num_df.head())

all_reg_num_df = (
    all_reg_num_df.drop(columns=['applicant_weight'])

    .groupby([f'{ar}_{year_style}_period', region_corporation])[['reg_num']]
    .sum()
    .reset_index()
    .sort_values(by=['reg_num'], ascending=[False])
    .reset_index(drop=True)
)
display(all_reg_num_df)

#%%
sep_year_reg_num_df_list = []
for sep_year_df in sep_year_df_list:
    sep_year_applicant_weight_df = (
        sep_year_df.groupby([f'{ar}_{year_style}_period', 'reg_num'])[
            [region_corporation]
        ]
        .nunique()
        .reset_index(drop=False)
        .rename(columns={region_corporation: 'applicant_weight'})
        .copy()
    )
    sep_year_reg_num_df = pd.merge(
        sep_year_df.drop(columns=[f'{ar}_{year_style}']),
        sep_year_applicant_weight_df,
        on=[f'{ar}_{year_style}_period', 'reg_num'],
        how='left',
    )
    sep_year_reg_num_df['reg_num'] = round(
        1 / sep_year_reg_num_df['applicant_weight'], 2
    )
    # display(sep_year_reg_num_df.head())
    sep_year_reg_num_df = (
        sep_year_reg_num_df.drop(columns=['applicant_weight'])
        .groupby([f'{ar}_{year_style}_period', region_corporation])[['reg_num']]
        .sum()
        .reset_index()
    )
    sep_year_reg_num_df = sep_year_reg_num_df.sort_values(
        by=['reg_num'], ascending=[False]
    ).reset_index(drop=True)
    sep_year_reg_num_df_list.append(sep_year_reg_num_df)

# sep_year_reg_num_df = pd.concat(sep_year_reg_num_df_list, axis='index', ignore_index=True)
# sep_year_reg_num_df

#%%
if extract_population == 'all':
    if top_p_or_num[0] == 'p':
        top = (all_reg_num_df[region_corporation].nunique() * top_p_or_num[1]) // 100
    elif top_p_or_num[0] == 'num':
        top = top_p_or_num[1]

    all_reg_num_top_df = all_reg_num_df.head(top)
    all_reg_num_top_df
    all_reg_num_top_df.to_csv(
        f'{output_dir}{condition}.csv',
        encoding='utf-8',
        sep=',',
        index=False,
    )
elif extract_population == 'sep_year':
    sep_year_reg_num_top_df_list = []
    for sep_year_reg_num_df in sep_year_reg_num_df_list:
        if top_p_or_num[0] == 'p':
            top = (
                sep_year_reg_num_df[region_corporation].nunique() * top_p_or_num[1]
            ) // 100
        elif top_p_or_num[0] == 'num':
            top = top_p_or_num[1]
        sep_year_reg_num_top_df_list.append(sep_year_reg_num_df.head(top))
    sep_year_reg_num_top_df = pd.concat(
        sep_year_reg_num_top_df_list, axis='index', ignore_index=True
    )
    sep_year_reg_num_top_df.to_csv(
        f'{output_dir}{condition}.csv',
        encoding='utf-8',
        sep=',',
        index=False,
    )

#%%
print('対象特許権者もしくは都道府県数（日本）：', all_reg_num_top_df[region_corporation].nunique())
print(
    '対象特許数（日本）',
    all_df[all_df[region_corporation].isin(all_reg_num_top_df[region_corporation])][
        'reg_num'
    ].nunique(),
)