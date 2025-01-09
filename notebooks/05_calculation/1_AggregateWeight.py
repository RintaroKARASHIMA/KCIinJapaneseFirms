#! (root)/notebooks/3_calculate/1_AggregateWeight.py python3
# -*- coding: utf-8 -*-
#%%
%run 0_LoadLibraries.py
## Import Library
### Processing Data

### Visualization

### Third Party

### Set Visualization Parameters

## Import Original Modules

## Arrange variables
ipc_digit = 3
data_dir = '../../data/interim/internal/filtered_before_agg/'
filter_dir = '../../data/interim/internal/filter_after_agg/'
output_dir = '../../data/interim/internal/filtered_after_agg/'

#%%
# 全体
all_df = pd.read_csv(f'{data_dir}japan.csv', 
                     encoding='utf-8', 
                     sep=',', 
                     usecols=['reg_num', 
                              region_corporation, 
                              f'{ar}_{year_style}', 
                              'ipc'], 
                            #   f'{classification}'], 
                     dtype={'reg_num': str, 
                            region_corporation: str, 
                            f'{ar}_{year_style}': np.int64, 
                            'ipc': str})
                            # f'{classification}': str})

all_df[f'ipc{ipc_digit}'] = all_df['ipc'].str[:ipc_digit]
all_df = all_df[all_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))]\
    .drop_duplicates()\
    .drop(columns=['ipc'])\
    .drop_duplicates()
classification = f'ipc{ipc_digit}'
display(all_df.head())
print(all_df['right_person_name'].nunique())

# 各期間
sep_year_df_dict = {}

for year in range(year_start, year_end+1, year_range):
    sep_year_df_dict[f'{year}-{year+year_range-1}'] = all_df[all_df[f'{ar}_{year_style}'].isin(range(year, year+year_range))]


#%%
# 特許分類による重みづけ
# 全体
if class_weight == 'fraction':
    all_df = weight.by_classification(all_df, region_corporation, classification)
elif class_weight == 'duplication':
    all_df['class_weight'] = 1
all_df[f'{ar}_{year_style}_period'] = f'{year_start}-{year_end}'


# 期間ごと
# sep_year_df_dict = {}
# sep_year_reg_num_top_df_dict = {}
for period, sep_year_df in sep_year_df_dict.items():
    if class_weight == 'fraction':
        sep_year_df_dict[period] = weight.by_classification(sep_year_df, region_corporation, classification)
    elif class_weight == 'duplication':
        sep_year_df_dict[period] = sep_year_df.groupby([region_corporation, classification])[['reg_num']].nunique().reset_index(drop=False)
    sep_year_df_dict[period][f'{ar}_{year_style}_period'] = period

# 共同出願の重みづけ
# 全体
if applicant_weight == 'fraction':
    all_df = weight.by_applicant(all_df, region_corporation)
elif applicant_weight == 'duplication':
    all_df['applicant_weight'] = 1
all_df[f'{ar}_{year_style}_period'] = f'{year_start}-{year_end}'


# 期間ごと
# sep_year_df_dict = {}
# sep_year_reg_num_top_df_dict = {}
for period, sep_year_df in sep_year_df_dict.items():
    if applicant_weight == 'fraction':
        sep_year_df_dict[period] = weight.by_applicant(sep_year_df, region_corporation)
    elif applicant_weight == 'duplication':
        sep_year_df_dict[period]['applicant_weight'] = 1
    sep_year_df_dict[period][f'{ar}_{year_style}_period'] = period

#%%
all_reg_num_df = all_df.copy()
all_reg_num_df['reg_num'] = 1 / all_reg_num_df['class_weight'] / all_reg_num_df['applicant_weight']
all_reg_num_df = all_reg_num_df.groupby([f'{ar}_{year_style}_period', region_corporation, classification])[['reg_num']]\
                               .sum().reset_index(drop=False)\
                               .sort_values(['reg_num'], ascending=[False])
sep_year_reg_num_df_dict = sep_year_df_dict.copy()
for period, sep_year_reg_num_df in sep_year_reg_num_df_dict.items():
    sep_year_reg_num_df['reg_num'] = 1 / sep_year_reg_num_df['class_weight'] / sep_year_reg_num_df['applicant_weight']
    sep_year_reg_num_df = sep_year_reg_num_df.groupby([f'{ar}_{year_style}_period', region_corporation, classification])[['reg_num']]\
                                             .sum().reset_index(drop=False)\
                                             .sort_values(['reg_num'], ascending=[False])
    sep_year_reg_num_df_dict[period] = sep_year_reg_num_df
sep_year_reg_num_df = pd.concat([sep_year_reg_num_df for sep_year_reg_num_df in sep_year_reg_num_df_dict.values()], axis='index', ignore_index=True)
sep_year_reg_num_df

#%%
# フィルタリング
reg_num_filter_df = pd.read_csv(f'{filter_dir}{filter_condition}.csv',
                                encoding='utf-8',
                                sep=',', 
                                usecols=[f'{ar}_{year_style}_period', region_corporation],
                                dtype=str)
reg_num_filter_df

#%%

if extract_population == 'all':
    all_reg_num_top_df = pd.merge(
        all_reg_num_df,
        reg_num_filter_df,
        on=[f'{ar}_{year_style}_period', region_corporation],
        how='inner',
    )
    # sep_year_reg_num_top_df = pd.merge(
    #     sep_year_reg_num_df,
    #     reg_num_filter_df[[region_corporation]],
    #     on=[region_corporation], 
    #     how='inner'
    # )
    sep_year_reg_num_top_df = sep_year_reg_num_df[sep_year_reg_num_df[region_corporation].isin(reg_num_filter_df[region_corporation])]
sep_year_reg_num_top_df

#%%
reg_num_top_df = pd.concat([all_reg_num_top_df, sep_year_reg_num_top_df], 
                           axis='index', ignore_index=True)
reg_num_top_df

# output_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
reg_num_top_df.to_csv(f'{output_dir}{output_condition}.csv', 
                      encoding='utf-8', 
                      sep=',', 
                      index=False)

# %%
