#! (root)/notebooks/3_calculate/1_AggregateWeight.py python3
# -*- coding: utf-8 -*-
#%%
## Load Global Settings
%run ../../src/initialize/load_libraries.py
%run ../../src/initialize/initial_conditions.py

## Load Local Settings
%run 0_LoadLibraries.py


#%%
## Arrange variables
in_dir = f'{IN_IN_DIR}filtered_before_agg/'
filter_dir = '../../data/interim/internal/filter_after_agg/'
out_dir = '../../data/interim/internal/filtered_after_agg/'

#%%
# 全体
all_df = pd.read_csv(
                     f'{in_dir}japan.csv', 
                     encoding='utf-8', 
                     sep=',', 
                     usecols=['reg_num', 
                              region_corporation, 
                              f'{ar}_{year_style}', 
                              'ipc', 
                              'schmoch35'], 
                            #   f'{classification}'], 
                     dtype={'reg_num': str, 
                            region_corporation: str, 
                            f'{ar}_{year_style}': np.int64, 
                            'ipc': str, 
                            'schmoch35': str},)\
            .query(
                   f'{year_start} <= {ar}_{year_style} <= {year_end}'
                   )

if 'ipc' in classification:
    all_df = all_df.assign(
                        #    **{classification: (lambda x: x['ipc'].str[:digit])}
                           **{classification: all_df['ipc'].str[:digit]}
                          )
all_df = all_df.filter(
                       items=['reg_num',
                              region_corporation, 
                              f'{ar}_{year_style}', 
                              classification])\
               .drop_duplicates()

display(all_df.head())
print(all_df['right_person_name'].nunique())

#%%
# 各期間
sep_year_df_dict = {f'{year}-{year+year_range-1}': all_df.query(f'{year} <= {ar}_{year_style} < {year+year_range}')
                     for year in range(year_start, year_end+1, year_range)}

#%%
# 特許分類による重みづけ
# 全体
if class_weight == 'fraction':
    all_df = weight.by_classification(all_df, region_corporation, classification)
elif class_weight == 'duplication':
    all_df['class_weight'] = 1
all_df = all_df.assign(
                       **{f'{ar}_{year_style}_period': f'{year_start}-{year_end}'}
                      )

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
reg_num_filter_df = pd.read_csv(f'{filter_dir}{condition}.csv',
                                encoding='utf-8',
                                sep=',', 
                                usecols=[f'{ar}_{year_style}_period', region_corporation],
                                dtype=str)
reg_num_filter_df

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 15
plot_df = all_reg_num_df[['right_person_name', 'reg_num']]\
                        .groupby('right_person_name', as_index=False)\
                        .sum()\
                        .sort_values('reg_num', ascending=True)\
                        .assign(
                            reg_p = lambda x: x['reg_num'] / x['reg_num'].sum()
                        )

def ccdf(node_degree_dict):
        freq_array = np.array(np.bincount(list(node_degree_dict.values)))
        p_list = []
        cumsum = 0.0
        s = float(freq_array.sum())
        for freq in freq_array:
            if freq != 0:
                cumsum += freq / s
                p_list.append(cumsum)
            else:
                p_list.append(cumsum)
                
        ccdf_array = np.array(p_list)
        # if ccdf_array[0] == 0:
        #     ccdf_array[0] = 1.0
        return ccdf_array
# plt.
# np.bincount(ccdf(round(plot_df['reg_num']).value_counts())))
# s = np.bincount(round(plot_df['reg_num']).value_counts().values)
# cdf = np.array([i/s for i in np.bincount(round(plot_df['reg_num']).value_counts().values)]).cumsum()
cdf = ccdf(round(plot_df['reg_num']).value_counts())
plt.scatter([_ for _ in range(len(cdf))], cdf, 
            color='red', 
            s=15)
plt.xlabel('Weighted Patent Counts')
plt.ylabel('Cumlative Distribution Function (CDF)')
plt.xscale('log')
plt.axhline(y=0.93, color='black', linestyle='--')
plt.axvline(x=21, color='black', linestyle='--')

#%%
for i,_ in enumerate(cdf):
    if _ > 0.93:
        print(i)
        break


#%%
all_df[all_df['right_person_name'].isin(reg_num_filter_df[region_corporation])]
#%%

if extract_population == 'all':
    all_reg_num_top_df = pd.merge(
        all_reg_num_df,
        reg_num_filter_df,
        on=[f'{ar}_{year_style}_period', region_corporation],
        how='inner',
        copy=False
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
display(reg_num_top_df)

# output_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
reg_num_top_df.to_csv(f'{out_dir}{condition}.csv', 
                      encoding='utf-8', 
                      sep=',', 
                      index=False)

# %%
