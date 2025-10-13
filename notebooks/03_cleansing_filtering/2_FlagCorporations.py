#! (root)/notebooks/03_cleansing_filtering/2_FilteringBeforeAgg.py python3
# -*- coding: utf-8 -*-


#%%
## If necessary, Import Libraries
### Processing Data
from glob import glob
import pandas as pd
import numpy as np
import IPython.display as display

### Original Module
from cleansing_filtering import CreateFilterBeforeAgg

## Initialize Global Variables
global data_dir, output_dir
data_dir = '../../data/interim/internal/fixed/'
filter_dir = '../../data/interim/internal/filter_before_agg/'
output_dir = '../../data/interim/internal/filtered_before_agg/'


#%%
# 企業統廃合処理あり
df = pd.read_csv(sorted(glob(data_dir + '*.csv'))[-1], 
                 encoding='utf-8', 
                 sep=',', 
                 dtype=str)
# df.head()

#%%
stage0_df = df.copy()
stage0_df = stage0_df[['reg_num', 
                       'app_year_month_day', 
                       'set_reg_year_month_day', 
                       'ipc', 
                       'right_person_addr', 
                       'right_person_name']]\
                     .dropna()
stage0_df.head()

#%%
pref_list = CreateFilterBeforeAgg.pref_list
fix_list = CreateFilterBeforeAgg.fix_list
jp_exception = pd.read_csv(f'{filter_dir}jp_address.csv', encoding='utf-8')
# jp_exception

# 住所による絞込
stage1_df = stage0_df.copy()

conditions = (
    (stage1_df['right_person_addr'].str.contains('|'.join(pref_list)+'|縣|県|日本国'))|(stage1_df['right_person_addr'].isin(fix_list))
)
stage1_clone_df = stage1_df[conditions].copy()
stage1_clone_df = pd.concat([stage1_clone_df, stage1_df[(stage1_df['right_person_name'].isin(stage1_clone_df['right_person_name']))\
                                      &(stage1_df['right_person_addr'].str.contains('省略'))]], 
                     ignore_index=True, axis='index')
stage1_clone_df = pd.concat([stage1_clone_df, stage1_df[stage1_df['right_person_name'].isin(jp_exception['name'])\
                                      &stage1_df['right_person_addr'].str.contains('省略')]], ignore_index=True, axis='index')
stage1_clone_df = stage1_clone_df.drop_duplicates(subset=['right_person_name', 'reg_num', 'ipc'], keep='first')
# stage1_df = stage1_clone_df.drop(columns=['right_person_addr'])

#%%
# 氏名による絞込
stage2_df = stage1_df.copy()
stage2_df = stage2_df[stage2_df['right_person_name'].str.contains('会社|法人|大学$|組合|機構$|研究所', regex=True)]

stage3_df = stage1_df[~stage1_df['right_person_name'].isin(stage2_df['right_person_name'])].copy()
# pd.DataFrame(stage3_df['right_person_name'].unique()).to_csv(f'{filter_dir}jp_firm.csv', encoding='utf-8', index=False, sep=',')

extra_jp_df = pd.read_csv(f'{filter_dir}jp_firm_flagged.csv', sep=',', encoding='utf-8', dtype=object)

stage4_df = stage1_df[stage1_df['right_person_name'].isin(extra_jp_df['name'])].copy()
stage4_df = pd.concat([stage2_df, stage4_df], 
                      ignore_index=True, 
                      axis='index')
stage4_df['right_person_name'] = stage4_df['right_person_name'].str.replace('東京都新宿区戸塚町１丁目１０４番地', '学校法人早稲田大学')
# stage4_df

# stage1_df[(stage1_df['app_year_month_day'].astype(np.int64).isin(range(19810401, 20160400)))&(stage1_df['right_person_name'].isin(stage4_df['right_person_name']))]\
#     .groupby('right_person_name')[['reg_num']].nunique().describe()

# 年月日から年に処理
# さらに年を年度に処理
stage5_df = stage4_df.copy()
stage5_df = stage5_df.rename(columns={'set_reg_year_month_day': 'reg_year_month_day'})
for ar in ['app', 'reg']:
    stage5_df[f'{ar}_year'] = stage5_df[f'{ar}_year_month_day'].str[:4].astype(np.int64)
    stage5_df[f'{ar}_month'] = stage5_df[f'{ar}_year_month_day'].str[4:6].astype(np.int64)
    stage5_df[f'{ar}_nendo'] = np.where(stage5_df[f'{ar}_month'] <= 3, 
                                        stage5_df[f'{ar}_year'] - 1, 
                                        stage5_df[f'{ar}_year'])

stage5_df = stage5_df.drop(columns=['app_year_month_day', 'app_month', 
                                    'reg_year_month_day', 'reg_month'])
#                     [['reg_num', 'right_person_name', 'reg_nendo', 'app_nendo', 'ipc']]\
#                     .rename(columns={'app_year_jp':'app_year', 'reg_year_jp':'reg_year'})
stage5_df.head()



stage5_df.to_csv(f'{output_dir}corporations.csv', 
                 encoding='utf-8', 
                 index=False, 
                 sep=',')
# stage6_df[['reg_num', 'right_person_name', 'app_year', 'ipc_class']]\
#          .to_csv('../Data/Dealed/app_notmerged.csv', 
#                  encoding='utf-8', 
#                  index=False, 
#                  sep=',')
