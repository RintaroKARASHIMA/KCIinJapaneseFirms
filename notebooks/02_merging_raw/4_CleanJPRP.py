#! (root)/notebooks/02_merging_raw/4_CleanJPRP.py python3
# -*- coding: utf-8 -*-

#%%
## If necessary, Import Libraries
import pandas as pd
import numpy as np
import IPython.display as display

### Processing Data
# %run ../../src/initialize/load_libraries.py
# %run 0_LoadLibraries.py

## Initialize Global Variables
global DATA_DIR, OUTPUT_DIR
DATA_DIR = '../../data/original/internal/stack/JPRP/'
OUTPUT_DIR = '../../data/interim/internal/stack/'

#%%
needed_col_dict = {
                   'upd_mgt_info_p.tsv':[
                                        # 'processing_type', # 処理種別
                                        #  'law_cd', # 四法コード
                                         'reg_num', # 登録番号
                                        #  'split_num',# 分割番号
                                         'app_num', # 出願番号
                                         'app_year_month_day', # 出願日
                                         'set_reg_year_month_day', # 設定登録日
                                         # 'pri_cntry_name_cd' # 優先権国コード
                                         ], 
                   'upd_right_person_art_p.tsv':[
                                                #  'law_cd', # 四法コード
                                                 'reg_num', # 登録番号
                                                #  'split_num', # 分割番号
                                                # 'right_person_appl_id', # 特許権者申請人ID
                                                 'right_person_addr', # 特許権者住所
                                                 'right_person_name' # 特許権者名
                                                 ]
                   }

#%%
original_df_dict = {file: pd.read_csv(DATA_DIR + file,
                                      sep='\t', 
                                      encoding='utf-8', 
                                      dtype=str, 
                                      usecols=needed_col_dict[file])
                    for file in needed_col_dict.keys()}


#%%
## 登録情報マスタ
mgt_df = original_df_dict['upd_mgt_info_p.tsv'].copy()
# mgt_df = mgt_df[needed_col_dict['upd_mgt_info_p.tsv']]
mgt_df.head()
# mgt_df.describe(include='all')
mgt_df.to_csv(f'{OUTPUT_DIR}reg_info.csv', 
              sep=',', 
              encoding='utf-8', 
              index=False)

#%%
## 特許権者全数ver.
hr_df = original_df_dict['upd_right_person_art_p.tsv'].copy()
# hr_df = hr_df[needed_col_dict['upd_right_person_art_p.tsv']]
hr_df.head()
# hr_df#.describe(include='all')
hr_df.to_csv(f'{OUTPUT_DIR}hr.csv', 
             sep=',', 
             encoding='utf-8', 
             index=False)
