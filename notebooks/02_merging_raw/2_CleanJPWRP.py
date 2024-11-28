#! (root)/notebooks/02_merging_raw/2_CleanJPWRP.py python3
# -*- coding: utf-8 -*-

#%%
# %load 0_LoadLibraries.py
## Import Library
### Processing Data
from glob import glob
import pandas as pd
import numpy as np


### Visualization
from IPython.display import display

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format

## Initialize Global Variables
global DATA_DIR, OUTPUT_DIR
DATA_DIR = '../../data/original/internal/bulk/JPWRP/'
OUTPUT_DIR = '../../data/interim/internal/bulk/'


#%%
# Set Extract Information
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

# Load and Store Original Data 
# Weekly Patent Registration Data
original_df_dict = {file: [
                            pd.read_csv(path,
                                       sep='\t', 
                                       encoding='utf-8', 
                                #        dtype=str)\
                                       dtype=str, 
                                       usecols=needed_col_dict[file])\
                            for path in glob(DATA_DIR+file.split('.')[0]+'/*')
                           ]\
                            for file in needed_col_dict.keys()
                            }

#%%
# Grasp the Data
display(original_df_dict['upd_mgt_info_p.tsv'][0].head())

#%% 
# 
mgt_df_list = original_df_dict['upd_mgt_info_p.tsv'].copy()
mgt_df = pd.concat(mgt_df_list, 
                   ignore_index=True, 
                   axis='index')\
            .drop_duplicates(keep='first')\
            .reset_index(drop=True)
display(mgt_df.head())
# mgt_df.describe(include='all')

mgt_df.to_csv(f'{OUTPUT_DIR}reg_info.csv', 
              sep=',', 
              encoding='utf-8', 
              index=False)

#%%
hr_df_list = original_df_dict['upd_right_person_art_p.tsv'].copy()
hr_df = pd.concat([df for df in hr_df_list], 
                   ignore_index=True, 
                   axis='index')\
            [needed_col_dict['upd_right_person_art_p.tsv']]\
            .drop_duplicates(keep='first')\
            .reset_index(drop=True)
display(hr_df.head())
# hr_df.describe(include='all')

hr_df.to_csv(f'{OUTPUT_DIR}hr.csv', 
              sep=',', 
              encoding='utf-8', 
              index=False)
