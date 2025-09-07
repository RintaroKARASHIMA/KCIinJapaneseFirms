#! (root)/notebooks/02_merging_raw/6_MergeBulkStack.py python3
# -*- coding: utf-8 -*-

#%%
## If necessary, Import Libraries
# %run ../../src/initialize/load_libraries.py
# %run 0_LoadLibraries.py

### Processing Data
from glob import glob
import pandas as pd
import numpy as np
import IPython.display as display
import time
import datetime
import pytz

## Initialize Global Variables
global DATA_DIR, OUTPUT_DIR
DATA_DIR = '../../data/interim/internal/'
OUTPUT_DIR = '../../data/interim/internal/merged/'

#%%
file_name_list = [
                  'hr', # 特許権者
                  'ipc', # IPC
                  'reg_info' # 登録情報
                  ]
bulk_stack_df_dict = {
                      file_name: pd.concat([pd.read_csv(f'{DATA_DIR}/{bs}/{file_name}.csv', 
                                                        sep=',', 
                                                        encoding='utf-8', 
                                                        dtype=str) for bs in ['bulk', 'stack']], 
                                           ignore_index=True, 
                                           axis='index')
                      for file_name in file_name_list
                      }

#%%
df = pd.merge(bulk_stack_df_dict['reg_info'].copy(), bulk_stack_df_dict['hr'].copy(), 
              on='reg_num', 
              how='inner')

df = pd.merge(df, bulk_stack_df_dict['ipc'].copy(), 
              on='app_num', 
              how='inner')

df = df.drop(columns=['app_num'])\
       .drop_duplicates(keep='first')\
       .reset_index(drop=True)


#%%
jst = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(jst)
str_now = now.strftime('%Y%m')

df.to_csv(f'{OUTPUT_DIR}{str_now}.csv', 
          sep=',', 
          encoding='utf-8', 
          index=False)
