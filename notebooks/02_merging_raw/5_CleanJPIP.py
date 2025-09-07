#! (root)/notebooks/02_merging_raw/5_CleanJPRP.py python3
# -*- coding: utf-8 -*-

#%%
## If necessary, Import Libraries
# %run ../../src/initialize/load_libraries.py
# %run 0_LoadLibraries.py

import pandas as pd
import numpy as np
import IPython.display as display

## If necessary, Import Original Modules
# from initialize import initial_conditions
# reload(initial_conditions)
# from calculation import weight
# from visualize import rank as vr

## Initialize Global Variables
global DATA_DIR, OUTPUT_DIR
DATA_DIR = '../../data/original/internal/stack/JPIP/'
OUTPUT_DIR = '../../data/interim/internal/stack/'

#%%
needed_col_dict = {
                   'upd_dsptch_fin_ipc.tsv':[
                                            'doc_key_num', # 文献番号
                                            'ipc', # IPC
                                            'first_class_flg'
                                            ]
                   }

original_df_dict = {file: pd.read_csv(DATA_DIR + file, 
                                       sep='\t', 
                                       encoding='utf-8', 
                                       dtype=str, 
                                       usecols=needed_col_dict[file])\
                    for file in needed_col_dict.keys()}

#%%
ipc_df = original_df_dict['upd_dsptch_fin_ipc.tsv'].copy()

ipc_df['patent_flag'] = ipc_df['doc_key_num'].str[:1]
# ipc_df['doc_key_num'] = ipc_df['doc_key_num'].astype(np.int64) # ここでエラーが出れば，例外処理が必要
ipc_df['app_num'] = ipc_df['doc_key_num'].str[1:]
ipc_exception_df = ipc_df[ipc_df['doc_key_num'].str.contains('JPB')].copy()

# 特許文献のみを抽出
ipc_df = ipc_df[ipc_df['patent_flag']=='1']\
               .reset_index(drop=True)\
               .drop(['patent_flag', 'doc_key_num'], axis='columns')

# 第一分類を使用
ipc_df = ipc_df[ipc_df['first_class_flg']=='F'][['ipc', 'app_num']]
ipc_df

ipc_df.to_csv(f'{output_dir}ipc.csv', 
              sep=',', 
              encoding='utf-8', 
              index=False)
# ipc_exception_df.to_csv('../Data/Extracted/ipc_exception.csv', sep=',', encoding='utf-8', index=False)
