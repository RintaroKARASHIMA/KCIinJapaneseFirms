# %%
## Import Libraries
import pandas as pd
import numpy as np
from glob import glob
import re
import unicodedata

# import mojimoji

# %%
path_list = glob("../../data/original/external/ipc/*.xlsx")
print(path_list)
# original_df_list = [pd.read_excel(path, engine='openpyxl') for path in path_list]
original_df = pd.concat(
    [pd.read_excel(path, engine="openpyxl") for path in path_list], axis="index"
).iloc[:, [1, 3]]
display(original_df)

#%%
## 3 digits
digit3_df = original_df.copy()
digit3_df = digit3_df[(digit3_df['記号'].astype(str).str.len() <= 4)\
        & (digit3_df['記号'].astype(str).str.len() != 1)].fillna(np.nan)
digit3_df['記号'] = digit3_df['記号'].replace('＜注＞', np.nan).ffill()
digit3_df = digit3_df[digit3_df['記号'].astype(str).str.len()==3]\
                     .reset_index(drop=True)
digit3_df['タイトル'] = digit3_df['タイトル'].astype(str).str.replace('\u3000|\t|\n', '')
digit3_df = digit3_df.groupby('記号').agg({'タイトル': '\n'.join}).reset_index()
digit3_df

digit3_df.to_csv('../Data/Dealed/ipc_3digit.csv', sep=',', encoding='utf-8', index=False)

#%%
## 4 digits
df = original_df.copy().dropna(axis='index')
df = df[df['記号'].str.contains(r'^[A-Z]{1}\d{2}[A-Z]{1}$')].reset_index(drop=True)\
                  .rename(columns={'記号':'class', 'タイトル':'class_jp'})
for ind in range(len(df)):
    df.loc[ind, 'class_jp'] = unicodedata.normalize('NFKC', df.loc[ind, 'class_jp'])
df
for s in ['\u3000', '\t', '\n', r'']:
    df['class_jp'] = df['class_jp'].str.replace(s, '')
display(df.head())
display(df.info())
for ind in range(len(df)):
    df.loc[ind, 'class_jp'] = unicodedata.normalize('NFKC', df.loc[ind, 'class_jp'])
df

#%%
## margin
# df[df['class_jp'].str.contains(r'[０-９]')]

for ind in range(len(df)):
    df.loc[ind, 'class_jp'] = mojimoji.zen_to_han(df.loc[ind, 'class_jp'], kana=False, ascii=False, digit=True)
df['class_jp'].unique()
df.to_csv('../Data/Dealed/IPC_class4digit.csv', sep=',', encoding='utf-8')