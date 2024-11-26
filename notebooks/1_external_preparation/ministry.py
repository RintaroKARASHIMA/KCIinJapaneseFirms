
#%%
## Import Library
import pandas as pd
import numpy as np
import re
import regex

#%%
# Set Global Variables
global output_dir
output_dir = '../../data/processed/external/ministry/'

#%%
# 省庁合併のタプルを作る用のデータ
minis_office_df_list = pd.read_html('https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E8%A1%8C%E6%94%BF%E6%A9%9F%E9%96%A2')

now_minis_office_df = minis_office_df_list[2].copy().iloc[:, 1:-1]
now_minis_office_df.columns = ['office_1', 'office_2', 'office_3', 'minister_name']
now_minis_office_df = now_minis_office_df.ffill(axis='index')
for col in now_minis_office_df.columns:
    now_minis_office_df[col] = now_minis_office_df[col].str.replace('\[.*\]|（|）', '', regex=True)
now_minis_office_df = pd.concat([now_minis_office_df[['office_2', 'office_1']].rename(columns={'office_2':'office'}), 
                                 now_minis_office_df[['office_3', 'office_1']].rename(columns={'office_3':'office'}), 
                                 ], axis='index', ignore_index=True)\
                                .drop_duplicates()
now_minis_office_df = now_minis_office_df[now_minis_office_df['office']!=now_minis_office_df['office_1']]
now_minis_office_df.columns = ['old_office', 'after_office']

#%%
old_minis_office_df = minis_office_df_list[4].copy()
old_minis_office_df['設置年'] = old_minis_office_df['設置年月日'].str[:4].replace('－.*|明治*', np.nan, regex=True).astype(np.float64)
old_minis_office_df['廃止年'] = old_minis_office_df['廃止年月日'].str[:4].replace('－.*', np.nan, regex=True).astype(np.float64)
old_minis_office_df = old_minis_office_df[(old_minis_office_df['廃止年'] >= 1971)\
                                          |(old_minis_office_df['廃止年'].isna())\
                                          &(old_minis_office_df['設置年'] <= 2015)]
old_minis_office_df = old_minis_office_df[['名称', '主な後身']]
old_minis_office_df['名称'] = old_minis_office_df['名称'].str.replace('\[注.*\]|（.*）', '', regex=True)
old_minis_office_df['主な後身'] = old_minis_office_df['主な後身'].str.replace(' .*庁', '', regex=True).replace('－', np.nan)
old_minis_office_df = old_minis_office_df[['名称', '主な後身']]#.dropna()
old_minis_office_df.columns = ['old_office', 'after_office']
# old_minis_office_df['長'] = old_minis_office_df['長'].str.replace('\[注.*\]|（.*）', '', regex=True)

#%%
minis_office_df = pd.concat([now_minis_office_df, 
                             old_minis_office_df
                             ], axis='index', ignore_index=True)
minis_office_df = minis_office_df.drop_duplicates().drop_duplicates(subset=['old_office'], keep='first').reset_index(drop=True)
minis_office_df['after_office'] = minis_office_df['after_office'].fillna(minis_office_df['old_office'])
conditions = [
    (minis_office_df['old_office'].isin(['宮内庁', '金融庁', '消費者庁', 'こども家庭庁', '消防庁', '検察庁', '在外公館', 
                                         '日本学士院', 'スポーツ庁', '文化庁', '林野庁', '水産庁', '資源エネルギー庁', 
                                         '特許庁', '中小企業庁', '国土地理院', '観光庁', '気象庁', '情報本部', '警察庁', 
                                         '日本芸術院', '内閣', '大蔵省', '文部省', '厚生省', '運輸省', '農林省', '労働省', '行政管理庁', 
                                         '建設省', '通商産業省', '郵政省', '総理府', '法務省', '防衛庁', '経済企画庁', '科学技術庁', 
                                         '自治省', '環境', '国土庁', '総務庁', '内閣府', '総務省', '財務省', '文部科学省', '厚生労働省', 
                                         '農林水産省', '経済産業省', '国土交通省', '環境庁', '環境省', '防衛省', '復興庁'
                                         ])), 
    (minis_office_df['old_office'].str.contains('北海道開発')),
]
choices = ['.*' + minis_office_df['old_office'] + '.*', '.*' + minis_office_df['old_office'] + '.*|北海道開発局.*']
minis_office_df['old_office'] = np.select(conditions, choices, default='.*' + minis_office_df['old_office'].str[:-1] + '.*')
minis_office_df = minis_office_df[~minis_office_df['old_office'].str.contains('、')]
# minis_office_df['old_office'].unique()

# minis_office_df.head()

#%%
minis_office_df.to_csv(f'{output_dir}minis_office.csv', 
                       sep=',', 
                       encoding='utf-8', 
                       index=False)
