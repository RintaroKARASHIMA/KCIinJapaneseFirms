#! (root)/notebooks/03_cleansing_filtering/4_FlagAddress.py python3
# -*- coding: utf-8 -*-

# %%
# Import Library
import pandas as pd
import numpy as np
import IPython.display as display

## If necessary, Import Original Modules
# from initialize import initial_conditions
# reload(initial_conditions)
# from calculation import weight
# from visualize import rank as vr


# Initialize Global Variables
global ex_data_dir, in_data_dir, in_filter_dir
ex_data_dir = '../../data/processed/external/'
in_data_dir = '../../data/interim/internal/filtered_before_agg/'
in_filter_dir = '../../data/interim/internal/filter_before_agg/'

# %%
pref_list = [
    '北海道',
    '青森県',
    '岩手県',
    '宮城県',
    '秋田県',
    '山形県',
    '福島県',
    '茨城県',
    '栃木県',
    '群馬県',
    '埼玉県',
    '千葉県',
    '東京都',
    '神奈川県',
    '新潟県',
    '富山県',
    '石川県',
    '福井県',
    '山梨県',
    '長野県',
    '岐阜県',
    '静岡県',
    '愛知県',
    '三重県',
    '滋賀県',
    '京都府',
    '大阪府',
    '兵庫県',
    '奈良県',
    '和歌山県',
    '鳥取県',
    '島根県',
    '岡山県',
    '広島県',
    '山口県',
    '徳島県',
    '香川県',
    '愛媛県',
    '高知県',
    '福岡県',
    '佐賀県',
    '長崎県',
    '熊本県',
    '大分県',
    '宮崎県',
    '鹿児島県',
    '沖縄県',
]
df = pd.read_csv(in_data_dir + 'addedclassification.csv', 
                encoding='utf-8', 
                sep=',', 
                dtype=str)
df

city_df = pd.read_csv(
    in_filter_dir + "jp_address_flagged.csv", encoding="utf-8", sep=",", dtype=str
)
city_df

#%%
japan_df = pd.merge(df, city_df, left_on='right_person_addr', right_on='city', 
         how='left').drop(columns=['city']).copy()
for pref in pref_list:
    japan_df['prefecture'] = np.where(japan_df['right_person_addr'].str.startswith(pref), 
                                pref, 
                                japan_df['prefecture'])
japan_df = japan_df.dropna(subset=['prefecture'])\
                    .drop(columns=['right_person_addr'])\
                    .rename(columns={'prefecture': 'right_person_addr'})
japan_df

#%%
japan_df['right_person_addr'] = japan_df['right_person_addr'].str.replace('省略', '')
japan_df['right_person_addr'] = pd.Categorical(japan_df['right_person_addr'], 
                                               categories=pref_list+[''], 
                                               ordered=True)
japan_df = japan_df.sort_values(by=['app_year', 'right_person_name', 'right_person_addr'], 
                                ascending=True)
japan_df['right_person_addr'] = japan_df['right_person_addr'].replace('', np.nan).ffill()
japan_df

#%%
japan_df.to_csv(in_data_dir + 'japan.csv', 
                encoding='utf-8', 
                sep=',', 
                index=False)