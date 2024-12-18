#! (root)/notebooks/03_cleansing_filtering/1_FizSpelling.py python3
# -*- coding: utf-8 -*-

# %%
# Import Library
import pytz
import datetime
import regex
import re
from glob import glob
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

# Processing Data

# Initialize Global Variables
global data_dir, ex_data_dir, output_dir
data_dir = '../../data/interim/internal/merged/'
ex_data_dir = '../../data/processed/external/'
output_dir = '../../data/interim/internal/fixed/'


# %%
df = pd.read_csv(sorted(glob(data_dir + '*.csv'))[-1],
                 sep=',',
                 encoding='utf-8',
                 dtype=str)
# df.head()
print('表記ゆれ処理前の特許権者数：', df['right_person_name'].nunique())
print('表記ゆれ処理前の特許数：', df['reg_num'].nunique())
print('表記ゆれ処理前のIPC数：', df['ipc'].nunique())

# %%
trans_kanjikana_df = pd.read_csv(f'{ex_data_dir}letter/kanjikana.csv',
                                 sep=',',
                                 encoding='utf-8',
                                 dtype=str)
company_master_df = pd.read_csv(f'{ex_data_dir}nistep/company_master.csv',
                                sep=',',
                                encoding='utf-8',
                                dtype=str)
display(company_master_df)

# %%
trans_specialstr_tuple = (
    ('?b', '高'),
    ('?C', '吉'),
    ('?D', '塚'),
    ('?F', '崎'),
    ('?H', '徳'),
    ('?P', '濾'),
    ('?芟ｴ', '桑原'),
    ('??', '')
)
for trans_specialstr in trans_specialstr_tuple:
    df['right_person_name'] = df['right_person_name'].str.replace(
        trans_specialstr[0], trans_specialstr[1], regex=False)

# デバッグ用コード
# list(df[df['right_person_name'].str.contains('??', regex=False)]['right_person_name'].unique())

# %%
# 一文字から一文字への変換辞書
trans_one_letter_dict = dict(zip(trans_kanjikana_df['old_jikei'].values,
                                 trans_kanjikana_df['new_jikei'].values))

# 消す文字のリスト
trans_noise_list = ['\u3000', '\?ｫ', '\?ｬ', '▲', '▼', ' ']

# 旧字体を新字体に変換，消す文字を消す
for col_name in ['right_person_addr', 'right_person_name']:
    df[col_name] = df[col_name].str.replace('|'.join(trans_noise_list), '', regex=True)\
        .str.translate(str.maketrans(trans_one_letter_dict))

for col_name in ['before_name', 'after_name']:
    company_master_df[col_name] = company_master_df[col_name].str.replace('|'.join(trans_noise_list), '', regex=True)\
                                                             .str.translate(str.maketrans(trans_one_letter_dict))

#%%
# 省庁合併のタプルを作る用のデータ
minis_office_df = pd.read_csv(f'{ex_data_dir}ministry/minis_office.csv', 
                              encoding='utf-8', sep=',', dtype=object)
minis_office_df.head()

# 大臣とかの修正
trans_minister_tuple = tuple([('大臣', '省')])\
                       +tuple(zip(minis_office_df['old_office'].values, minis_office_df['after_office'].values))
# trans_minister_tuple

exception_list = []
i = 0
for trans_minister in trans_minister_tuple:
    if i in [0, 31, 37, 47, 51, 52, 56, 61, 63, 73, 74, 88, 89, 91, 92, 93, 94, 96, 97, 98, 100, 102, 104, 109, 110, 112, 115, 116, 117, 118, 119, 120, 121, 122, 123]:
        exception_list.append(df[df['right_person_name'].str.contains(trans_minister[0])]['right_person_name'].str.replace(trans_minister[0], '').unique())
        df['right_person_name'] = df['right_person_name'].str.replace(trans_minister[0], trans_minister[1], regex=True)
    i += 1
# df


#%%
# 何らかの長の表記ゆれ
trans_top_tuple = (
               ('総長|長官', ''), 
               ('所長', '所'), 
               ('局長', '局'), 
               ('院長', '院'), 
               ('大学{1,2}長', '大学'), 
               ('校長', '校'), 
               ('課長', '課'), 
               ('部長', '部'), 
               ('センター長', 'センター'), 
               ('機構長', '機構'), 
               ('署長', '署'), 
               ('場長', '場'), 
               ('市長', '市')
               )

# 企業名の表記ゆれ
trans_lp_tuple = (('コーポレイシヨン', 'コーポレーシヨン'), 
                  ('株式会社', '株式'), 
                  ('株式', '株式会社'))

trans_else_tuple = (
               ('パナソニツクＩＰマネジメント株式会社', 'パナソニツク株式会社'), 
               ('ッ', 'ツ')
                    )

for trans_top in trans_top_tuple+trans_lp_tuple:
    df['right_person_name'] = df['right_person_name'].str.replace(trans_top[0], trans_top[1], regex=True)
    for col_name in ['before_name', 'after_name']:
        company_master_df[col_name] = company_master_df[col_name].str.replace(trans_top[0], trans_top[1], regex=True)
df.head()


# # 合併
df = pd.merge(df, company_master_df.rename(columns={'before_name':'right_person_name'}), 
              on='right_person_name', 
              how='left')\
             .rename(columns={'after_name':'after_right_person_name'})
df['after_right_person_name'] = df['after_right_person_name'].fillna(df['right_person_name'])
df = df.drop(columns=['right_person_name'])\
       .rename(columns={'after_right_person_name':'right_person_name'})
df.head()


#%%
# df.to_csv('../Data/Tmp/fixed.csv', encoding='utf-8', sep=',', index=False)
jst = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(jst)
str_now = now.strftime('%Y%m')

df.to_csv(f'{output_dir}{str_now}.csv', 
          sep=',', 
          encoding='utf-8', 
          index=False)
