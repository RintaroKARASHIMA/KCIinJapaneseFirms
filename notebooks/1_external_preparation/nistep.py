import pandas as pd
import numpy as np

global data_dir, output_dir
data_dir = '../../data/original/external/nistep/'
output_dir = '../../data/processed/external/nistep/'

original_df_dict = pd.read_excel(f'{data_dir}企業名辞書v2023_1.xlsx', 
                                 sheet_name=None, 
                                 engine='openpyxl')

master_df = original_df_dict['企業名辞書v2023_1'].copy()
col_dict = dict(zip(master_df.columns, master_df.iloc[0, :].copy()))
needed_col_list = ['comp_id', 'comp_name', 'comp_code', 'post_comp_id', 'parent_compid', 'parent_comp_name']

master_df.columns = [col_dict[col] for col in master_df.columns]
master_df = master_df[needed_col_list].iloc[1:, :].replace('\\N', np.nan)

# 親子関係
parent_df = master_df[['comp_id', 'parent_compid', 'parent_comp_name']].drop_duplicates().dropna()
parent_df

# 法人格込みの企業名
id_name_code_df = master_df[['comp_id', 'comp_name', 'comp_code']].drop_duplicates().astype(str)
conv_dict = {'KB':'株式会社', 
             'YG':'有限会社', 
             'GD':'合同会社', 
             'SG':'相互会社', 
             'GS':'合資会社', 
             'GM':'合名会社'}
id_name_code_df['comp_kind'] = id_name_code_df['comp_code'].str[:2].replace(conv_dict)
conditions = [
    (id_name_code_df['comp_code'].str[-1]==i)&~(id_name_code_df['comp_name'].str.contains('会社'))
    for i in ['1', '2']
]
choice = [id_name_code_df['comp_kind']+id_name_code_df['comp_name'], 
          id_name_code_df['comp_name']+id_name_code_df['comp_kind']]
id_name_code_df['full_comp_name'] = np.select(conditions, choice, default=id_name_code_df['comp_name'])
id_name_code_df = id_name_code_df[['comp_id', 'full_comp_name']].drop_duplicates()
id_name_code_df

# 新旧関係
before_after_df = pd.merge(master_df[['comp_id', 'post_comp_id']].drop_duplicates().rename(columns={'comp_id':'1st_comp_id'}),
                           master_df[['comp_id', 'post_comp_id']].drop_duplicates().rename(columns={'comp_id':'2nd_comp_id', 'post_comp_id':'3rd_comp_id'}), 
                           left_on='post_comp_id', right_on='2nd_comp_id', how='left')\
                           [['1st_comp_id', '2nd_comp_id', '3rd_comp_id']]
before_after_df = pd.merge(before_after_df,
                           master_df[['comp_id', 'post_comp_id']].drop_duplicates().rename(columns={'comp_id':'3rd_comp_id', 'post_comp_id':'4th_comp_id'}), 
                           on='3rd_comp_id', how='left')\
                           [['1st_comp_id', '2nd_comp_id', '3rd_comp_id', '4th_comp_id']]
for i in range(4, 11+1):
    before_after_df = pd.merge(before_after_df,
                               master_df[['comp_id', 'post_comp_id']].drop_duplicates().rename(columns={'comp_id':f'{i}th_comp_id', 
                                                                                                        'post_comp_id':f'{i+1}th_comp_id'}), 
                               on=f'{i}th_comp_id', how='left')
before_after_df['after_id'] = before_after_df['11th_comp_id'].copy()
for i in range(10, 4-1, -1):
    before_after_df['after_id'] = before_after_df['after_id'].fillna(before_after_df[f'{i}th_comp_id'])
    if i == 4:
        for j in ['3rd', '2nd', '1st']:
            before_after_df['after_id'] = before_after_df['after_id'].fillna(before_after_df[f'{j}_comp_id'])
before_after_df = before_after_df[['1st_comp_id', 'after_id']]\
                                 .rename(columns={'1st_comp_id':'before_id'})\
                                 .astype(np.int64)
before_after_df

# 新旧親子関係
before_parent_df = pd.merge(before_after_df, parent_df, 
                            left_on='after_id', right_on='comp_id', how='left')
# before_parent_df['after_parentid'] = before_parent_df['parent_compid'].fillna(before_parent_df['after_id']).astype(np.int64)
# before_parent_df = before_parent_df[['before_id', 'after_parentid']].rename(columns={'after_parentid':'after_id'})
before_parent_df = before_parent_df[['before_id', 'after_id']]
before_parent_df

before_after_name_df = pd.merge(before_parent_df.astype(str), id_name_code_df, 
                                left_on='before_id', right_on='comp_id', how='left')\
                                [['before_id', 'after_id', 'full_comp_name']]\
                                .rename(columns={'full_comp_name':'before_name'})
before_after_name_df = pd.merge(before_after_name_df.astype(str), id_name_code_df, 
                                left_on='after_id', right_on='comp_id', how='left')\
                                [['before_name', 'full_comp_name']]\
                                .rename(columns={'full_comp_name':'after_name'})
before_after_name_df

letter_conv_df = pd.read_csv('../../data/processed/external/letter/kanjikana.csv')

# 旧字体と新字体の辞書を作成
old_kanji = list(letter_conv_df['old_jikei'])
new_kanji = list(letter_conv_df['new_jikei'])
jitai_dict = dict(zip(old_kanji, new_kanji))

# 辞書を検索して文字列を置き換える関数を作成
def kyujitai_to_shinjitai(text):
    encoded_text = text.translate(str.maketrans(jitai_dict))
    return encoded_text
#%%
before_after_name_df['before_name'] = before_after_name_df['before_name'].str.translate(str.maketrans(jitai_dict))
before_after_name_df['after_name'] = before_after_name_df['after_name'].str.translate(str.maketrans(jitai_dict))
before_after_name_df

#%%
before_after_name_df.to_csv(f'{output_dir}company_master.csv', 
                            sep=',', 
                            encoding='utf-8', 
                            index=False)
