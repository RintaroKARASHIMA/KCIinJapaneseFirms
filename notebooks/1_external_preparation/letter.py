import pandas as pd
import numpy as np
import re
from glob import glob
import regex


global output_dir
output_dir = '../../data/processed/external/letter/'

# 旧字体の辞書作る用のデータ
old_new_letter_df = pd.read_html(
    'https://www.asahi-net.or.jp/~ax2s-kmtn/ref/old_chara.html',
    encoding='utf-8'
)

df_tmp_list = []
for i in [1, 2]:
    for c in range(2, 14+1, 6):
        df_tmp = old_new_letter_df[i].iloc[:, c:c+1+1]
        df_tmp.columns = ['new_jikei', 'old_jikei']
        df_tmp_list.append(df_tmp)

old_new_letter_df = pd.concat(df_tmp_list, 
                              axis='index')\
                      .dropna(how='all')\
                      .reset_index(drop=True)

katakana_dict = {
    'ァ': 'ア',
    'ィ': 'イ',
    'ゥ': 'ウ',
    'ェ': 'エ',
    'ォ': 'オ',
    'ヵ': 'カ',
    'ヶ': 'ケ',
    'ㇰ': 'ク',
    'ㇱ': 'シ',
    'ㇲ': 'ス',
    'ッ': 'ツ',
    'ㇳ': 'ト',
    'ㇴ': 'ヌ', 
    'ㇵ': 'ハ',
    'ㇶ': 'ヒ',
    'ㇷ': 'フ',
    'ㇸ': 'ヘ',
    'ㇹ': 'ホ',
    'ㇺ': 'ム',
    'ャ': 'ヤ',
    'ュ': 'ユ',
    'ョ': 'ヨ',
    'ㇻ': 'ラ',
    'ㇼ': 'リ',
    'ㇽ': 'ル',
    'ㇾ': 'レ',
    'ㇿ': 'ロ',
    'ヮ': 'ワ', 
    '―': 'ー',  
    '－': 'ー',
    '－': 'ー'
    }

katakana_df = pd.DataFrame(katakana_dict.items(), 
                           columns=['old_jikei', 'new_jikei'])
# katakana_df

trans_letter_df = pd.concat([old_new_letter_df, katakana_df], 
                            axis='index', 
                            ignore_index=True)
trans_letter_df


trans_letter_df.to_csv(f'{output_dir}kanjikana.csv', 
                       sep=',', 
                       encoding='utf-8', 
                       index=False)
