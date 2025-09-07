# %%

import time
from bs4 import BeautifulSoup
from urllib import request
import pandas as pd
import numpy as np
import re

from IPython.display import display
from pprint import pprint

raw_dir = '../../data/raw/external/grp/'
proc_dir = '../../data/processed/external/grp/'

#%%
main_html_path = 'https://www.esri.cao.go.jp/jp/sna/data/data_list/kenmin/files/files_kenmin.html'
sub_html_path_dict = {}
with request.urlopen(main_html_path) as response:
    content = response.read().decode('utf-8')
    soup = BeautifulSoup(content, 'html.parser')
    for li in soup.find(id='mainContents').find('ul', class_='bulletList').find_all('li'):
        key = li.text.split('‐')[0]
        if not ('※' in li.text) and (key not in sub_html_path_dict):
            sub_html_path_dict[key] = 'https://www.esri.cao.go.jp/jp/sna/data/data_list/kenmin/files/'\
                                       + li.find('a').get('href')

pprint(sub_html_path_dict)
    
# %%
excel_path_base = 'https://www.esri.cao.go.jp/jp/sna/data/data_list/kenmin/files/contents/'
excel_path_dict = {}
for key, value in sub_html_path_dict.items():
    with request.urlopen(value) as response:
        time.sleep(3)
        content = response.read().decode('utf-8')
        soup = BeautifulSoup(content, 'html.parser')
        excel_path_dict[key] = {}
        for a in soup.find(id='mainContents').find('ol').find_all('a'):
            for name in ['実質', '総人口']:
                if name in a.text:
                    with request.urlopen(excel_path_base + a.get('href')) as web_file:
                        time.sleep(3)
                        with open(raw_dir + key + '_' + name + '.' + a.get('href').split('.')[-1], 'wb') as local_file:
                            local_file.write(web_file.read())
                            excel_path_dict[key][name] = raw_dir + key + '_' + name + '.' + a.get('href').split('.')[-1]
pprint(excel_path_dict)

# %%
grp_df = pd.DataFrame()
capita_df = pd.DataFrame()
for key, value in excel_path_dict.items():
    for name, excel_path in value.items():
        excel_sheet_names = pd.ExcelFile(excel_path).sheet_names
        df = pd.read_excel(
              excel_path, 
              engine='openpyxl' if excel_path.endswith('.xlsx') else 'xlrd', 
              sheet_name=[s for s in excel_sheet_names if re.match(r'.*実.{1,2}$', s)][0],
              header=4,
              index_col=1
              )\
              .filter(regex=r'^(?!Unnamed.*)', axis='columns')\
              .filter(regex=r'.*[都道府県]$', axis='index')
        if name == '実質': grp_df = pd.concat([grp_df, df], axis='columns')
        if name == '総人口': capita_df = pd.concat([capita_df, df], axis='columns')
grp_df = pd.melt(
                 grp_df, 
                 var_name='year', 
                 value_name='GRP', 
                 ignore_index=False
                 )\
           .reset_index(names='prefecture', drop=False)\
           .drop_duplicates(subset=['prefecture', 'year'], keep='first')
capita_df = pd.melt(
                 capita_df, 
                 var_name='year', 
                 value_name='capita', 
                 ignore_index=False
                 )\
           .reset_index(names='prefecture', drop=False)\
           .drop_duplicates(subset=['prefecture', 'year'], keep='first')

grp_capita_df = pd.merge(grp_df, 
                      capita_df, 
                      on=['prefecture', 'year'], 
                      how='inner'
                      )\
                     .assign(
                         GRP = lambda x: x['GRP'].replace('-', np.nan).astype(np.float64),
                         capita = lambda x: x['capita'].astype(np.float64), 
                         GRP_yen = lambda x: x['GRP'] * 1_000_000,
                         GRP_per_capita_yen = lambda x: x['GRP_yen'] / x['capita'],
                         GRP_per_capita_1000yen = lambda x: x['GRP_per_capita_yen'] / 1000
                     )                     
display(grp_capita_df)
# %%
area_df = pd.read_excel(
    'https://www.stat.go.jp/data/nihon/zuhyou/n250100100.xlsx', 
    engine='openpyxl',
    header=3
    )\
    .dropna(subset=['都道府県'])\
    .filter(regex=r'(都道府県.*)|(面積.*)$', axis='columns')\
    .iloc[:-2, :]
area_df = pd.concat([
    area_df.filter(regex=r'(都道府県.*)', axis='columns').stack().reset_index(drop=True),
    area_df.filter(regex=r'(面積.*)', axis='columns').stack().reset_index(drop=True)
    ], axis='columns', ignore_index=True)\
    .rename(columns={0: 'prefecture', 1: 'area'})\
    .query('prefecture != "全国"').reset_index(drop=True)
    
area_df
# %%
grp_capita_area_df = pd.merge(
    grp_capita_df.assign(
        _non_prefix = lambda x: np.where(x['prefecture'] == '北海道', '北海道', x['prefecture'].str[:-1])
        ),
    area_df,
    left_on='_non_prefix',
    right_on='prefecture',
    how='left'
    ).drop(columns=['_non_prefix', 'prefecture_y'])\
    .rename(columns={'prefecture_x': 'prefecture'})

# %%

grp_capita_area_df.to_csv(
                     proc_dir + 'grp_capita.csv', 
                     index=False,
                     encoding='utf-8',
                     sep=','
                     )
# %%
