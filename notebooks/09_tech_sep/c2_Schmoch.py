#! (root)/notebooks/08_producer_sep/r1_Prefecture.py python3
# -*- coding: utf-8 -*-

#%%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

#%%
print(input_condition)

# %%

global data_dir, ex_dir, output_dir
data_dir = "../../data/processed/internal/tech/"
ex_dir = "../../data/processed/external/schmoch/"
output_dir = "../../output/figures/"


# %%
df = pd.read_csv(
                    f'{DATA_DIR}{input_condition}.csv', 
                    encoding='utf-8',
                    sep=','
                    )

schmoch_df = pd.read_csv(
                         f'{EX_DIR}schmoch/35.csv', 
                         encoding='utf-8', 
                         sep=',', 
                         )\
                         .drop_duplicates()


pd.merge(df, schmoch_df, 
              left_on=classification, right_on='Field_en', 
              how='left'
              )\
            .drop(columns=['Field_number', classification])\
            .rename(columns={'Field_en': classification})\
            .sort_values(f'{ar}_{year_style}_period', key=lambda col: col.map(period_order_dict))\
            .query(f'{ar}_{year_style}_period == "{year_start}-{year_end}"')\
            .assign(
            tci = lambda x: minmax_scale(x['tci']) * 100,
            )

# display(df)


#%%
k_trans_df = df[(df[f'{ar}_{year_style}_period']==f'{year_start}-{year_end}')].sort_values('tci', ascending=False).copy()
k_trans_df_dict = {'0': k_trans_df.melt(id_vars=['schmoch35'], value_vars=[f'ubiquity']).sort_values('value', ascending=False)}
fs = (32, 40)
for i in range(2, 20+1, 2):
    k_trans_df_dict[f'{i}'] = k_trans_df.melt(id_vars=['schmoch35'], value_vars=[f'ki_{i}']).sort_values('value', ascending=False)
    # if i >= 12: display(sample_df_dict[f'{i}'].head(10))
k_trans = vr.rank_doubleaxis(k_trans_df_dict, 
                rank_num=35,
                member_col='schmoch35', 
                value_col='value',
                prop_dict={
                    'figsize': fs,
                    'xlabel': 'N',
                    'ylabel': '',
                    'title': '',
                    'fontsize': 24, 
                    'year_range': 2, 
                    'ascending': False, 
                    # 'color': color_dict
                    'color': 'default'
                })

# plt.savefig(f'{output_dir}co_ranking/ktrans_{fig_name_base}', bbox_inches="tight")
plt.show()

#%%
df_dict = {}
combi_dict = {# 1: ['right_person_name', 'diversity', f'特許権者次数(=diversity，k_h0)ランキング 値が小さいものTop15の推移（出願期間：{year_start}-{year_end}年度）', True], 
              # 1: ['right_person_name', '', f'特許権者次数(=diversity，k_h0)ランキング 値が小さいものTop15の推移（出願期間：{year_start}-{year_end}年度）', True], 
            #   2: ['right_person_name', 'diversity', f'特許権者次数(=diversity，k_h0)ランキング 値が大きいものTop15の推移（出願期間：{year_start}-{year_end}年度）', False], 
            #   3: ['right_person_name', 'eci', f'KCIランキング 値が小さいものTop15の推移（出願期間：{year_start}-{year_end}年度）', True], 
              # 3: ['right_person_name', 'ipc_class_num', f'IPC数ランキング 値が大きいものTop15の推移（出願期間：{year_start}-{year_end}年度）', False], 
            #   4: ['right_person_name', 'kci', f'KCIランキング 値が大きいものTop15の推移（出願期間：{year_start}-{year_end}年度）', False], 
              4: ['schmoch35', 'tci', '', False]
              }



for i, combi in combi_dict.items():
    div_df = df[[f'{ar}_{year_style}_period']+combi[:1+1]].sort_values(by=[f'{ar}_{year_style}_period', combi[1]], ascending=[True, False]).copy()
    div_df = div_df.drop_duplicates(keep='first')
    div_df_dict = {}
    for year in range(year_start, year_end+1, year_range):
        period = f'{year}-{year+year_range-1}'
        div_df_dict[period] = div_df[div_df[f'{ar}_{year_style}_period']==period].copy()
        # display(div_df_dict[f'{year}-{year+year_range-1}'].head(15))
    if i==4:
        df['tci_rank'] = df.groupby(f'{ar}_{year_style}_period')[['tci']].rank(ascending=False, method='first').reset_index(drop=False)['tci']
        # display(c_df[c_df['right_person_name'].str.contains('三菱重工業')])
    if i > 4: fs = (12, 15)
    else: fs = (24, 24)
    # display(div_df)
    sample = vr.rank_doubleaxis(div_df_dict, 
                    rank_num=35,
                    member_col=combi[0], 
                    value_col=combi[1],
                    prop_dict={
                        'figsize': fs,
                        'xlabel': '期間',
                        'ylabel': '',
                        'title': combi[2],
                        'fontsize': 30, 
                        'year_range': year_range, 
                        'ascending': combi[3], 
                        # 'color': color_dict
                        'color': 'default'
                    })
    plt.xticks(rotation=45)
    # plt.savefig(f'{output_dir}co_ranking/{combi[1]}_{fig_name_base}', bbox_inches="tight")
    plt.show()


#%%
fiveyears_df_dict = {
    f"{year}": classification_df[
        classification_df[f"{ar}_{year_style}_period"] == f"{year}"
    ][[f"{ar}_{year_style}_period", classification, "tci"]].drop_duplicates(
        keep="first"
    )
    for year in classification_df[f"{ar}_{year_style}_period"].unique()
    if year != f"{year_start}-{year_end}"
}

rank.rank_doubleaxis(
    fiveyears_df_dict,
    rank_num=124,
    member_col=classification,
    value_col="tci",
    prop_dict={
        "figsize": (16, 10),
        "xlabel": "Period",
        "ylabel": "Technological Fields",
        "title": "",
        "fontsize": 15,
        "year_range": 15,
        "ascending": False,
        "color": "default",
    },
)

classification_df.to_csv(
    f"{output_dir}tech/{output_condition}.csv", encoding="utf-8", sep=",", index=False
)
# classification_df[classification_df[f'{ar}_{year_style}_period']==f'{year_start}-{year_end}']\
#     [['schmoch35', 'reg_num', 'ubiquity', 'tci']]\
#     .rename(columns={'reg_num':'patent count', 'ubiquity':'degree centrality', 'tci':'TCI'})\
#     .to_excel('../../output/tables/TCI.xlsx',
#                          index=False,
#                          sheet_name=output_condition)