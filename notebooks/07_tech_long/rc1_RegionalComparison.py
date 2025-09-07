#! (root)/notebooks/05_calculation/2_Complexity.py python3
# -*- coding: utf-8 -*-

# %%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py
# Import Libraries
from ecomplexity import ecomplexity
from ecomplexity import proximity
# import openpyxl

# %%
# Import Original Modules
from calculation import method_of_reflections as mor

# %%
# Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = "../../data/interim/internal/filtered_after_agg/"
OUTPUT_DIR = "../../data/processed/internal/"
EX_DIR = "../../data/processed/external/"
# classification = 'ipc3'
output_dir = '../../data/processed/internal/'

## Check the condition
print(input_condition)
print(output_condition)


# %%
schmoch_df = pd.read_csv(
    f"{EX_DIR}schmoch/35.csv",
    encoding="utf-8",
    sep=",",
     usecols=['Field_number', 'Field_en']
).drop_duplicates()

reg_num_top_df = pd.read_csv(
    f"{DATA_DIR}app_nendo_1981_2010_5_all_p_3_right_person_name_fraction_schmoch35_fraction.csv", encoding="utf-8", sep=","
)
display(reg_num_top_df.head())
print(reg_num_top_df[region_corporation].nunique())

# %%
# pd.read_csv("../../data/interim/internal/filtered_before_agg/japan.csv").query(
#     "1981 <= app_nendo <= 2010"
# ).query(f"right_person_addr in {list(reg_num_top_df[region_corporation].unique())}")[
#     "reg_num"
# ].nunique()
#%%
reg_num_top_df.query('app_nendo_period == "1981-2010"')\
        .assign(
            _c_total=lambda x: x.groupby('schmoch35', observed=True)['reg_num'].transform("sum"),
            _p_total=lambda x: x.groupby('right_person_name', observed=True)['reg_num'].transform("sum"),
        )\
        .assign(
            rta=lambda x: (x['reg_num'] / x["_c_total"]) / (x["_p_total"] / x['reg_num'].sum()),
            class_q=lambda x: x.groupby('schmoch35')['reg_num'].transform(
                lambda s: (s.quantile(0.75)-s.quantile(0.25))*1.5
            )
        )\
        .assign(
            mpc=lambda x: np.where(
                (x["rta"] >= 1.0) | (x['reg_num'] >= x["class_q"]),
                1, 0
            ).astype(np.int64)
        )

#%%

# %%
classification = 'schmoch35'
trade_cols = {
    "time": f"{ar}_{year_style}_period",
    "loc": region_corporation,
    "prod": classification,
    "val": "mpc",
}
rename_col_dict = {"eci": "kci", "pci": "tci"}
col_order_list = [
    f"{ar}_{year_style}_period",
    region_corporation,
    classification,
    "reg_num",
    "rca",
    "mcp",
    "diversity",
    "ubiquity",
    "kci",
    "tci",
]

c_df = ecomplexity(reg_num_top_df, cols_input=trade_cols, rca_mcp_threshold=1,
                   presence_test="manual")
# prox_df = proximity(c_df, trade_cols)
# c_out_df = c_df.copy()
print(c_df.columns)

classification = 'schmoch35'
c_df = c_df[c_df["reg_num"] > 0].rename(columns=rename_col_dict)[col_order_list]
c_df = pd.concat(
    [
        mor.kh_ki(c_df[c_df[f"{ar}_{year_style}_period"] == period], 'right_person_name', classification)
        for period in c_df[f"{ar}_{year_style}_period"].unique()
    ],
    axis="index",
    ignore_index=True,
)

#%%
df_2010 = c_df.query('app_nendo_period == "1981-2010"')\
              .assign(
                  tech_share = lambda x: x['reg_num'] / x.groupby('schmoch35')['reg_num'].transform('sum'),
                  tech_share_sq = lambda x: ((x['reg_num'] / x.groupby('schmoch35')['reg_num'].transform('sum'))*100)**2,
                  hhi = lambda x: x.groupby('schmoch35')['tech_share_sq'].transform('sum'),
              )
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x='tci', y='hhi', data=df_2010, ax=ax)
ax.set_title(f'TCI vs HHI(corr= {df_2010["tci"].corr(df_2010["hhi"]).round(3)})')
# ax.plot([0,100],[0,100], color='red', linewidth=2, linestyle='--')
ax.set_xlabel('TCI')
ax.set_ylabel('HHI')
plt.show()

#%%
df_2010.groupby('schmoch35')['tech_share'].sum()

# %%
# 各期間
classification_df = pd.merge(
    c_df.groupby([f"{ar}_{year_style}_period", classification])[["reg_num"]]
    .sum()
    .reset_index(drop=False),
    c_df.groupby([f"{ar}_{year_style}_period", classification])[[region_corporation]]
    .nunique()
    .reset_index(drop=False),
    on=[f"{ar}_{year_style}_period", classification],
    how="inner",
)
classification_df = pd.merge(
    classification_df,
    c_df[
        [f"{ar}_{year_style}_period", classification, "ubiquity", "tci"]
        + [f"ki_{i}" for i in range(1, 20 + 1)]
    ].drop_duplicates(keep="first"),
    on=[f"{ar}_{year_style}_period", classification],
    how="inner",
)


# schmoch_df['ipc3'] = schmoch_df['IPC_code'].str[:3]
# classification_df = pd.merge(classification_df, schmoch_df.drop(columns=['IPC_code', 'Field_number']), 
#                              on='ipc3', 
#                              how='left')
# classification_df.to_csv(
#     f"{output_dir}05_2_4_tech/{output_condition}.csv",
#     encoding="utf-8",
#     sep=",",
#     index=False,
# )
#%%
data_dir = '../../data/interim/internal/filtered_before_agg/'
filter_dir = '../../data/interim/internal/filter_after_agg/'
output_dir = '../../data/interim/internal/filtered_after_agg/'
all_df = pd.read_csv(f'{data_dir}japan.csv', 
                     encoding='utf-8', 
                     sep=',', 
                     usecols=['right_person_name',
                               'app_nendo',
                               'right_person_addr'],
                     dtype={
                            'right_person_name': str,
                            'app_nendo': np.int64,
                            'right_person_addr': str
                     })\
            .query('1981 <= app_nendo <= 2010')\
            .query('right_person_addr in ("東京都", "大阪府", "愛知県")')
all_df

trade_cols = {
    "time": f"{ar}_{year_style}_period",
    "loc": region_corporation,
    "prod": 'ipc3',
    "val": "reg_num",
}
rename_col_dict = {"eci": "kci", "pci": "tci"}
col_order_list = [
    f"{ar}_{year_style}_period",
    region_corporation,
    'ipc3',
    "reg_num",
    "rca",
    "mcp",
    "diversity",
    "ubiquity",
    "kci",
    "tci",
]
# %%
all_df = all_df[all_df['right_person_name'].isin(c_df['right_person_name'].unique())]
tokyo_df = all_df.query('right_person_addr == "東京都"')
osaka_df = all_df.query('right_person_addr == "大阪府"')
aichi_df = all_df.query('right_person_addr == "愛知県"')

#%%
display(reg_num_top_df.head())
#%%
tokyo_df['right_person_name'].nunique(), osaka_df['right_person_name'].nunique()
#%%
tokyo_c_df = ecomplexity(reg_num_top_df[reg_num_top_df['right_person_name'].isin(tokyo_df['right_person_name'].unique())], 
                        cols_input=trade_cols, rca_mcp_threshold=1)\
                        .sort_values(by=['pci'], ascending=False)
osaka_c_df = ecomplexity(reg_num_top_df[reg_num_top_df['right_person_name'].isin(osaka_df['right_person_name'].unique())],
                        cols_input=trade_cols, rca_mcp_threshold=1)\
                        .sort_values(by=['pci'], ascending=False)
aichi_c_df = ecomplexity(reg_num_top_df[reg_num_top_df['right_person_name'].isin(aichi_df['right_person_name'].unique())],
                        cols_input=trade_cols, rca_mcp_threshold=1)\
                        .sort_values(by=['pci'], ascending=False)
#%%
display(tokyo_c_df.head())
# osaka_c_df= osaka_c_df[osaka_c_df['pci']!=osaka_c_df['pci'].min()]
# %%
import matplotlib.pyplot as plt
tokyo_osaka_df = pd.merge(tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
                          osaka_c_df[osaka_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
                          on='ipc3', how='inner', copy=False)

tokyo_osaka_df = pd.merge(tokyo_osaka_df,
                          aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
                          on='ipc3', how='inner', copy=False)\
                          .rename(columns={'pci_x': 'pci_x', 'pci_y': 'pci_y', 'pci': 'pci_z'})\
                          .query('ipc3 not in ("A63", "G07")')
display(tokyo_osaka_df)
# %%
aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')
# %%
all_vs_3pref_df = pd.merge(tokyo_osaka_df,
                           c_df[c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'tci']].drop_duplicates(keep='first').sort_values(by='tci', ascending=False),
                           on='ipc3', how='inner', copy=False)\
                         .assign(pci_x=lambda x: (x['pci_x']-x['pci_x'].min())/(x['pci_x'].max()-x['pci_x'].min())*100,
                                 pci_y=lambda x: (x['pci_y']-x['pci_y'].min())/(x['pci_y'].max()-x['pci_y'].min())*100,
                                 pci_z=lambda x: (x['pci_z']-x['pci_z'].min())/(x['pci_z'].max()-x['pci_z'].min())*100,
                                 tci=lambda x: (x['tci']-x['tci'].min())/(x['tci'].max()-x['tci'].min())*100)\
                         .drop_duplicates(keep='first')
display(c_df[c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'tci']].drop_duplicates(keep='first')\
       .sort_values(by='tci', ascending=False))
display(tokyo_osaka_df.sort_values(by='pci_x', ascending=False))

#%%
import matplotlib.pyplot as plt
from adjustText import adjust_text

plt.rcParams['font.size'] = 25
def plot_3pref(df, col, pref, ax, thresh=20):
    ax.scatter(df[col], df['tci'], color='black', s=60)
    ax.plot([0,100],[0,100], color='red', linewidth=2, linestyle='--')

    # ラベル付け対象
    mask = (df[col] - df['tci']).abs() >= thresh
    if col == 'pci_z':
        mask &= ~((df['pci_z'] >= 60) & (df['tci'] >= 40))

    # G09, G04, G11 を special として抽出
    special_codes = ['G09','G04','G11', 'C40', 'B42']
    special = df[mask & df['ipc3'].isin(special_codes)]
    general = df[mask & ~df['ipc3'].isin(special_codes)]

    # 一般点は adjustText に任せる
    texts = []
    for _, row in general.iterrows():
        texts.append(
            ax.text(row[col], row['tci'], row['ipc3'], fontsize=14)
        )
    adjust_text(texts, ax=ax,
                only_move={'points':'','text':'xy'},
                expand_points=(1.2,1.2),
                expand_text=(1.2,1.2),
                arrowprops=None)

    # G09, G04, G11 は手動微調整
    for _, row in special.iterrows():
        code = row['ipc3']
        if code == 'G09':
            dx, dy = -15, +5   # 左上へ
            ha, va = 'right','bottom'
        elif code == 'G04':
            dx, dy = +5, -15   # 右下へ
            ha, va = 'left','top'
        elif code == 'G11':
            dx, dy = -15, +5   # 左上へ
            ha, va = 'right','bottom'
        elif code == 'C40' and pref == 'Tokyo':
            dx, dy = -5, +5   # 左上へ
            ha, va = 'right','bottom'
        elif code == 'C40' and pref == 'Osaka':
            dx, dy = +5, -15   # 右下へ
            ha, va = 'left','bottom'
        elif code == 'B42':
            dx, dy = +5, -5   # 右下へ
            ha, va = 'left','top'
        else:
            dx, dy = 0, 0
            ha, va = 'center','center'
        ax.annotate(
            code,
            xy=(row[col], row['tci']),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=14,
            ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', lw=0.5, color='gray')
        )

    ax.set_xlabel(f'Corporate TCI ({pref})')
    ax.set_xticks(range(0,101,20))
    ax.set_yticks(range(0,101,20))
    ax.set_aspect('equal')

# plt.rcParams["figure.autolayout"]=False
fig, (ax_tk, ax_os, ax_ai) = plt.subplots(1, 3, 
                                          figsize=(15, 5), 
                                          constrained_layout=True,
                                          )
plot_3pref(all_vs_3pref_df, 'pci_x', 'Tokyo', ax_tk)
plot_3pref(all_vs_3pref_df, 'pci_y', 'Osaka', ax_os)
plot_3pref(all_vs_3pref_df, 'pci_z', 'Aichi', ax_ai)
# 図全体の左右中央に縦ラベル
fig.supylabel('Corporate TCI (All Prefectures)', fontsize=25)

# plt.tight_layout(h_pad=0.15)
plt.show()

#%%
import matplotlib.pyplot as plt
from adjustText import adjust_text

plt.rcParams['font.size'] = 25

def plot_3pref(df, col, pref, ax, thresh=20):
    # 軸入れ替え: x=全都府県TCI, y=府県別TCI
    ax.scatter(df['tci'], df[col], color='black', s=60)
    ax.plot([0,100], [0,100], color='red',
            linewidth=2, linestyle='--')

    # ラベル付け対象のマスク自体はそのまま
    mask = (df[col] - df['tci']).abs() >= thresh
    if col == 'pci_z':
        mask &= ~((df['pci_z'] >= 60) & (df['tci'] >= 40))

    special_codes = ['G09','G04','G11','C40','B42']
    special = df[mask & df['ipc3'].isin(special_codes)]
    general = df[mask & ~df['ipc3'].isin(special_codes)]

    # 一般点テキスト (x,y を逆に)
    texts = []
    for _, row in general.iterrows():
        texts.append(
            ax.text(row['tci'], row[col], row['ipc3'], fontsize=14)
        )
    adjust_text(texts, ax=ax,
                only_move={'points':'','text':'xy'},
                expand_points=(1.2,1.2),
                expand_text=(1.2,1.2),
                arrowprops=None)

    # special は annotate でも同様に (x,y) を逆に
    for _, row in special.iterrows():
        code = row['ipc3']
        if code == 'G09':
            dx, dy, ha, va = -15, +5, 'right','bottom'
        elif code == 'G04':
            dx, dy, ha, va = +5, -15, 'left','top'
        elif code == 'G11':
            dx, dy, ha, va = -15, +5, 'right','bottom'
        elif code == 'C40' and pref == 'Tokyo':
            dx, dy, ha, va = -5, -5, 'left','top'
        elif code == 'C40' and pref == 'Osaka':
            dx, dy, ha, va = +5, -15, 'left','bottom'
        elif code == 'B42':
            dx, dy, ha, va = +5, -5, 'left','top'
        else:
            dx, dy, ha, va = 0, 0, 'center','center'
        ax.annotate(
            code,
            xy=(row['tci'], row[col]),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=14,
            ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', lw=0.5, color='gray')
        )

    # 軸ラベルを入れ替え
    ax.set_ylabel(f'Corporate TCI ({pref})')
    ax.set_xticks(range(0,101,20))
    ax.set_yticks(range(0,101,20))
    ax.set_aspect('equal')


fig, (ax_tk, ax_os, ax_ai) = plt.subplots(1, 3,
                                          figsize=(15, 5),
                                          constrained_layout=True)

plot_3pref(all_vs_3pref_df, 'pci_x', 'Tokyo', ax_tk)
plot_3pref(all_vs_3pref_df, 'pci_y', 'Osaka', ax_os)
plot_3pref(all_vs_3pref_df, 'pci_z', 'Aichi', ax_ai)

# 図全体の縦ラベル→横ラベルに
fig.supxlabel('Corporate TCI (All Prefectures)', fontsize=25)

plt.show()



#%%
tokyo_c_df.query('app_nendo_period == "1981-2010"\
                  & mcp != 0', 
                 engine='python')\
          .drop_duplicates(subset=['right_person_name'], keep='first')\
          [['diversity']]\
            .sort_values(by='diversity', ascending=False)\
            .head(10)
            # .plot(kind='bar')

#%%
osaka_c_df.query('app_nendo_period == "1981-2010"\
                  & mcp != 0', 
                 engine='python')\
          .drop_duplicates(subset=['right_person_name'], keep='first')\
          [['diversity']].mean()
#%%
aichi_c_df.query('app_nendo_period == "1981-2010"\
                  & mcp != 0', 
                 engine='python')\
          .drop_duplicates(subset=['right_person_name'], keep='first')\
          [['diversity']].mean()

#%%
# display(
    c_df.query('app_nendo_period == "1981-2010"\
               & mcp != 0', engine='python')\
        .drop_duplicates(subset=['right_person_name'], keep='first')\
        .filter(items=['diversity']).sort_values(by='diversity', ascending=False)\
            .head(10)
        # .plot(kind='bar')
# )

#%%
display(
    mor.kh_ki(tokyo_c_df.query('app_nendo_period == "1981-2010"', 
                 engine='python'), 'ipc3', n=19)\
        .query('app_nendo_period == "1981-2010"\
               & mcp != 0', engine='python')\
        .drop_duplicates(subset=['ipc3'], keep='first')\
        .filter(items=['ubiquity']).mean()
)

#%%
display(
    c_df.query('app_nendo_period == "1981-2010"\
               & mcp != 0', engine='python')\
        .drop_duplicates(subset=['ipc3'], keep='first')\
        .filter(items=['ubiquity']).mean()
)
#%%
display(
    mor.kh_ki(tokyo_c_df.query('app_nendo_period == "1981-2010"', 
                 engine='python'), 'ipc3', n=19)\
        .query('app_nendo_period == "1981-2010"\
               & mcp != 0', engine='python')\
        .drop_duplicates(subset=['ipc3'], keep='first')\
        .filter(items=['ki_1']).mean()
)

display(
    c_df.query('app_nendo_period == "1981-2010"\
               & mcp != 0', engine='python')\
        .drop_duplicates(subset=['ipc3'], keep='first')\
        .filter(items=['ki_1']).mean()
)


#%%
all_df.query('right_person_addr == "愛知県"')['right_person_name'].nunique()
# %%
combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    # 3: ["reg_num_jp", "TCI_jp", "relation between the patent counts and the TCIs in Japan", "Patent Counts", "TCIs", "center left", ],
    # 4: ["TCI_jp", "reg_num_jp", "relation between the patent counts and the TCIs in Japan", "TCIs", "Patent Counts", "center left", ],
    # 6: ["TCI_jp", "ubiquity", "relation between the ubiquity and the TCIs in Japan", "TCIs", "Ubiquity", "center left", ],
    # 7: ["ubiquity", "tci", "", "Degree Centrality $K_{T, 0}$", "TCI", "center left", ],
    7: ["pci_x", "pci", "", "TOKYO", "OSAKA", "center left", ],
    8: ["pci_y", "pci", "", "TOKYO", "ALL PREFECTURE", "center left", ], 
}
tokyo_aichi_df = pd.merge(
    aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
    tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
    on=['ipc3'], 
    how='inner'
).rename(
    columns={'pci_x':'aichi', 'pci_y':'tokyo'}
).query('aichi not in aichi.nsmallest(2)', engine='python')\
.assign(
    aichi = lambda df: ((df['aichi']-df['aichi'].min())/(df['aichi'].max()-df['aichi'].min()))*100,
    tokyo = lambda df: ((df['tokyo']-df['tokyo'].min())/(df['tokyo'].max()-df['tokyo'].min()))*100,
)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(
    tokyo_aichi_df['aichi'], 
    tokyo_aichi_df['tokyo'], 
    c='black'
)
ax.set_aspect('equal')
ax.set_xticks(range(0, 100+1, 20))
ax.set_xticklabels(range(0,100+1, 20))
ax.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')
ax.set_xlabel('Corporate TCI in Aichi')
ax.set_ylabel('Corporate TCI in Tokyo')
plt.show()
# display(tokyo_aichi_df)
# display(aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')['pci'])
# display(tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')['pci'])
#%%
all_vs_3pref_df.to_clipboard()

#%%
import matplotlib.pyplot as plt
from adjustText import adjust_text

plt.rcParams['font.size'] = 25

def plot_pair(df, xcol, ycol, xpref, ypref, ax, thresh=20):
    """
    x軸に xcol（例: pci_y=Osaka）、y軸に ycol（例: pci_x=Tokyo）をとる散布図を描画。
    右上がりの y=x の基準線、差分がthresh以上の点にラベルを付与。
    """
    # 散布 & y=x
    ax.scatter(df[xcol], df[ycol], color='black', s=60)
    ax.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')

    # ラベリング対象（xとyの差で判定）
    mask = (df[ycol] - df[xcol]).abs() >= thresh

    # 以前の特例（pci_z の高値領域での密集回避）をペア用に調整
    if ycol == 'pci_z':
        mask &= ~((df['pci_z'] >= 60) & (df[xcol] >= 40))

    special_codes = ['G09', 'G04', 'G11', 'C40', 'B42']
    special = df[mask & df['ipc3'].isin(special_codes)]
    general = df[mask & ~df['ipc3'].isin(special_codes)]

    # 一般コードのテキスト
    texts = []
    for _, row in general.iterrows():
        texts.append(
            ax.text(row[xcol], row[ycol], row['ipc3'], fontsize=14)
        )
    adjust_text(
        texts, ax=ax,
        only_move={'points': '', 'text': 'xy'},
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        arrowprops=None
    )

    # special の注記（オフセットは従来のルールを y側pref に合わせて踏襲）
    for _, row in special.iterrows():
        code = row['ipc3']
        if code == 'G09':
            dx, dy, ha, va = -15, +5, 'right', 'bottom'
        elif code == 'G04':
            dx, dy, ha, va = +5, -15, 'left', 'top'
        elif code == 'G11':
            dx, dy, ha, va = -15, +5, 'right', 'bottom'
        elif code == 'C40' and ypref == 'Tokyo':
            dx, dy, ha, va = -5, +5, 'right', 'bottom'
        elif code == 'C40' and ypref == 'Osaka':
            dx, dy, ha, va = +5, -15, 'left', 'bottom'
        elif code == 'B42':
            dx, dy, ha, va = +5, -5, 'left', 'top'
        else:
            dx, dy, ha, va = 0, 0, 'center', 'center'

        ax.annotate(
            code,
            xy=(row[xcol], row[ycol]),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=14,
            ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', lw=0.5, color='gray')
        )

    # 軸ラベル
    ax.set_xlabel(f'Corporate TCI ({xpref})')
    ax.set_ylabel(f'Corporate TCI ({ypref})')

    # 体裁
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_aspect('equal')


# ── 3面プロット（左・中・右の順で作る）
fig, (ax_left, ax_mid, ax_right) = plt.subplots(
    1, 3, figsize=(15, 5), constrained_layout=True
)

# 左：pci_z vs. pci_x  => x=Tokyo(pci_x), y=Aichi(pci_z)
plot_pair(all_vs_3pref_df, 'pci_x', 'pci_z', 'Tokyo', 'Aichi', ax_left)

# 中：pci_y vs. pci_z  => x=Aichi(pci_z), y=Osaka(pci_y)
plot_pair(all_vs_3pref_df, 'pci_z', 'pci_y', 'Aichi', 'Osaka', ax_mid)

# 右：pci_x vs. pci_y  => x=Osaka(pci_y), y=Tokyo(pci_x)
plot_pair(all_vs_3pref_df, 'pci_y', 'pci_x', 'Osaka', 'Tokyo', ax_right)

plt.show()


#%%
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.size'] = 25

def plot_pair(df: pd.DataFrame,
              xcol: str, ycol: str,
              xpref: str, ypref: str,
              ax: plt.Axes,
              thresh: int = 20,
              special_offsets: dict | None = None) -> None:
    """
    x軸=xcol, y軸=ycol の散布図。
    注記対象:
      - |y - x| >= thresh
      - 両軸とも60以上ではない
      - （Aichiの軸なら）Aichiの値が80未満
    """
    ax.scatter(df[xcol], df[ycol], color='black', s=60)
    ax.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')

    # 基本の注記マスク
    mask = (df[ycol] - df[xcol]).abs() >= thresh
    mask &= ~((df[xcol] >= 60) & (df[ycol] >= 60))

    # Aichi（pci_z）が軸にある場合は、Aichiが80以上を除外
    if xcol == 'pci_z':
        mask &= df['pci_z'] < 80
    if ycol == 'pci_z':
        mask &= df['pci_z'] < 80

    # 注記（基本は右下）
    for _, row in df[mask].iterrows():
        code = row['ipc3']
        dx, dy, ha, va = +6, -6, 'left', 'top'  # 右下

        # 個別指定があれば上書き
        if special_offsets and code in special_offsets:
            o = special_offsets[code]
            dx = o.get('dx', dx); dy = o.get('dy', dy)
            ha = o.get('ha', ha); va = o.get('va', va)

        ax.annotate(
            code,
            xy=(row[xcol], row[ycol]),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=14,
            ha=ha, va=va,
            arrowprops=None,
            clip_on=True
        )

    ax.set_xlabel(f'Corporate TCI ({xpref})')
    ax.set_ylabel(f'Corporate TCI ({ypref})')
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_aspect('equal')


# ── 3面プロット
fig, (ax_left, ax_mid, ax_right) = plt.subplots(
    1, 3, figsize=(15, 5), constrained_layout=True
)

# 左：Osaka vs. Tokyo  => x=Tokyo(pci_x), y=Osaka(pci_y)
# C40 を右真横に固定
left_special = {
    "C40": {"dx": 8, "dy": 0, "ha": "left", "va": "center"}
}
plot_pair(all_vs_3pref_df, 'pci_x', 'pci_y', 'Tokyo', 'Osaka',
          ax_left, thresh=20, special_offsets=left_special)

# 中：Aichi vs. Osaka => x=Osaka(pci_y), y=Aichi(pci_z)
plot_pair(all_vs_3pref_df, 'pci_y', 'pci_z', 'Osaka', 'Aichi',
          ax_mid, thresh=20)

# 右：Tokyo vs. Aichi => x=Aichi(pci_z), y=Tokyo(pci_x)
plot_pair(all_vs_3pref_df, 'pci_z', 'pci_x', 'Aichi', 'Tokyo',
          ax_right, thresh=20)

plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List

plt.rcParams['font.size'] = 25

def _decrowd(df: pd.DataFrame, xcol: str, ycol: str,
             max_labels: int = 18, cell: float = 6.0, per_bin: int = 2) -> pd.DataFrame:
    """密集回避：グリッド間引き + x軸10刻みビンごとの上位抽出 + 総数上限。"""
    d = df.copy()
    d["_diff"] = (d[ycol] - d[xcol]).abs()

    # 1) グリッド間引き（セル内で |y-x| 最大だけ残す）
    d["_gx"] = (d[xcol] // cell).astype(int)
    d["_gy"] = (d[ycol] // cell).astype(int)
    d = d.sort_values("_diff", ascending=False).drop_duplicates(["_gx", "_gy"])

    # 2) x軸10刻みビンで各ビンの上位 per_bin 件だけ
    d["_bin"] = (d[xcol] // 10).astype(int)
    d = d.sort_values("_diff", ascending=False).groupby("_bin", as_index=False).head(per_bin)

    # 3) 総数を上限でカット
    d = d.nlargest(max_labels, "_diff")

    return d.drop(columns=["_diff", "_gx", "_gy", "_bin"])


def plot_pair(df: pd.DataFrame,
              xcol: str, ycol: str,
              xpref: str, ypref: str,
              ax: plt.Axes,
              thresh: int = 20,
              decrowd: bool = True,
              max_labels: int = 18,
              cell: float = 6.0,
              per_bin: int = 2,
              suppress_codes: Optional[List[str]] = None,
              special_offsets: Optional[Dict[str, Dict]] = None) -> None:
    """
    x軸=xcol, y軸=ycol の散布図を描画し、注記は以下の条件を満たす点のみ：
      - |y - x| >= thresh
      - (x >= 60) & (y >= 60) ではない
      - Aichi 軸が含まれる場合は pci_z < 80
    その後、密集回避（グリッド間引き＋ビン上位＋上限）を適用。
    """
    ax.scatter(df[xcol], df[ycol], color='black', s=60)
    ax.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')

    mask = (df[ycol] - df[xcol]).abs() >= thresh
    mask &= ~((df[xcol] >= 60) & (df[ycol] >= 60))

    # Aichi（pci_z）が軸にある場合は Aichi>=80 を除外
    if xcol == 'pci_z':
        mask &= df['pci_z'] < 80
    if ycol == 'pci_z':
        mask &= df['pci_z'] < 80

    cand = df[mask].copy()

    # コードの強制除外
    if suppress_codes:
        cand = cand[~cand['ipc3'].isin(suppress_codes)]

    # 密集回避
    if decrowd:
        cand = _decrowd(cand, xcol, ycol, max_labels=max_labels, cell=cell, per_bin=per_bin)

    # 注記（基本は右下）
    for _, row in cand.iterrows():
        code = row['ipc3']
        dx, dy, ha, va = +6, -6, 'left', 'top'  # 右下
        if special_offsets and code in special_offsets:
            o = special_offsets[code]
            dx = o.get('dx', dx); dy = o.get('dy', dy)
            ha = o.get('ha', ha); va = o.get('va', va)

        ax.annotate(
            code,
            xy=(row[xcol], row[ycol]),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=14,
            ha=ha, va=va,
            arrowprops=None,
            clip_on=True
        )

    ax.set_xlabel(f'Corporate TCI ({xpref})')
    ax.set_ylabel(f'Corporate TCI ({ypref})')
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_aspect('equal')


# ── 3面プロット
fig, (ax_left, ax_mid, ax_right) = plt.subplots(
    1, 3, figsize=(15, 5), constrained_layout=True
)

# 左：Osaka vs. Tokyo  => x=Tokyo(pci_x), y=Osaka(pci_y)
left_special = {
    "C40": {"dx": 8, "dy": 0, "ha": "left", "va": "center"}  # 右真横に固定
}
plot_pair(
    all_vs_3pref_df, 'pci_x', 'pci_y', 'Tokyo', 'Osaka', ax_left,
    thresh=20, decrowd=True, max_labels=16, cell=6.0, per_bin=2,
    suppress_codes=['F41'],  # ← F41 を除外
    special_offsets=left_special
)

# 中：Aichi vs. Osaka => x=Osaka(pci_y), y=Aichi(pci_z)
plot_pair(
    all_vs_3pref_df, 'pci_y', 'pci_z', 'Osaka', 'Aichi', ax_mid,
    thresh=20, decrowd=True, max_labels=14, cell=6.0, per_bin=2
)

# 右：Tokyo vs. Aichi => x=Aichi(pci_z), y=Tokyo(pci_x)
plot_pair(
    all_vs_3pref_df, 'pci_z', 'pci_x', 'Aichi', 'Tokyo', ax_right,
    thresh=20, decrowd=True, max_labels=14, cell=6.0, per_bin=2
)

plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

plt.rcParams['font.size'] = 25

def plot_pair(df, xcol, ycol, xpref, ypref, ax,
              thresh=20, highhigh_cutoff=50,
              skip_ipcs=None, aichi_hide_ge=85,
              special_offsets=None):
    skip_ipcs = set(skip_ipcs or [])
    special_offsets = special_offsets or {}

    # 散布（戻り値を保持して adjust_text に渡す）
    sc = ax.scatter(df[xcol], df[ycol], color='black', s=60)

    # y=x
    ax.plot([0,100],[0,100], color='red', linewidth=2, linestyle='--')

    # 注記対象
    mask = (df[ycol] - df[xcol]).abs() >= thresh
    mask &= ~((df[xcol] >= highhigh_cutoff) & (df[ycol] >= highhigh_cutoff))
    if aichi_hide_ge is not None:
        if xpref == 'Aichi':
            mask &= (df[xcol] < aichi_hide_ge)
        if ypref == 'Aichi':
            mask &= (df[ycol] < aichi_hide_ge)

    cand = df[mask & ~df['ipc3'].isin(skip_ipcs)].copy()

    # special / general
    special = cand[cand['ipc3'].isin(special_offsets.keys())]
    general = cand[~cand['ipc3'].isin(special_offsets.keys())]

    texts = []
    # ---- general：Annotationで作る（矢印は常に xy=点 を指す）
    for _, row in general.iterrows():
        x, y, code = row[xcol], row[ycol], row['ipc3']
        # 初期オフセットを対角線の上下で変える
        if y >= x:
            dx, dy, ha, va = -6, +6, 'right', 'bottom'
        else:
            dx, dy, ha, va = +6, -6, 'left', 'top'
        ann = ax.annotate(
            code, xy=(x, y), xytext=(dx, dy),
            textcoords='offset points', fontsize=14, ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'),
            zorder=3
        )
        texts.append(ann)
    pad = 3  # 2〜5 くらいで調整
    ax.set_xlim(-pad, 100 + pad)
    ax.set_ylim(-pad, 100 + pad)
    # ラベルだけ動かす（points は動かさない）
    adjust_text(
        texts, ax=ax, add_objects=[sc],
        only_move={'points': '', 'text': 'xy'},  # テキスト(x,y)移動のみ
        expand_points=(1.2, 1.2), expand_text=(1.2, 1.2),
        force_text=0.8
    )

    # ---- special：個別オフセットを固定で指定
    for _, row in special.iterrows():
        code = row['ipc3']; x, y = row[xcol], row[ycol]
        dx, dy, ha, va = special_offsets[code]
        ax.annotate(
            code, xy=(x, y), xytext=(dx, dy),
            textcoords='offset points', fontsize=14, ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'),
            zorder=3
        )

    ax.set_xlabel(f'Corporate TCI ({xpref})')
    ax.set_ylabel(f'Corporate TCI ({ypref})')
    ax.set_xticks(range(0,101,20))
    ax.set_yticks(range(0,101,20))
    ax.set_aspect('equal')


# ── 3面
fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

# 左：Osaka vs Tokyo（F41除外）
plot_pair(all_vs_3pref_df, 'pci_x', 'pci_y', 'Tokyo', 'Osaka', ax_left,
          skip_ipcs={'F41'})

# 中：Aichi vs Osaka（Aichi>=85非表示）
plot_pair(all_vs_3pref_df, 'pci_y', 'pci_z', 'Osaka', 'Aichi', ax_mid,
          aichi_hide_ge=85)

# 右：Tokyo vs Aichi（Aichi>=85非表示）
plot_pair(all_vs_3pref_df, 'pci_z', 'pci_x', 'Aichi', 'Tokyo', ax_right,
          aichi_hide_ge=85)

plt.show()

# %%
