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
    f"{DATA_DIR}app_nendo_1981_2010_5_all_p_3_right_person_name_fraction_ipc3_fraction.csv", encoding="utf-8", sep=","
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
result_df = reg_num_top_df.query('app_nendo_period == "1981-2010"')\
        .assign(
            _c_total=lambda x: x.groupby('ipc3', observed=True)['reg_num'].transform("sum"),
            # _c_total=lambda x: x.groupby('schmoch35', observed=True)['reg_num'].transform("sum"),
            _p_total=lambda x: x.groupby('right_person_name', observed=True)['reg_num'].transform("sum"),
        )\
        .assign(
            rta=lambda x: (x['reg_num'] / x["_c_total"]) / (x["_p_total"] / x['reg_num'].sum()),
            class_q=lambda x: x.groupby('ipc3')['reg_num'].transform(
                lambda s: (s.quantile(0.75)-s.quantile(0.25))*1.5
            )
        )\
        .assign(
            mpc=lambda x: np.where(
                # (x["rta"] >= 1.0) | (x['reg_num'] >= x["class_q"]),
                (x["rta"] >= 1.0),
                1, 0
            ).astype(np.int64)
        )

#%%

# %%
classification = 'ipc3'
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
    "rta",
    "mcp",
    "diversity",
    "ubiquity",
    "kci",
    "tci",
]

c_df = ecomplexity(result_df, cols_input=trade_cols, rca_mcp_threshold=1,
                   presence_test="manual")
# prox_df = proximity(c_df, trade_cols)
# c_out_df = c_df.copy()
print(c_df.columns)

# classification = 'schmoch35'
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
# df_2010 = c_df.query('app_nendo_period == "1981-2010"')\
#               .assign(
#                   tech_share = lambda x: x['reg_num'] / x.groupby('schmoch35')['reg_num'].transform('sum'),
#                   tech_share_sq = lambda x: ((x['reg_num'] / x.groupby('schmoch35')['reg_num'].transform('sum'))*100)**2,
#                   hhi = lambda x: x.groupby('schmoch35')['tech_share_sq'].transform('sum'),
#               )
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.scatter(x='tci', y='hhi', data=df_2010, 
#            color='tab:red',
#            marker='o',
#            s=15,
#            )
# ax.set_title(f'TCI vs HHI(corr= {df_2010["tci"].corr(df_2010["hhi"]).round(3)})')
# # ax.plot([0,100],[0,100], color='red', linewidth=2, linestyle='--')
# ax.set_xlabel('TCI')
# ax.set_ylabel('HHI')
# plt.show()

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
# 位置プリセット
import re

# 位置プリセット（右下=rb は ha='left', va='top'）
_POS = {
    "rb": dict(dx=+5,  dy=-5, ha="left",  va="top"),      # 右下（既定）
    "rt": dict(dx=+5,  dy=+5, ha="left",  va="bottom"),   # 右上
    "lb": dict(dx=-5,  dy=-5, ha="right", va="top"),      # 左下
    "lt": dict(dx=-5,  dy=+5, ha="right", va="bottom"),   # 左上
    "r":  dict(dx=+6,  dy=0,  ha="left",  va="center"),   # 右
    "l":  dict(dx=-6,  dy=0,  ha="right", va="center"),   # 左
    "t":  dict(dx=0,   dy=+6, ha="center",va="bottom"),   # 上
    "b":  dict(dx=0,   dy=-6, ha="center",va="top"),      # 下
}

def _norm_code(s: str) -> str:
    """'H03', 'H03-*', ' h03 ' などを 'H03' に正規化"""
    if not isinstance(s, str):
        s = str(s)
    m = re.match(r'\s*([A-Z])\s*([0-9]{2})', s.strip().upper())
    return f"{m.group(1)}{m.group(2)}" if m else s.strip().upper()[:3]

def plot_pref_pair(df, col_x, col_y, name_x, name_y, ax,
                   mode="auto", max_x=40, max_y=80,
                   exceptions=None,  # 例: {'H03':'lt', 'G04':'lt', 'G03':'lb'}
                   respect_mask_for_exceptions=True):
    """
    mode='auto'  : (x<=max_x & y<=max_y) のみ注記
    mode='fixed' : 特定コードのみ注記（Tokyo-Osaka用）
    exceptions   : {'CODE3':'rb|lt|lb|r|...'} で注記位置を個別指定
    respect_mask_for_exceptions : Trueなら例外もマスクを適用
    """
    # 例外マップを正規化キーに
    ex_map = {}
    if exceptions:
        ex_map = { _norm_code(k): v for k, v in exceptions.items() }

    # 散布と45度線
    ax.scatter(df[col_x], df[col_y], color='black', s=60)
    ax.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')

    # 注記対象選定
    if mode == "fixed":
        label_df = df[df['ipc3'].apply(_norm_code).isin({'F41','C40','A24'})].copy()
    else:
        base_mask = (df[col_x] <= max_x) & (df[col_y] <= max_y)
        if col_x == 'pci_y': base_mask = base_mask | ((df[col_x] <= 90)&(df[col_y] <= 60))
        if exceptions and not respect_mask_for_exceptions:
            # 例外コードはマスク外でも拾う
            is_exc = df['ipc3'].apply(_norm_code).isin(set(ex_map.keys()))
            base_mask = base_mask | is_exc
        label_df = df[base_mask].copy()

    # 注記描画（既定は右下=rb）
    for _, row in label_df.iterrows():
        code_raw = row['ipc3']
        code = _norm_code(code_raw)
        pos_key = ex_map.get(code, 'rb')
        pos = _POS[pos_key]
        ax.annotate(
            code, (row[col_x], row[col_y]),
            xytext=(pos["dx"], pos["dy"]), textcoords="offset points",
            fontsize=12, ha=pos["ha"], va=pos["va"],
            arrowprops=dict(arrowstyle='->', lw=1, color='gray')
        )

    # 軸
    ax.set_xlabel(f'Corporate TCI ({name_x})')
    ax.set_ylabel(f'Corporate TCI ({name_y})')
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_aspect('equal')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# 1) Tokyo-Osaka：F41 と C40 のみ。位置は既定=右下（ha='left', va='top'）
plot_pref_pair(all_vs_3pref_df, 'pci_x', 'pci_y', 'Tokyo', 'Osaka',
               ax1, mode="fixed", exceptions={'F41':'lb', 'C40':'rt', 'A24':'lb'})

# 2) Tokyo-Aichi：x<=40 & y<=80。H03/G04=左上、G03=左下。他は右下
plot_pref_pair(all_vs_3pref_df, 'pci_x', 'pci_z', 'Tokyo', 'Aichi',
               ax2, mode="auto", max_x=70, max_y=80,
               exceptions={'H03':'lt', 'G04':'lt', 'G03':'lb', 'G08':'lt','A44':'lb'})

# 3) Osaka-Aichi：x<=40 & y<=80。F21のみ右、それ以外は右下
plot_pref_pair(all_vs_3pref_df, 'pci_y', 'pci_z', 'Osaka', 'Aichi',
               ax3, mode="auto", max_x=70, max_y=80,
               exceptions={'F21':'r', 'H03':'rt', 'A45':'rt'})

plt.show()


# %%
