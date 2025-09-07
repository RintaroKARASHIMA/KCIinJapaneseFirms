#! (root)/notebooks/10_pref_long/1_tmp.py python3

#%%
import sys
sys.path.append("../../src")
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Yu Mincho'
plt.rcParams['font.size'] = 15

#%%
raw_df = pd.read_csv(
    '../../data/interim/internal/filtered_before_agg/japan.csv',
    encoding='utf-8',
    sep=',',
    dtype={
        'ipc': object,
        'reg_num': object,
        'app_year': np.int64,
        'app_nendo': np.int64,
        'reg_year': np.int64,
        'reg_nendo': np.int64,
        'right_person_addr': str,
        'schmoch35': np.int64,
    }
    )\
    .assign(
        ipc3 = lambda x: x['ipc'].str[:3],
        ipc4 = lambda x: x['ipc'].str[:4],
    )\
    .drop(
        columns=['ipc','right_person_name']
    )\
    .drop_duplicates(
        keep='first'
    )
display(raw_df)
schmoch35_df = pd.read_csv(
    '../../data/processed/external/schmoch/35.csv',
    encoding='utf-8',
    sep=',',
    ).filter(items=['Field_number', 'Field_en'])\
    .drop_duplicates(
        keep='first'
    )
 
display(schmoch35_df)


#%%
fig, ax = plt.subplots(figsize=(10, 5))
app_reg_year_df = pd.concat(
    [
        raw_df.groupby('app_year')[['reg_num']].nunique()\
              .rename(columns={'reg_num': 'app_year'}),
        raw_df.groupby('reg_year')[['reg_num']].nunique()\
              .rename(columns={'reg_num': 'reg_year'})
    ],
    axis='columns'
    )\
    .plot(
        kind='bar', 
        ax=ax,
        legend=True, 
        color=['orange', 'green'], 
        fontsize=15, 
        width=0.8
    )
ax.set_xlabel('Year')
ax.set_ylabel('Number of Patents')
ax.set_xticks(
    range(0, raw_df['app_year'].nunique() + 1, 5)
)
ax.set_xticklabels(
    range(raw_df['app_year'].min(), raw_df['app_year'].max() + 1, 5),
    rotation=45
)
plt.show()


# %%


fig, ax = plt.subplots(figsize=(10, 5))
app_reg_year_df = pd.concat(
    [
        raw_df.groupby('app_year')[['reg_num']].nunique()\
              .rename(columns={'reg_num': 'app_year'}),
        raw_df.groupby('app_nendo')[['reg_num']].nunique()\
              .rename(columns={'reg_num': 'app_nendo'})
    ],
    axis='columns'
    )\
    .plot(
        kind='bar', 
        ax=ax,
        legend=True, 
        color=['orange', 'tab:blue'], 
        fontsize=15, 
        width=0.8
    )
ax.set_xlabel('Year')
ax.set_ylabel('Number of Patents')
ax.set_xticks(
    range(0, raw_df['app_year'].nunique() + 1, 5)
)
ax.set_xticklabels(
    range(raw_df['app_year'].min(), raw_df['app_year'].max() + 1, 5),
    rotation=45
)
plt.show()
# %%
df = raw_df.query('1981 <= app_nendo <= 2015', engine='python')\
           .drop(columns=['app_year', 'reg_year', 'reg_nendo'])\
           .drop_duplicates(keep='first')\
           .reset_index(drop=True)

#%%
per_plot_dict = {
    'schmoch35': 'tab:orange',
    'ipc3': 'tab:blue',
    'ipc4': 'tab:green', 
}

for key, value in per_plot_dict.items():
    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby(by=['right_person_addr', key], as_index=False)[['reg_num']].nunique()\
    .sort_values(by='reg_num', ascending=False)\
    .plot(
        kind='scatter',
        x='right_person_addr',
        y='reg_num',
        alpha=0.5,
        s=10,
        ax=ax, 
        color=value
    )
    ax.set_yscale('log')
    ax.set_title(f'Number of Patents in each technology per Prefecture({key})')
    ax.set_xlabel('Prefectures')
    ax.set_ylabel('Number of Patents in each technology')
    ax.set_xticks(
        range(0, df['right_person_addr'].nunique())
    )
    ax.set_xticklabels(
        list(df['right_person_addr'].unique()),
        rotation=90, 
        fontsize=9
    )
    ax.grid(
        axis='y', linestyle='--', alpha=0.4
        )

# %%
long_df = (
    df.drop(
        columns=['ipc3', 'ipc4']
        )\
        .drop_duplicates(keep='first')\
        .assign(
            app_nendo_period = '1981-2015'
        )\
        .drop(columns=['app_nendo'])\
        .assign(
            addr_count=lambda x: x.groupby('reg_num')['right_person_addr'].transform('nunique'),
            class_count=lambda x: x.groupby('reg_num')['schmoch35'].transform('nunique')
        )\
        .assign(
            weight=lambda x: 1 / (x['addr_count'] * x['class_count'])
        )\
        .groupby(['app_nendo_period', 'right_person_addr', 'schmoch35'], as_index=False)\
        .agg(
            patent_count=('weight', 'sum')
        )
)
long_df

#%%
import numpy as np
import pandas as pd
from typing import Literal
from typing import Literal
import numpy as np
import pandas as pd

def compute_pref_schmoch_lq(
    df: pd.DataFrame,
    aggregate: Literal[True, False] = True,
    *,
    prefecture_col: str = "right_person_addr",
    class_col: str = "schmoch35",
    count_col: str = "patent_count",
    q: float = 0.5,
) -> pd.DataFrame:
    """Compute LQ (a.k.a. RTA/RCA-like) and mpc with a per-class quantile cutoff.

    For each technology class k, set a_k to the q-th percentile (default: 25th)
    of the distribution of patent counts across locations. Then define:
      mpc = 1 if (rta >= 1) OR (count >= a_k), else 0.

    Args:
        df: A DataFrame containing at least (prefecture_col, class_col, count_col).
        aggregate: If True, pre-aggregate counts by (prefecture, class).
        prefecture_col: Column name of location/prefecture.
        class_col: Column name of technology class.
        count_col: Column name of patent count for (prefecture, class).
        q: Quantile for per-class cutoff a_k (default 0.25).

    Returns:
        DataFrame with columns:
          [prefecture_col, class_col, count_col, rta, class_q, mpc]
        where class_q is the per-class quantile cutoff a_k.
    """
    cols = [prefecture_col, class_col, count_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    # 必要列だけ抽出（メモリ節約）
    base = df[cols]

    # 事前集計（重複( p,c )がある場合を安全に処理）
    if aggregate:
        base = (
            base.groupby([prefecture_col, class_col], observed=True, sort=False, as_index=False)[count_col]
                .sum()
        )

    # 総計（スカラー）
    total = float(base[count_col].sum())
    if total == 0:
        # 全ゼロなら早期リターン（rta=NaN, mpc=0）
        out = base.copy()
        out["rta"] = np.nan
        out["class_q"] = 0.0
        out["mpc"] = 0
        return out

    # 各クラス内での地域別件数分布の q 分位点（= a_k）を算出し列として付与
    # ※ groupby.transform でクラスごとの同一長ベクトルを返す
    result = (
        base
        .assign(
            _c_total=lambda x: x.groupby(class_col, observed=True)[count_col].transform("sum"),
            _p_total=lambda x: x.groupby(prefecture_col, observed=True)[count_col].transform("sum"),
        )
        .assign(
            rta=lambda x: (x[count_col] / x["_c_total"]) / (x["_p_total"] / total)
        )
        .drop(columns=["_c_total", "_p_total"])
        .assign(
            class_q=lambda x: x.groupby(class_col)[count_col].transform(
                lambda s: (s.quantile(0.75)-s.quantile(0.25))*1.5
            )
        )
        .assign(
            mpc=lambda x: np.where(
                (x["rta"] >= 1.0) | (x[count_col] >= x["class_q"]),
                1, 0
            ).astype(np.int64)
        )
    )

    return result

long_mcp = compute_pref_schmoch_lq(long_df)\
    .assign(
        app_nendo_period = '1975-2015'
    )

trade_cols = {
    "time": "app_nendo_period",
    "loc": "right_person_addr",
    "prod": "schmoch35",
    "val": "mpc",
}
c_df_mcp = ecomplexity(long_mcp, cols_input=trade_cols, rca_mcp_threshold=1,
                       presence_test="manual"
                   )
rename_col_dict = {"eci": "kci", "pci": "tci"}
col_order_list = [
    "app_nendo_period",
    "right_person_addr",
    "schmoch35",
    "patent_count",
    "mcp",
    "diversity",
    "ubiquity",
    "kci",
    "tci",
]

c_df_mcp = c_df_mcp[c_df_mcp["patent_count"] > 0].rename(columns=rename_col_dict)

c_df_mcp = pd.concat(
    [
        mor.kh_ki(c_df_mcp[c_df_mcp["app_nendo_period"] == period], 
                  "right_person_addr", "schmoch35")
        for period in c_df_mcp["app_nendo_period"].unique()
    ],
    axis="index",
    ignore_index=True,
    ).merge(
        schmoch35_df,
        left_on='schmoch35',
        right_on='Field_number',
        how='left'
    )\
    .drop(columns=['Field_number', 'schmoch35'])\
    .rename(columns={'Field_en': 'schmoch35'})

c_df_mcp#.query('right_person_addr == "東京都"')

#%%
import seaborn as sns


fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=c_df_mcp, x='schmoch35', y='patent_count', ax=ax)
plt.yscale('log')
plt.xticks(rotation=90)

#%%
period_col = "app_nendo_period"
agg_dict = {
    "patent_count": ("patent_count", "sum"),
    "unique_classes": ("schmoch35", "nunique"),
    "diversity": ("diversity", "first"),
    "kci": ("kci", "first"),
    **{f"kh_{i}": (f"kh_{i}", "first") for i in range(1, 21)}
}
right_person_df_mcp = (
    c_df_mcp.groupby([period_col, "right_person_addr"])
    .agg(**agg_dict)
    .reset_index()
)
right_person_df_mcp.sort_values(by='kci', ascending=False, ignore_index=True)

#%%
right_person_df_mcp.sort_values(by='kci', ascending=False, ignore_index=True)


#%%
from ecomplexity import ecomplexity
from calculation import method_of_reflections as mor
from importlib import reload
reload(mor)

trade_cols = {
    "time": "app_nendo_period",
    "loc": "right_person_addr",
    "prod": "schmoch35",
    "val": "patent_count",
}
rename_col_dict = {"eci": "kci", "pci": "tci"}
col_order_list = [
    "app_nendo_period",
    "right_person_addr",
    "schmoch35",
    "patent_count",
    "rca",
    "mcp",
    "diversity",
    "ubiquity",
    "kci",
    "tci",
]
c_df = ecomplexity(long_df, cols_input=trade_cols, rca_mcp_threshold=1,
                   )
c_df = c_df[c_df["patent_count"] > 0].rename(columns=rename_col_dict)[col_order_list]
c_df = pd.concat(
    [
        mor.kh_ki(c_df[c_df["app_nendo_period"] == period], "right_person_addr", "schmoch35")
        for period in c_df["app_nendo_period"].unique()
    ],
    axis="index",
    ignore_index=True,
)
c_df
period_col = "app_nendo_period"
agg_dict = {
    "patent_count": ("patent_count", "sum"),
    "unique_classes": ("schmoch35", "nunique"),
    "diversity": ("diversity", "first"),
    "kci": ("kci", "first"),
    **{f"kh_{i}": (f"kh_{i}", "first") for i in range(1, 21)}
}
right_person_df = (
    c_df.groupby([period_col, "right_person_addr"])
    .agg(**agg_dict)
    .reset_index()
)
right_person_df.sort_values(by='kci', ascending=False, ignore_index=True)
# outdir = '../../data/processed/internal/10_1_1_prefecture/'
# right_person_df.to_csv(
#     outdir + 'long.csv',
#     index=False, 
#     encoding='utf-8',
#     sep=','
# )

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    right_person_df['kci'],
    right_person_df['patent_count'],
    s=15,
    color='tab:blue'
)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('KCI vs. Patent Count')
ax.set_xlabel('KCI')
ax.set_ylabel('Patent Count')
# %%

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(
    right_person_df['diversity'],
    right_person_df['kh_1'],
    s=15,
    color='tab:blue'
)
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_title('Diversity vs. Average Ubiquity')
ax.set_xlabel('Diversity')
ax.set_ylabel('Average Ubiquity')
ax.set_xlim(0, 25)
ax.set_xticks(
    range(0, 25, 5)
)
ax.set_xticklabels(
    range(0, 25, 5)
)
# %%
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from typing import Dict, Tuple, Optional

# --- 末尾付きの辞書（略称, 地域） ---
pref_dict: Dict[str, Tuple[str, str]] = {
    "北海道": ("HK", "Hokkaido"),
    "青森県": ("AO", "Tohoku"),
    "岩手県": ("IW", "Tohoku"),
    "宮城県": ("MG", "Tohoku"),
    "秋田県": ("AK", "Tohoku"),
    "山形県": ("YT", "Tohoku"),
    "福島県": ("FS", "Tohoku"),
    "茨城県": ("IB", "Kanto"),
    "栃木県": ("TC", "Kanto"),
    "群馬県": ("GM", "Kanto"),
    "埼玉県": ("ST", "Kanto"),
    "千葉県": ("CH", "Kanto"),
    "東京都": ("TK", "Kanto"),
    "神奈川県": ("KN", "Kanto"),
    "新潟県": ("NI", "Chubu"),
    "富山県": ("TY", "Chubu"),
    "石川県": ("IS", "Chubu"),
    "福井県": ("FI", "Chubu"),
    "山梨県": ("YN", "Chubu"),
    "長野県": ("NN", "Chubu"),
    "岐阜県": ("GF", "Chubu"),
    "静岡県": ("SZ", "Chubu"),
    "愛知県": ("AI", "Chubu"),
    "三重県": ("ME", "Kansai"),
    "滋賀県": ("SH", "Kansai"),
    "京都府": ("KY", "Kansai"),
    "大阪府": ("OS", "Kansai"),
    "兵庫県": ("HG", "Kansai"),
    "奈良県": ("NR", "Kansai"),
    "和歌山県": ("WK", "Kansai"),
    "鳥取県": ("TT", "Chugoku"),
    "島根県": ("SM", "Chugoku"),
    "岡山県": ("OY", "Chugoku"),
    "広島県": ("HS", "Chugoku"),
    "山口県": ("YC", "Chugoku"),
    "徳島県": ("TS", "Shikoku"),
    "香川県": ("KG", "Shikoku"),
    "愛媛県": ("EH", "Shikoku"),
    "高知県": ("KC", "Shikoku"),
    "福岡県": ("FO", "Kyushu"),
    "佐賀県": ("SG", "Kyushu"),
    "長崎県": ("NS", "Kyushu"),
    "熊本県": ("KM", "Kyushu"),
    "大分県": ("OT", "Kyushu"),
    "宮崎県": ("MZ", "Kyushu"),
    "鹿児島県": ("KS", "Kyushu"),
    "沖縄県": ("ON", "Kyushu"),
}

# 地域→色
region_colors = {
    "Hokkaido": "tab:gray",
    "Tohoku": "navy",
    "Kanto": "tab:red",
    "Chubu": "tab:blue",
    "Kansai": "tab:orange",
    "Chugoku": "tab:green",
    "Shikoku": "purple",
    "Kyushu": "tab:brown",
}

# 軸ラベル変換
label_map = {
    "kci": "KCI",
    "kh_1": r"Ubiquity ($k_{p,1}$)",
    "diversity": r"Diversity ($k_{p,0}$)",
    "patent_count": "Patent counts",
}

def _abbr_and_region(jp_name: str) -> Tuple[str, Optional[str]]:
    """都道府県名から (略称, 地域) を返す"""
    if jp_name in pref_dict:
        return pref_dict[jp_name][0], pref_dict[jp_name][1]
    return jp_name[:2], None  # fallback

def plot_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    region_coloring: bool = True,
    show_legend: bool = True,
    s: int = 25,
) -> None:
    """主要指標の散布図（略称ラベル + 地域色分け + 参照線 + ログ軸）を描画する"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # region/abbr 列を作成
    _tmp = df.copy()
    _tmp["__abbr__"], _tmp["__region__"] = zip(*_tmp["right_person_addr"].map(_abbr_and_region))

    # 散布
    texts = []
    handles = []
    if region_coloring:
        for region, sub in _tmp.groupby("__region__"):
            color = region_colors.get(region, "tab:red")
            sc = ax.scatter(sub[xcol], sub[ycol], s=s, color=color, alpha=0.8, label=region or "Unknown")
            handles.append(sc)
            for _, row in sub.iterrows():
                texts.append(ax.text(row[xcol], row[ycol], row["__abbr__"], fontsize=8))
    else:
        ax.scatter(_tmp[xcol], _tmp[ycol], s=s, color="tab:red")
        for _, row in _tmp.iterrows():
            texts.append(ax.text(row[xcol], row[ycol], row["__abbr__"], fontsize=8))

    # log scale
    if "patent_count" in xcol:
        ax.set_xscale("log")
    if "patent_count" in ycol:
        ax.set_yscale("log")

    # 参照線
    def add_reference_line(col: str, axis: str) -> None:
        val = 0.0 if col == "kci" else float(df[col].mean())
        if axis == "x":
            ax.axvline(val, color="gray", linestyle="--", lw=1)
        else:
            ax.axhline(val, color="gray", linestyle="--", lw=1)
        if col == "diversity":
            if axis == "x":
                ax.set_xlim(-2, 37)
                ax.set_xticks(range(0, 35+1, 5))
                ax.set_xticklabels(range(0, 35+1, 5))
            else:
                ax.set_ylim(-2, 37)
                ax.set_yticks(range(0, 35+1, 5))
                ax.set_yticklabels(range(0, 35+1, 5))

    add_reference_line(xcol, "x")
    add_reference_line(ycol, "y")

    # 軸ラベル・タイトル
    xlabel = label_map.get(xcol, xcol)
    ylabel = label_map.get(ycol, ycol)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs. {ylabel} (corr: {df[xcol].corr(df[ycol]):.2f})")
    
    # ラベル調整
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        expand_points=(1.05, 1.05),
        expand_text=(1.1, 1.2),
        force_text=0.8,
        lim=200,
    )

    # 凡例
    if region_coloring and show_legend and handles:
        ax.legend(frameon=False, title="Region", loc="best", fontsize=9)

    # plt.tight_layout()
    # ax.set_aspect('equal', adjustable='box')
    
    plt.show()

col_list = [
    'diversity', 'kh_1', 'kci', 'patent_count'
    ]
for i, col in enumerate(col_list):
    for j, col2 in enumerate(col_list):
        plot_scatter(right_person_df_mcp, col, col2, region_coloring=False)
plot_scatter(right_person_df_mcp, 'kci', 'diversity', region_coloring=False)

# %%
c_df[c_df['right_person_addr'] == '東京都']\
    .sort_values(by='ubiquity', ascending=False)\
    .filter(items=['right_person_addr', 'mcp', 'ubiquity', 'rca'])

#%%
c_df[c_df['right_person_addr'] == '大阪府']\
    .sort_values(by='ubiquity', ascending=False)\
    .filter(items=['right_person_addr', 'mcp', 'ubiquity', 'rca'])
#%%
c_df[c_df['right_person_addr'] == '島根県']\
    .sort_values(by='ubiquity', ascending=False)\
    .filter(items=['right_person_addr', 'mcp', 'ubiquity', 'rca'])

# %%
grp_df = pd.read_csv(
    '../../data/processed/external/grp/2015.csv',
    encoding='utf-8',
    sep=',',
)
right_person_df_mcp.merge(
    grp_df,
    left_on='right_person_addr',
    right_on='prefecture',
    how='left'
)
# %%
plot_scatter(
    right_person_df_mcp.merge(
        grp_df,
        left_on='right_person_addr',
        right_on='prefecture',
        how='left'
    ),
    'GRP_per_capita_1000yen', 'kci', region_coloring=False
)

# %%