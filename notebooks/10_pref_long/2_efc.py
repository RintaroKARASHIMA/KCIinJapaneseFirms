#%%
import numpy as np
from scipy.stats import spearmanr


# Computation of RCA
def rca(biadjm):
    RCA = np.zeros_like(biadjm, dtype=float)
    rows_degrees = biadjm.sum(axis=1)
    cols_degrees = biadjm.sum(axis=0)
    tot_degrees = biadjm.sum()
    for i in range(np.shape(biadjm)[0]):
        if rows_degrees[i] != 0:
            for j in range(np.shape(biadjm)[1]):
                if cols_degrees[j] != 0:
                    RCA[i,j] = (biadjm[i, j] / rows_degrees[i]) / (cols_degrees[j] / tot_degrees)
    return RCA


# Computation of Fitness and Complexity
def Fitn_Comp(biadjm):
    FFQQ_ERR = 10 ** -4
    spe_value = 10**-3
    bam = np.array(biadjm)
    c_len, p_len = bam.shape
    ff1 = np.ones(c_len)
    qq1 = np.ones(p_len)
    ff0 = np.sum(bam, axis=1)
    ff0 = ff0 / np.mean(ff0)
    qq0 = 1. / np.sum(bam, axis=0)
    qq0 = qq0 / np.mean(qq0)

    ff0 = ff1
    qq0 = qq1
    ff1 = np.dot(bam, qq0)
    qq1 = 1./(np.dot(bam.T, 1. / ff0))
    ff1 /= np.mean(ff1)
    qq1 /= np.mean(qq1)
    coef = spearmanr(ff0, ff1)[0]

    coef = 0.
    i=0
    while np.sum(abs(ff1 - ff0)) > FFQQ_ERR and np.sum(abs(qq1 - qq0)) > FFQQ_ERR and 1-abs(coef)>spe_value:
        i+=1
        print(i)
        ff0 = ff1
        qq0 = qq1
        ff1 = np.dot(bam, qq0)
        qq1 = 1./(np.dot(bam.T, 1. / ff0))
        ff1 /= np.mean(ff1)
        qq1 /= np.mean(qq1)
        coef = spearmanr(ff0, ff1)[0]
    return (ff0, qq0)


# Computation of Coherent Diversification
def coherence(biadjm,B_network):
    bam = np.array(biadjm)
    B = np.array(B_network)
    div = np.sum(bam,axis=1)
    gamma = np.nan_to_num(np.dot(B,bam.T).T)
    GAMMA = bam * gamma
    return np.nan_to_num(np.sum(GAMMA,axis=1)/div)
# %%

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


# %%
df = raw_df.query('2015 <= app_nendo <= 2015', engine='python')\
           .drop(columns=['app_year', 'reg_year', 'reg_nendo'])\
           .drop_duplicates(keep='first')\
           .reset_index(drop=True)

# %%
long_df = (
    df.drop(
        columns=['ipc3', 'ipc4']
        )\
        .drop_duplicates(keep='first')\
        .assign(
            app_nendo_period = '2015-2015'
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
        )\
        .drop(columns=['app_nendo_period'])\
        .rename(columns={'right_person_addr':'prefecture'})
)
display(long_df)



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
    prefecture_col: str = "prefecture",
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
long_mcp


#%%
# 1) 行=地域, 列=技術 の行列にピボット
biadjm_df = long_df.pivot_table(
    index="prefecture",
    columns="schmoch35",
    values="patent_count",
    aggfunc="sum",
    fill_value=0
)

# 2) 0のみの行/列を削除（READMEが「重要」と明記）
biadjm_df = biadjm_df.loc[biadjm_df.sum(axis=1) > 0, :]
biadjm_df = biadjm_df.loc[:, biadjm_df.sum(axis=0) > 0]

# 3) NumPy配列へ
biadjm_counts = biadjm_df.values  # 重み付き（特許数）

# 4) presence行列（RCA>=1 を 1、それ以外 0）を作成（Fitn_Compに推奨）
R = rca(biadjm_counts)
# biadjm_presence = (R >= 1.0).astype(int)
biadjm_presence = long_mcp.pivot_table(
    index="prefecture",
    columns="schmoch35",
    values="mpc",
    aggfunc="sum",
    fill_value=0
)

#%%
# （オプション）presence が全0になった行/列を再度落とす
# presence_df = pd.DataFrame(biadjm_presence, index=biadjm_df.index, columns=biadjm_df.columns)
# presence_df = presence_df.loc[presence_df.sum(axis=1) > 0, :]
# presence_df = presence_df.loc[:, presence_df.sum(axis=0) > 0]
# biadjm_presence = presence_df.values

# 5) Fitness & Complexity（presenceを入力）
fitness, complexity = Fitn_Comp(biadjm_presence.values)
fitness
prefectures = biadjm_presence.index
result_df = pd.DataFrame(
    {
        'prefecture': prefectures,
        'fitness': fitness
    }
).sort_values(by='fitness', ascending=False, ignore_index=True)



# %%
result_df.assign(
    rank = lambda x: x['fitness'].rank(method='min', ascending=False).astype(np.int64)
).sort_values(by='rank', ascending=True, ignore_index=True)
# %%
pref_df = pd.merge(
    long_mcp,
    result_df.assign(
        rank = lambda x: x['fitness'].rank(method='min', ascending=False).astype(np.int64)
    ).sort_values(by='rank', ascending=True, ignore_index=True),
    on=['prefecture'],
    how='left'
    )
# %%

def aggregate_prefecture(df: pd.DataFrame) -> pd.DataFrame:
    """
    prefectureごとに集計を行い、新しいDataFrameを作成する。

    Args:
        df (pd.DataFrame): 入力DataFrame
            必須列: ["prefecture","schmoch35","patent_count","rta","class_q",
                     "mpc","app_nendo_period","fitness","rank"]

    Returns:
        pd.DataFrame: 集計後のDataFrame
    """
    agg_df = (
        df.groupby("prefecture", as_index=False)
        .agg({
            # schmoch35は削除 → 集計しない
            "patent_count": ["sum", list],  # 合計列とlist列を作成
            "rta": list,
            "class_q": list,
            "mpc": "sum",  # degree_centrality
            "app_nendo_period": lambda x: list(set(x))[0],
            "fitness": lambda x: list(set(x))[0],
            "rank": lambda x: list(set(x))[0]
        })
    )

    # MultiIndex列をフラット化
    agg_df.columns = [
        "prefecture",
        "patent_count_sum",
        "patent_count_list",
        "rta_list",
        "class_q_list",
        "degree_centrality",
        "app_nendo_period_unique",
        "fitness_unique",
        "rank_unique",
    ]

    return agg_df

# %%
grp_df = pd.read_csv(
    '../../data/processed/external/grp/2015.csv',
    encoding='utf-8',
    sep=',',
)
grp_df
# %%
agg_pref_df = aggregate_prefecture(pref_df)\
              .merge(
    grp_df,
    left_on='prefecture',
    right_on='prefecture',
    how='left'
)
agg_pref_df
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
    s: int = 30,
) -> None:
    """主要指標の散布図（略称ラベル + 地域色分け + 参照線 + ログ軸）を描画する"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # region/abbr 列を作成
    _tmp = df.copy()
    _tmp["__abbr__"], _tmp["__region__"] = zip(*_tmp["prefecture"].map(_abbr_and_region))

    # 散布
    texts = []
    handles = []
    if region_coloring:
        for region, sub in _tmp.groupby("__region__"):
            color = region_colors.get(region, "tab:red")
            sc = ax.scatter(sub[xcol], sub[ycol], s=s, color=color, label=region or "Unknown", alpha=0.8)
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
        expand_text=(1.05, 1.05),
        force_text=0.8,
        lim=200,
    )

    # 凡例
    if region_coloring and show_legend and handles:
        ax.legend(frameon=False, title="Region", loc="best", fontsize=9)

    # plt.tight_layout()
    # ax.set_aspect('equal', adjustable='box')
    
    plt.show()

#%%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'degree_centrality',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'fitness_unique',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'degree_centrality',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'degree_centrality',
    'fitness_unique',
    region_coloring=False
)

# %%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'GRP_per_capita_1000yen',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'degree_centrality',
    'GRP_per_capita_1000yen',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'GRP_per_capita_1000yen',
    'fitness_unique',
    region_coloring=False
)


# %%
