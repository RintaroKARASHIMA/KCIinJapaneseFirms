#%%
import sys
sys.path.append("../../src")
import os

import numpy as np
import pandas as pd
from typing import Literal


import matplotlib.pyplot as plt
from IPython.display import display

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Yu Mincho'
plt.rcParams['font.size'] = 15

from scipy.stats import spearmanr



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
df = raw_df.query('1981 <= app_nendo <= 2015', engine='python')\
           .drop(columns=['app_year', 'reg_year', 'reg_nendo'])\
           .drop_duplicates(keep='first')\
           .reset_index(drop=True)


#%%
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

window_size = 5
sep_mcp = pd.concat(
                   [
                    compute_pref_schmoch_lq(
                        df.drop(
                                columns=['ipc3', 'ipc4']
                                )\
                                .drop_duplicates(keep='first')\
                                .query('@window-@window_size+1 <= app_nendo <= @window', engine='python')\
                                .assign(
                                    addr_count=lambda x: x.groupby('reg_num')['right_person_addr'].transform('nunique'),
                                    class_count=lambda x: x.groupby('reg_num')['schmoch35'].transform('nunique')
                                )\
                                .assign(
                                    weight=lambda x: 1 / (x['addr_count'] * x['class_count'])
                                )\
                                .groupby(['right_person_addr', 'schmoch35'], as_index=False)\
                                .agg(
                                    patent_count=('weight', 'sum')
                                )\
                                .rename(columns={'right_person_addr':'prefecture'}))\
                                .assign(
                                    app_nendo_period = lambda x: f'{window-window_size+1}-{window}'
                                )
                    for window in range(1985, 2015+1)
                    ], 
                   axis='index',
                   ignore_index=True)
sep_mcp



#%%
def define_mcp(sep_mcp: pd.DataFrame) -> pd.DataFrame:
    
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
    biadjm_presence = sep_mcp.pivot_table(
        index="prefecture",
        columns="schmoch35",
        values="mpc",
        aggfunc="sum",
        fill_value=0
    )
    fitness, complexity = Fitn_Comp(biadjm_presence.values)
    prefectures = biadjm_presence.index
    result_df = pd.DataFrame(
        {
            'prefecture': prefectures,
            'fitness': fitness
        }
    ).sort_values(by='fitness', ascending=False, ignore_index=True)
    return result_df
    

# 5) Fitness & Complexity（presenceを入力）
result_df = pd.concat([define_mcp(sep_mcp.query('app_nendo_period == @period', engine='python'))\
                        .assign(
                                app_nendo_period = period, 
                                # rank = lambda x: x['fitness'].rank(method='min', ascending=False).astype(np.int64)
                                )
                        for period in sep_mcp['app_nendo_period'].unique()], ignore_index=True)
result_df

#%%
result_df['app_nendo_period'].unique()

# %%
pref_df = pd.merge(
    sep_mcp.groupby(['app_nendo_period', 'prefecture']).agg(
        {'mpc': 'sum', 'patent_count': 'sum'}
    ),
    result_df,
    on=['app_nendo_period', 'prefecture'],
    how='left'
    )
pref_df
# %%

# %%
grp_df = pd.read_csv(
    '../../data/processed/external/grp/grp_capita.csv',
    encoding='utf-8',
    sep=',',
    )\
    .sort_values(by=['prefecture', 'year'], ascending=True, ignore_index=True)\
    .assign(
        ln_GRP = lambda x: np.log(x['GRP']),
        ln_GRP_t5 = lambda x: x.groupby('prefecture')['ln_GRP'].shift(-5),
        g5_bar = lambda x: (x['ln_GRP_t5'] - x['ln_GRP'])/5,
        ln_GRP_pc_yen = lambda x: np.log(x['GRP_per_capita_yen']), 
        ln_GRP_pc_yen_t5 = lambda x: x.groupby('prefecture')['ln_GRP_pc_yen'].shift(-5),
        g5_bar_pc_yen = lambda x: (x['ln_GRP_pc_yen_t5'] - x['ln_GRP_pc_yen'])/5,
        ln_capita = lambda x: np.log(x['capita']),
    )\
    .rename(columns={'year': 'tau'})\
    .drop_duplicates(keep='first', ignore_index=True)\
    .query('(1981 <= tau <= 2015)', engine='python')
grp_df
#%%
fitness_df = pref_df.copy()\
                    .assign(
                        tau = lambda x: x['app_nendo_period'].str[-4:].astype(np.int64),
                        ln_patent = lambda x: np.log1p(x['patent_count']), 
                        ln_fitness = lambda x: np.log(x['fitness']),
                        ln_mcp = lambda x: np.log(x['mpc']),
                        z_fitness = lambda x: (x['fitness'] - x['fitness'].mean()) / x['fitness'].std(),
                    )
panel_df = pd.merge(
    grp_df,
    fitness_df,
    on=['prefecture', 'tau'],
    how='inner'
    )\
    .set_index(['prefecture', 'tau'])
panel_df


#%%
from linearmodels.panel import PanelOLS


model = PanelOLS.from_formula(
    "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + EntityEffects + TimeEffects",
    # "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + ln_capita",
    data=panel_df
)
# Driscoll-Kraay SE (空間+時間の依存を許容)
res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=7)
print(res.summary)

# %%
grp_df
# %%
