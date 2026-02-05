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
    '../../data/interim/internal/jp_filtered/japan_corporations.csv',
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
        columns=['ipc','corporation']
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
    producer_col: str = "prefecture",
    class_col: str = "schmoch35",
    count_col: str = "patent_count",
    q: float = 0.5,
) -> pd.DataFrame:
    """Compute LQ (a.k.a. RTA/RCA-like) and mpc with a per-class quantile cutoff.

    For each technology class k, set a_k to the q-th percentile (default: 25th)
    of the distribution of patent counts across locations. Then define:
      mpc = 1 if (rta >= 1) OR (count >= a_k), else 0.

    Args:
        df: A DataFrame containing at least (producer_col, class_col, count_col).
        aggregate: If True, pre-aggregate counts by (prefecture, class).
        producer_col: Column name of location/prefecture.
        class_col: Column name of technology class.
        count_col: Column name of patent count for (prefecture, class).
        q: Quantile for per-class cutoff a_k (default 0.25).

    Returns:
        DataFrame with columns:
          [producer_col, class_col, count_col, rta, class_q, mpc]
        where class_q is the per-class quantile cutoff a_k.
    """
    cols = [producer_col, class_col, count_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    # 必要列だけ抽出（メモリ節約）
    base = df[cols]

    # 事前集計（重複( p,c )がある場合を安全に処理）
    if aggregate:
        base = (
            base.groupby([producer_col, class_col], observed=True, sort=False, as_index=False)[count_col]
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
            _p_total=lambda x: x.groupby(producer_col, observed=True)[count_col].transform("sum"),
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
tech = 'schmoch35'
sep_mcp = pd.concat(
                   [
                    compute_pref_schmoch_lq(
                        df.drop(
                                columns=list(set(['ipc3', 'ipc4', 'schmoch35']) - set([tech]))
                                )\
                                .drop_duplicates(keep='first')\
                                .query('@window-@window_size+1 <= app_nendo <= @window', engine='python')\
                                .assign(
                                    addr_count=lambda x: x.groupby('reg_num')['prefecture'].transform('nunique'),
                                    class_count=lambda x: x.groupby('reg_num')[tech].transform('nunique')
                                )\
                                .assign(
                                    weight=lambda x: 1 / (x['addr_count'] * x['class_count'])
                                )\
                                .groupby(['prefecture', tech], as_index=False)\
                                .agg(
                                    patent_count=('weight', 'sum')
                                )
                                , class_col=tech
                                )\
                                .assign(
                                    app_nendo_period = lambda x: f'{window-window_size+1}-{window}'
                                )
                    for window in range(1981+window_size-1, 2015+1)
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
    def coherence(biadjm):
        bam = np.array(biadjm)
        u = biadjm.sum(axis=0)       # u_p
        d = biadjm.sum(axis=1)       # d_c

        B = np.zeros((biadjm.shape[1], biadjm.shape[1]))
        for p in range(biadjm.shape[1]):
            for p2 in range(biadjm.shape[1]):
                B[p, p2] = (biadjm[:, p] * biadjm[:, p2] / d).sum() / max(u[p], u[p2])
        div = np.sum(bam,axis=1)
        gamma = np.nan_to_num(np.dot(B,bam.T).T)
        GAMMA = bam * gamma
        return np.nan_to_num(np.sum(GAMMA,axis=1)/div)
    biadjm_presence = sep_mcp.pivot_table(
        index="prefecture",
        columns=tech,
        values="mpc",
        aggfunc="sum",
        fill_value=0
    )
    fitness, complexity = Fitn_Comp(biadjm_presence.values)
    prefectures = biadjm_presence.index
    result_df = pd.DataFrame(
        {
            'prefecture': prefectures,
            'fitness': fitness, 
            'coherence': coherence(biadjm_presence.values)
        }
    ).sort_values(by='fitness', ascending=False, ignore_index=True)
    tech_result_df = pd.DataFrame(
        {
            'tech': tech,
            'complexity': complexity
        }
    ).sort_values(by='complexity', ascending=False, ignore_index=True)
    return result_df, tech_result_df
    

# 5) Fitness & Complexity（presenceを入力）
result_df = pd.concat([define_mcp(sep_mcp.query('app_nendo_period == @period', engine='python'))\
                        [0]\
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
        capita_density = lambda x: x['capita'] / x['area'],
        ln_capita_density = lambda x: np.log(x['capita_density']),
        ln_GRP = lambda x: np.log(x['GRP']),
        ln_GRP_t5 = lambda x: x.groupby('prefecture')['ln_GRP'].shift(-window_size),
        g5_bar = lambda x: (x['ln_GRP_t5'] - x['ln_GRP'])/window_size,
        ln_GRP_pc_yen = lambda x: np.log(x['GRP_per_capita_yen']), 
        ln_GRP_pc_yen_t5 = lambda x: x.groupby('prefecture')['ln_GRP_pc_yen'].shift(-window_size),
        g5_bar_pc_yen = lambda x: (x['ln_GRP_pc_yen_t5'] - x['ln_GRP_pc_yen'])/window_size,
        ln_capita = lambda x: np.log(x['capita']),
        ln_g5_bar = lambda x: np.log1p(x['g5_bar']),
        ln_g5_bar_pc_yen = lambda x: np.log1p(x['g5_bar_pc_yen']),
    )\
    .rename(columns={'year': 'tau'})\
    .drop_duplicates(keep='first', ignore_index=True)\
    .query('(1981 <= tau <= 2020 & (tau-1980-@window_size)%@window_size==0)', engine='python')
grp_df
#%%
fitness_df = pref_df.copy()\
                    .assign(
                        tau = lambda x: x['app_nendo_period'].str[-4:].astype(np.int64),
                        ln_patent_count = lambda x: np.log1p(x['patent_count']), 
                        ln_fitness = lambda x: np.log(x['fitness']),
                        ln_mcp = lambda x: np.log(x['mpc']),
                        z_fitness = lambda x: (x['fitness'] - x['fitness'].mean()) / x['fitness'].std(),
                        fitness_lag1 = lambda x: x.groupby('prefecture')['fitness'].shift(1),
                        fitness_lag2 = lambda x: x.groupby('prefecture')['fitness'].shift(2),
                        fitness_lag3 = lambda x: x.groupby('prefecture')['fitness'].shift(3),
                        fitness_lag4 = lambda x: x.groupby('prefecture')['fitness'].shift(4),
                        fitness_lag5 = lambda x: x.groupby('prefecture')['fitness'].shift(5),
                        fitness_lead1 = lambda x: x.groupby('prefecture')['fitness'].shift(-1),
                        fitness_lead2 = lambda x: x.groupby('prefecture')['fitness'].shift(-2),
                        fitness_lead3 = lambda x: x.groupby('prefecture')['fitness'].shift(-3),
                        fitness_lead4 = lambda x: x.groupby('prefecture')['fitness'].shift(-4),
                        fitness_lead5 = lambda x: x.groupby('prefecture')['fitness'].shift(-5),
                    )
panel_df = pd.merge(
    grp_df,
    fitness_df,
    on=['prefecture', 'tau'],
    how='inner'
    )\
    .set_index(['prefecture', 'tau'])\
    .assign(
        grp_fitness_1000 = lambda x: x['fitness'] * x['GRP'] /1000,
        grp_fitness_100million = lambda x: x['fitness'] * x['GRP'] /100_000_000,
        pop_fitness = lambda x: x['fitness'] * x['capita_density'],
        patent_fitness = lambda x: x['fitness'] * x['patent_count'],
        ln_grp_fitness = lambda x: x['fitness'] * x['ln_GRP'],
        ln_pop_fitness = lambda x: x['fitness'] * x['ln_capita_density'],
        ln_patent_fitness = lambda x: x['fitness'] * x['ln_patent_count'],
    )
panel_df
#%%
panel_df.columns

#%%

#%%
# ===== モデル指定用の辞書（あなたのまま） =====
base_model_str = 'g5_bar_pc_yen ~ 1 + ln_GRP + fitness + ln_capita_density + ln_patent_count + EntityEffects + TimeEffects'
model_str_dict = {
    'effect': [base_model_str.replace('+ EntityEffects + TimeEffects', f'{effect}')
               for effect in ['', '+ EntityEffects', '+ TimeEffects', '+ EntityEffects + TimeEffects']],
    'lag': [base_model_str.replace('fitness', f'fitness{fit}')
            for fit in ['_lag1', '_lag2', '_lag3', '_lag4', '_lag5']],
    'lead': [base_model_str.replace('fitness', f'fitness{fit}')
             for fit in ['_lead1', '_lead2', '_lead3', '_lead4', '_lead5']],
    'interaction': [base_model_str + " + " + interaction
                    for interaction in [
                        'ln_GRP:fitness',
                        'ln_capita_density:fitness',
                        'ln_patent_count:fitness'
                    ]],
    'arrange': [
        base_model_str + ' + pop_fitness',
        base_model_str + ' + ln_pop_fitness',
        base_model_str + ' + ln_grp_fitness',
        base_model_str + ' + GRP_per_capita_1000yen',
        base_model_str + ' + grp_fitness_1000',  # 元コードで重複があったので残しています
    ],
}

#%%
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.model import PanelEffectsResults
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%
# ===== 繰り返し部分のみ最小限の関数化 =====
def _fit_and_print_panelols(
    panel_df: pd.DataFrame,
    formula: str,
    title: Optional[str] = None,
    bandwidth: int = 3,
) -> PanelEffectsResults:
    """PanelOLS を推定して summary を表示する（最小限の共通化）。

    Args:
        panel_df: MultiIndex (entity, time) のパネルデータ。
        formula: PanelOLS.from_formula の式。
        title: 表示用タイトル（Noneならformulaを表示）。
        bandwidth: Driscoll-Kraay kernel の bandwidth。

    Returns:
        推定結果。
    """
    print('*' * 115)
    print(title if title is not None else formula)
    model = PanelOLS.from_formula(formula, data=panel_df)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=bandwidth)
    print(res.summary)
    return res

#%%
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults


def _fit_and_print_panelols(
    panel_df: pd.DataFrame,
    formula: str,
    title: Optional[str] = None,
    bandwidth: int = 3,
) -> PanelEffectsResults:
    """PanelOLS を推定して summary と最終期間（7期のうち最後の1期）の予測結果を表示する。

    7期間ある想定のパネルに対して、最後の1期間を hold-out として学習（残り期間）→予測を行う。
    期間数が7でない場合でも、最後の1期間を hold-out として扱う。

    Args:
        panel_df: MultiIndex (entity, time) のパネルデータ。
        formula: PanelOLS.from_formula の式（例: "y ~ 1 + x1 + x2 + EntityEffects + TimeEffects"）。
        title: 表示用タイトル（Noneならformulaを表示）。
        bandwidth: Driscoll-Kraay kernel の bandwidth。

    Returns:
        学習データ（最後期を除く）で推定した結果。
    """
    if not isinstance(panel_df.index, pd.MultiIndex) or panel_df.index.nlevels < 2:
        raise ValueError("panel_df must have a MultiIndex with (entity, time).")

    entity_level = 0
    time_level = 1

    # 予測対象（最後の1期間）を決める
    times = panel_df.index.get_level_values(time_level)
    unique_times = pd.Index(times.unique())
    try:
        unique_times_sorted = unique_times.sort_values()
    except Exception:
        # ソートできない型の場合は、出現順を維持して最後を使う
        unique_times_sorted = unique_times

    if len(unique_times_sorted) < 2:
        raise ValueError("Need at least 2 time periods to hold out the last period for prediction.")

    last_time = unique_times_sorted[-1]

    # train / test split（最後期を hold-out）
    is_last = times == last_time
    train_df = panel_df.loc[~is_last].copy()
    test_df = panel_df.loc[is_last].copy()

    # 目的変数名を formula から取得（"y ~ ..." の左辺）
    lhs = formula.split("~", 1)[0].strip()
    if lhs == "":
        raise ValueError("Could not parse dependent variable from formula.")

    if lhs not in panel_df.columns:
        raise ValueError(f"Dependent variable '{lhs}' not found in panel_df columns.")

    print("*" * 115)
    print(title if title is not None else formula)

    # 学習データで推定
    model = PanelOLS.from_formula(formula, data=train_df)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=bandwidth)
    print(res.summary)

    # 最後期を予測して表示
    print("-" * 115)
    print(f"Hold-out prediction for the last period: time = {last_time!r}")
    try:
        y_pred = res.predict(data=test_df)
    except TypeError:
        # linearmodels のバージョン差異で引数名が違う場合に備える
        y_pred = res.predict(test_df)

    # y_pred は DataFrame/Series の可能性があるので Series に寄せる
    if isinstance(y_pred, pd.DataFrame):
        # 1列（予測値）のはず
        if y_pred.shape[1] == 1:
            y_pred_s = y_pred.iloc[:, 0]
        else:
            # 念のため
            y_pred_s = y_pred.mean(axis=1)
    else:
        y_pred_s = pd.Series(y_pred)

    y_true_s = test_df[lhs]

    # index 揃え
    common_index = y_true_s.index.intersection(y_pred_s.index)
    y_true_s = y_true_s.loc[common_index]
    y_pred_s = y_pred_s.loc[common_index]

    err = y_true_s - y_pred_s
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    corr = float(np.corrcoef(y_true_s.values, y_pred_s.values)[0, 1]) if len(common_index) >= 2 else float("nan")

    print(f"n_obs (hold-out): {len(common_index)}")
    print(f"MAE : {mae:.6g}")
    print(f"RMSE: {rmse:.6g}")
    print(f"Corr(y, yhat): {corr:.6g}")

    # 先頭だけ見やすく表示
    preview = pd.DataFrame({"y_true": y_true_s, "y_pred": y_pred_s, "error": err}).sort_index()
    print("\nPreview (first 10 rows):")
    print(preview.head(10).to_string())

    return res


#%%
_fit_and_print_panelols(panel_df, base_model_str, title="baseline")

#%%
def plot_last_period_prediction(
    panel_df: pd.DataFrame,
    res: PanelEffectsResults,
    formula: str,
    top_k: int = 15,
) -> pd.DataFrame:
    """最後の1期間の予測（y_true vs y_pred）を可視化する。

    図:
      1) y_true vs y_pred の散布図（45度線付き）
      2) 誤差（y_true - y_pred）の棒グラフ（絶対誤差が大きい順に上位 top_k）

    Args:
        panel_df: MultiIndex (entity, time) のパネルデータ。
        res: _fit_and_print_panelols が返す PanelOLS の推定結果（trainでfitされたもの）。
        formula: _fit_and_print_panelols と同じ式（左辺から目的変数名を抽出する）。
        top_k: 誤差棒グラフに表示する上位件数。

    Returns:
        最後期の予測結果 DataFrame（indexは(entity, time)）。
        columns: ["y_true", "y_pred", "error", "abs_error"]
    """
    if not isinstance(panel_df.index, pd.MultiIndex) or panel_df.index.nlevels < 2:
        raise ValueError("panel_df must have a MultiIndex with (entity, time).")

    time_level = 1
    times = panel_df.index.get_level_values(time_level)
    unique_times = pd.Index(times.unique())
    try:
        unique_times_sorted = unique_times.sort_values()
    except Exception:
        unique_times_sorted = unique_times
    if len(unique_times_sorted) < 2:
        raise ValueError("Need at least 2 time periods.")

    last_time = unique_times_sorted[-1]

    lhs = formula.split("~", 1)[0].strip()
    if lhs not in panel_df.columns:
        raise ValueError(f"Dependent variable '{lhs}' not found in panel_df columns.")

    test_df = panel_df.loc[times == last_time].copy()

    try:
        y_pred = res.predict(data=test_df)
    except TypeError:
        y_pred = res.predict(test_df)

    if isinstance(y_pred, pd.DataFrame):
        y_pred_s = y_pred.iloc[:, 0] if y_pred.shape[1] == 1 else y_pred.mean(axis=1)
    else:
        y_pred_s = pd.Series(y_pred, index=test_df.index)

    y_true_s = test_df[lhs]

    common_index = y_true_s.index.intersection(y_pred_s.index)
    y_true_s = y_true_s.loc[common_index]
    y_pred_s = y_pred_s.loc[common_index]

    out = pd.DataFrame(
        {
            "y_true": y_true_s,
            "y_pred": y_pred_s,
        }
    )
    out["error"] = out["y_true"] - out["y_pred"]
    out["abs_error"] = np.abs(out["error"])

    # ---- Plot 1: scatter ----
    plt.figure()

    x = out["y_pred"].to_numpy()
    y = out["y_true"].to_numpy()

    plt.scatter(x, y, s=18, alpha=0.6, linewidths=0)

    # 45度線（表示範囲に合わせる）
    # 外れ値に引っ張られないように、1%〜99%で軸範囲を決めるのがおすすめ
    x_lo, x_hi = np.nanpercentile(x, [1, 99])
    y_lo, y_hi = np.nanpercentile(y, [1, 99])
    lo = float(min(x_lo, y_lo))
    hi = float(max(x_hi, y_hi))

    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lo -= pad
    hi += pad

    plt.plot([lo, hi], [lo, hi])

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")  # 重要：歪みをなくす

    plt.axhline(0)
    plt.axvline(0)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Last-period prediction scatter (time={last_time!r})")
    plt.tight_layout()
    plt.show()


    # ---- Plot 2: top-k abs error bar ----
    # entity のラベルだけ取り出す（MultiIndex前提）
    entities = out.index.get_level_values(0).astype(str)
    out2 = out.copy()
    out2["entity"] = entities

    worst = out2.sort_values("abs_error", ascending=False).head(int(top_k))

    plt.figure()
    plt.bar(worst["entity"].values, worst["error"].values)
    plt.axhline(0)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Entity")
    plt.ylabel("Error (y_true - y_pred)")
    plt.title(f"Top-{top_k} errors in last period (time={last_time!r})")
    plt.tight_layout()
    plt.show()

    return out
res = _fit_and_print_panelols(panel_df, base_model_str + ' + grp_fitness_1000', title="grp_fitness_1000")
pred_df = plot_last_period_prediction(panel_df, res, base_model_str + ' + grp_fitness_1000')

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(pred_df['y_true'], 
           pred_df['y_pred'])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_title('Actual vs Predicted')
plt.show()

#%%
# ===== lead のループ（あなたの意図のまま、ぐちゃぐちゃだけ解消）=====
# for fit in ['_lag1', '_lag2', '_lag3', '_lag4', '_lag5']:
for formula in model_str_dict['lead']:
    # title は元の print('fitness'+fit) 相当の最小限表示
    # formula から fitness_leadX を抜く（雑に）
    title = [tok for tok in formula.split() if 'fitness_' in tok]
    title = title[0] if title else "lead"
    res = _fit_and_print_panelols(panel_df, formula, title=title)

#%%
# ===== fixed effect（あなたのまま）=====
for formula in [m for m in model_str_dict['effect'] if 'EntityEffects' in m and 'TimeEffects' in m]:
    res = _fit_and_print_panelols(panel_df, formula, title="+EntityEffects + TimeEffects")

#%%
# ===== last period を除いて推定→最後の期間を予測（あなたの関数はそのまま）=====
from typing import Tuple

def run_panel_and_predict_last_period(
    panel_df: pd.DataFrame,
) -> Tuple[PanelEffectsResults, pd.DataFrame]:
    """最後の1期間を除いて PanelOLS を推定し、その最後の1期間を予測する。"""

    time_level_name = panel_df.index.names[1]
    time_index = panel_df.index.get_level_values(time_level_name)
    last_period = time_index.max()

    train_df = panel_df[time_index < last_period]
    test_df = panel_df[time_index == last_period].copy()

    fixed_effect = "+EntityEffects + TimeEffects"

    formula = (
        "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + "
        "ln_capita_density + ln_patent_count" + fixed_effect
    )

    print("*" * 115)
    print(fixed_effect)
    model = PanelOLS.from_formula(formula, data=train_df)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    print(res.summary)

    y_pred = model.predict(res.params, data=test_df)
    test_df["g5_bar_pc_yen_pred"] = y_pred

    return res, test_df


if __name__ == "__main__":
    res, last_period_df = run_panel_and_predict_last_period(panel_df)

    print("\n=== Actual vs Predicted for the last period ===")
    print(last_period_df[["g5_bar_pc_yen", "g5_bar_pc_yen_pred"]].head(20))

#%%
# ===== 既存 plot は出力を変えないため据え置き（あなたのまま）=====
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(last_period_df['g5_bar_pc_yen'], last_period_df['g5_bar_pc_yen_pred'])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
plt.show()

#%%
last_period_df[["g5_bar_pc_yen", "g5_bar_pc_yen_pred"]].corr()

#%%
# ===== arrange 系（あなたの repeated を辞書側に寄せて最小限に）=====
for i, formula in enumerate(model_str_dict["arrange"], start=1):
    res = _fit_and_print_panelols(panel_df, formula, title=f"arrange_{i:02d}")

#%%
# ===== VIF（あなたの元コードの意図を維持しつつ、最小限の整理）=====
formula = "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + ln_capita_density + ln_patent_count + EntityEffects + TimeEffects"
res_for_vif = _fit_and_print_panelols(panel_df, formula, title="baseline for VIF")

y_name = "g5_bar_pc_yen"
x_vars = ["ln_GRP", "fitness", "ln_capita_density", "ln_patent_count"]

est_df = panel_df[[y_name] + x_vars].dropna()
X = est_df[x_vars].copy()

Gi = X.groupby(level=0).transform("mean")  # entity mean
Gt = X.groupby(level=1).transform("mean")  # time mean
G = X.mean()                               # overall mean
X_within = X - Gi - Gt + G

X_for_vif = sm.add_constant(X_within, has_constant="add").to_numpy()
vif_vals = []
for j, name in enumerate(["const"] + x_vars):
    try:
        v = variance_inflation_factor(X_for_vif, j)
    except np.linalg.LinAlgError:
        v = np.inf
    vif_vals.append((name, float(v)))

vif_df = pd.DataFrame(vif_vals, columns=["variable", "VIF"]).sort_values("VIF", ascending=False)
print("\n[Two-way FE (within) based VIF on the exact estimation sample]")
print(vif_df)

# ===== 既存の VIF plot は出力が変わらないよう据え置き（あなたのまま）=====
def plot_vif_horizontal(vif_df, threshold: float = 10.0):
    d = vif_df.sort_values("VIF", ascending=True)
    plt.figure(figsize=(7, 4))
    bars = plt.barh(d["variable"], d["VIF"], color='navy')
    plt.axvline(threshold, linestyle="--", label=f"閾値:{threshold}", color='red')
    for bar, v in zip(bars, d["VIF"]):
        plt.text(v + 0.2, bar.get_y() + bar.get_height()/2, f"{v:.2f}", va="center", fontsize=15)
    plt.xlabel("VIF"); #plt.title("VIF (Two-way FE within)")
    plt.yticks(range(0, 5), ['$ln({GRP})$','$Fitness$','$ln({capita\_density})$','$ln({patent\_count})$',
                             f'$constant$'][::-1], fontsize=15)
    plt.tight_layout(); plt.show()

plot_vif_horizontal(vif_df)

#%%
# ===== 残差自己相関（あなたのまま）=====
resid = res_for_vif.resids
resid_df = resid.unstack(level=0)
acf_1 = resid_df.apply(lambda x: x.autocorr(lag=3))
print(acf_1.sort_values(ascending=False))

#%%
# ===== 相関 heatmap（あなたのまま：seaborn指定も維持）=====
import seaborn as sns
plt.rcParams['font.size'] = 6
fig, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(panel_df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
plt.show()

#%%
# =============================================================================
# 追加: 各期間の回帰係数を時系列で plot（expanding window）
# =============================================================================

def plot_fitness_coef_over_time_with_ci(
    panel_df: pd.DataFrame,
    formula: str,
    bandwidth: int = 3,
    title: str = "Fitness coefficient over time (95% CI)",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """expanding window で fitness 係数の推移と 95%CI を描く（bandwidth自動調整付き）。"""

    time_level_name = panel_df.index.names[1]
    entity_level_name = panel_df.index.names[0]

    times = panel_df.index.get_level_values(time_level_name)
    uniq_times = pd.Index(times.unique()).sort_values()

    has_time_fe = "TimeEffects" in formula
    has_entity_fe = "EntityEffects" in formula
    def _z_value_from_alpha(alpha: float) -> float:
        """代表的な信頼水準に対応する z 値を返す。"""
        if alpha == 0.10:
            return 1.6448536269514722   # 90%
        elif alpha == 0.05:
            return 1.959963984540054    # 95%
        elif alpha == 0.03:
            return 2.170090377584560    # 97%
        elif alpha == 0.01:
            return 2.5758293035489004   # 99%
        else:
            raise ValueError("alpha must be one of {0.10, 0.05, 0.03, 0.01}")

    z = _z_value_from_alpha(alpha)

    rows = []
    for t in uniq_times:
        sub_df = panel_df[times <= t]

        n_time = sub_df.index.get_level_values(time_level_name).nunique()
        n_entity = sub_df.index.get_level_values(entity_level_name).nunique()
        n_obs = len(sub_df)

        # ---- 最小限のガード（TimeEffects/EntityEffects が成立する最低条件）----
        if n_obs == 0 or (has_time_fe and n_time < 2) or (has_entity_fe and n_entity < 2):
            rows.append({"time": t, "beta": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

        # ---- Driscoll-Kraay bandwidth をサンプルの time 数に合わせて縮める ----
        # DK は概ね「T-1 まで」しか取れないので、bw = min(requested, T-1)
        bw = min(bandwidth, int(n_time - 1))
        if bw < 1:
            rows.append({"time": t, "beta": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

        model = PanelOLS.from_formula(formula, data=sub_df)
        try:
            res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)

            beta = float(res.params.get("fitness", np.nan))

            # std_errors が covariance 計算で落ちる可能性があるので try
            try:
                se = float(res.std_errors.get("fitness", np.nan))
            except ValueError:
                se = np.nan

            ci_low = beta - z * se if np.isfinite(beta) and np.isfinite(se) else np.nan
            ci_high = beta + z * se if np.isfinite(beta) and np.isfinite(se) else np.nan

            rows.append({"time": t, "beta": beta, "se": se, "ci_low": ci_low, "ci_high": ci_high})

        except ValueError:
            # どうしても落ちる時点はスキップ（NaN）
            rows.append({"time": t, "beta": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

    out_df = pd.DataFrame(rows).set_index("time")

    # ---- plot（新規追加のみ。既存プロットは一切触らない）----
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    x = out_df.index
    y = out_df["beta"]
    lo = out_df["ci_low"]
    hi = out_df["ci_high"]
    plt.rcParams['font.size'] = 15
    ax.plot(x, y, marker="o", linewidth=1, label="fitness")
    ax.fill_between(x, lo, hi, alpha=0.2, label=f"{int(100*(1-alpha))}% CI")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Coefficient")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return out_df




# %%
# 97% CI
fitness_ts_df_97 = plot_fitness_coef_over_time_with_ci(
    panel_df=panel_df,
    formula=formula,
    bandwidth=3,
    alpha=0.03,
    title="Fitness coefficient over time (97% CI)",
)

# 99% CI
fitness_ts_df_99 = plot_fitness_coef_over_time_with_ci(
    panel_df=panel_df,
    formula=base_model_str+' + grp_fitness_1000',
    bandwidth=3,
    alpha=0.01,
    title="Fitness coefficient over time (99% CI)",
)



# %%
panel_df.reset_index(drop=False).columns

#%%

import pandas as pd
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional, Literal

def get_custom_colors(background: str = "light") -> List[str]:
    colors_for_light_bg = [
        "#4B7CCC", "#F2668B", "#03A688", "#FFAE3E", "#B782B8",
        "#A67F63", "#0E8B92", "#D4AC2C", "#7E9F5C", "#F7BCA3",
        "#E63946", "#7DA9A7", "#457B9D", "#E094AC", "#1D3557",
        "#2A9D8F", "#B38A44", "#C68045", "#264653",
    ]
    colors_for_dark_bg = [
        "#A8C9F4", "#FFB3C6", "#A0E5D6", "#FFEC88", "#E2A8D3",
        "#D0B89D", "#80D0D4", "#F1E59B", "#B4D79E", "#F2B48C",
        "#FFB4B4", "#A2D1D1", "#7FBCD1", "#F1A8D6", "#5C7F9E",
        "#70C7B7", "#C8A67F", "#D7A584", "#A4B6D4",
    ]
    return colors_for_dark_bg if background == "dark" else colors_for_light_bg


def prepare_bump_data(
    df: pd.DataFrame,
    *,
    entity_col: str = "prefecture",
    tau_col: str = "tau",
    value_col: str = "fitness",
    rank_method: Literal["dense", "min", "average", "first", "max"] = "dense",
    ascending: bool = False,  # fitnessが大きいほど上位なら False
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    dfからバンプチャート用データを作る。
    - 各 tau 内で value_col をランキング化して Ranking を作る
    - 必要に応じて top_k のみ残す
    """
    need = {entity_col, tau_col, value_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df[[entity_col, tau_col, value_col]].copy()
    d = d.dropna(subset=[entity_col, tau_col, value_col])

    # 1) tau の並び順を固定（数値/カテゴリ/文字列どれでもOKにする）
    #    - まずはユニークを取り、可能ならソート
    tau_vals = d[tau_col].unique().tolist()
    try:
        tau_order = sorted(tau_vals)
    except Exception:
        tau_order = tau_vals  # ソートできない型は出現順

    # 2) 各 tau 内でランキング（1が上）
    #    plotlyで上が1になるように yaxis を reversed にする前提
    #    rank() は昇順=Trueで小さいほど1、降順なら ascending=False にしたい
    d["Ranking"] = (
        d.groupby(tau_col)[value_col]
         .rank(method=rank_method, ascending=ascending)
         .astype(int)
    )

    # 3) top_k のみ残す（各 tau の上位kだけ残すと線が途切れるので、
    #    「どこかのtauでtop_kに入った県」を残す、が見やすいです）
    if top_k is not None:
        keep_entities = (
            d[d["Ranking"] <= top_k][entity_col]
            .drop_duplicates()
            .tolist()
        )
        d = d[d[entity_col].isin(keep_entities)]

    # 4) tau順に並べる
    d[tau_col] = pd.Categorical(d[tau_col], categories=tau_order, ordered=True)
    d = d.sort_values([entity_col, tau_col])

    # hover用に値列名を統一しておく（Income相当）
    d = d.rename(columns={entity_col: "Entity", tau_col: "Tau", value_col: "Value"})
    return d


def add_entity_traces(
    fig: go.Figure,
    df_bump: pd.DataFrame,
    custom_colors: List[str],
    tau_order: List,
    *,
    marker_size: int = 8,
    line_width: int = 2,
    label_buffer: float = 0.35,  # 最終tauの右側に置く用（カテゴリ軸なので index + buffer）
):
    """
    県（Entity）ごとに線を追加し、最終点の右に注釈ラベルを付ける。
    """
    entities = df_bump["Entity"].unique().tolist()

    for i, ent in enumerate(entities):
        ent_data = df_bump[df_bump["Entity"] == ent]
        if ent_data.empty:
            continue

        # 最終tauの点（ent_dataはTauでソート済み）
        last_point = ent_data.iloc[-1]

        # カテゴリ軸での位置（0,1,2,...)に変換して注釈を右にずらす
        last_tau = last_point["Tau"]
        last_tau_index = tau_order.index(last_tau)

        color = custom_colors[i % len(custom_colors)]

        fig.add_trace(
            go.Scatter(
                x=ent_data["Tau"],
                y=ent_data["Ranking"],
                mode="lines+markers",
                name=ent,
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size),
                customdata=ent_data[["Value"]],
                hovertemplate=(
                    "<b>Prefecture:</b> %{fullData.name}"
                    "<br><b>Tau:</b> %{x}"
                    "<br><b>Rank:</b> %{y}"
                    "<br><b>Fitness:</b> %{customdata[0]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_annotation(
            x=last_tau_index + label_buffer,
            y=last_point["Ranking"],
            text=ent,
            showarrow=False,
            font=dict(size=13, color=color),
            xanchor="left",
        )


def customize_layout(
    fig: go.Figure,
    *,
    tau_order: Optional[List] = None,
    height: int = 850,
    width: int = 900,
    show_tau_ticks: bool = True,
    y_dtick: int = 1,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    footer: Optional[str] = None,
):
    # yは順位なので上が1になるよう反転
    fig.update_yaxes(
        autorange="reversed",
        tickmode="linear",
        dtick=y_dtick,
        title=None,
    )

    # xをカテゴリとして順序固定
    if tau_order is not None:
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=tau_order,
            tickangle=270,
            showticklabels=show_tau_ticks,
            title=None,
        )

    fig.update_layout(
        title=title,
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(255,255,255,1)",
        font=dict(family="Poppins"),
        height=height,
        width=width,
        showlegend=False,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=40, r=220, t=90, b=90),  # 右に注釈スペース
    )

    # subtitle / footer
    if subtitle:
        fig.add_annotation(
            text=subtitle,
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="gray"),
            align="center",
        )
    if footer:
        fig.add_annotation(
            text=footer,
            x=0.5,
            y=-0.13,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
            align="center",
        )


def create_fitness_bump_chart(
    df: pd.DataFrame,
    *,
    entity_col: str = "prefecture",
    tau_col: str = "tau",
    fitness_col: str = "fitness",
    rank_method: Literal["dense", "min", "average", "first", "max"] = "dense",
    top_k: Optional[int] = 15,
    background: str = "light",
    title: Optional[str] = None,
):
    """
    tau×prefectureのfitnessランキング推移（バンプチャート）を作る。
    """
    df_bump = prepare_bump_data(
        df,
        entity_col=entity_col,
        tau_col=tau_col,
        value_col=fitness_col,
        rank_method=rank_method,
        ascending=False,  # fitness大きいほど上位
        top_k=top_k,
    )

    # tauの順序（Categorical categories）
    tau_order = list(df_bump["Tau"].cat.categories)

    fig = go.Figure()
    colors = get_custom_colors(background=background)

    add_entity_traces(
        fig,
        df_bump,
        colors,
        tau_order,
        marker_size=7,
        line_width=2,
        label_buffer=0.35,
    )

    customize_layout(
        fig,
        tau_order=tau_order,
        title=title,
        subtitle="Ranking changes by fitness across tau",
        footer="Note: Rank is computed within each tau (higher fitness = better rank).",
    )

    return fig


import pandas as pd
from typing import Optional, Literal

def prepare_bump_data(
    df: pd.DataFrame,
    *,
    entity_col: str = "prefecture",
    tau_col: str = "tau",
    value_col: str = "fitness",
    tie_break_col: Optional[str] = "GRP",
    rank_method: Literal["dense", "min", "average", "first", "max"] = "dense",
    ascending: bool = False,     # fitness が大きいほど上位なら False
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    バンプチャート用データ作成。
    - tie_break_col が指定されている場合:
        tau内で (value_col desc, tie_break_col desc) で並べて cumcount でユニーク順位
    - tie_break_col が None の場合:
        value_col の rank(method=rank_method) を使用
    """
    need = {entity_col, tau_col, value_col}
    if tie_break_col is not None:
        need.add(tie_break_col)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cols = [entity_col, tau_col, value_col] + ([tie_break_col] if tie_break_col else [])
    d = df[cols].copy()
    d = d.dropna(subset=[entity_col, tau_col, value_col] + ([tie_break_col] if tie_break_col else []))

    # tau の順序固定
    tau_vals = d[tau_col].unique().tolist()
    try:
        tau_order = sorted(tau_vals)
    except Exception:
        tau_order = tau_vals

    # --- ランキング作成 ---
    if tie_break_col is not None:
        d[tie_break_col] = pd.to_numeric(d[tie_break_col], errors="coerce")
        d[tie_break_col] = d[tie_break_col].fillna(-np.inf)
        # ★fitness同率をGRPで解消（ユニーク順位）
        d = d.sort_values(
            [tau_col, value_col, tie_break_col],
            ascending=[True, ascending, False],  # tau昇順、fitnessはdescending相当、GRPは降順
        )
        d["Ranking"] = d.groupby(tau_col).cumcount() + 1
    else:
        # 従来のrank
        d["Ranking"] = (
            d.groupby(tau_col)[value_col]
             .rank(method=rank_method, ascending=ascending)
             .astype(int)
        )

    # top_k（どこかのtauでtop_kに入ったentityだけ残す）
    if top_k is not None:
        keep_entities = d.loc[d["Ranking"] <= top_k, entity_col].drop_duplicates().tolist()
        d = d[d[entity_col].isin(keep_entities)]

    # 描画用にtau順序と並びを整形
    d[tau_col] = pd.Categorical(d[tau_col], categories=tau_order, ordered=True)
    d = d.sort_values([entity_col, tau_col])

    # hover用に列名を統一
    d = d.rename(columns={entity_col: "Entity", tau_col: "Tau", value_col: "Value"})
    if tie_break_col is not None:
        d = d.rename(columns={tie_break_col: "TieBreak"})
    return d
fig = create_fitness_bump_chart(
    panel_df.reset_index(drop=False),
    top_k=None,
    title="Fitness bump chart by prefecture",
)
fig.show()

#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional, Literal

REGION7_MAP = {
    # 北日本（北海道 + 東北）
    "北海道": "北海道・東北",
    "青森県": "北海道・東北", "岩手県": "北海道・東北", "宮城県": "北海道・東北",
    "秋田県": "北海道・東北", "山形県": "北海道・東北", "福島県": "北海道・東北",

    # 関東
    "茨城県": "関東", "栃木県": "関東", "群馬県": "関東",
    "埼玉県": "関東", "千葉県": "関東",
    "東京都": "関東", "神奈川県": "関東",

    # 中部
    "新潟県": "中部", "富山県": "中部", "石川県": "中部", "福井県": "中部",
    "山梨県": "中部", "長野県": "中部", "岐阜県": "中部",
    "静岡県": "中部", "愛知県": "中部",

    # 近畿
    "三重県": "近畿", "滋賀県": "近畿", "京都府": "近畿",
    "大阪府": "近畿", "兵庫県": "近畿", "奈良県": "近畿", "和歌山県": "近畿",

    # 中国
    "鳥取県": "中国", "島根県": "中国", "岡山県": "中国",
    "広島県": "中国", "山口県": "中国",

    # 四国
    "徳島県": "四国", "香川県": "四国", "愛媛県": "四国", "高知県": "四国",

    # 九州・沖縄
    "福岡県": "九州・沖縄", "佐賀県": "九州・沖縄", "長崎県": "九州・沖縄",
    "熊本県": "九州・沖縄", "大分県": "九州・沖縄",
    "宮崎県": "九州・沖縄", "鹿児島県": "九州・沖縄", "沖縄県": "九州・沖縄",
}

REGION7_PALETTE = {
    "北海道・東北":   "rgba(76, 114, 176, 0.60)",   # 青（北海道+東北）
    "関東":     "rgba(201, 162, 39, 0.75)",   # ゴールド（主役）
    "中部":     "rgba(85, 168, 104, 0.60)",   # 緑
    "近畿":     "rgba(214, 39, 40, 0.70)",    # ★ #d62728 + alpha
    "中国":     "rgba(140, 86, 75, 0.55)",    # ブラウン
    "四国":     "rgba(204, 121, 167, 0.55)",  # マゼンタ
    "九州・沖縄": "rgba(127, 127, 127, 0.45)",  # グレー
}


def add_region7_column(df: pd.DataFrame, prefecture_col: str = "prefecture") -> pd.DataFrame:
    out = df.copy()
    out["region7"] = out[prefecture_col].map(REGION7_MAP).fillna("その他")
    return out



# ======================
# 2) region7で色分けするtraces
# ======================
def add_entity_traces_region(
    fig: go.Figure,
    df_bump: pd.DataFrame,
    tau_order: List,
    *,
    region_col: str = "region7",
    palette: Dict[str, str] = REGION7_PALETTE,
    marker_size: int = 7,
    line_width: int = 2,
    label_buffer: float = 0.35,
    show_region_legend: bool = True,
):
    entities = df_bump["Entity"].unique().tolist()

    # region凡例を1回だけ出すためのフラグ
    region_legend_shown = set()

    for ent in entities:
        ent_data = df_bump[df_bump["Entity"] == ent]
        if ent_data.empty:
            continue

        region = ent_data[region_col].iloc[0] if region_col in ent_data.columns else "その他"
        color = palette.get(region, "#333333")

        last_point = ent_data.iloc[-1]
        last_tau_index = tau_order.index(last_point["Tau"])

        # regionごとにlegendは1回だけ
        showlegend = False
        legend_name = region
        if show_region_legend and (region not in region_legend_shown):
            showlegend = True
            region_legend_shown.add(region)

        fig.add_trace(
            go.Scatter(
                x=ent_data["Tau"],
                y=ent_data["Ranking"],
                mode="lines+markers",
                name=legend_name if showlegend else ent,  # 凡例に出すときだけregion名
                legendgroup=region,                       # regionでグルーピング
                showlegend=showlegend,
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size),
                customdata=ent_data[["Value", region_col]] if region_col in ent_data.columns else ent_data[["Value"]],
                hovertemplate=(
                    "<b>Prefecture:</b> " + ent +
                    "<br><b>Region:</b> %{customdata[1]}" if region_col in ent_data.columns else ""
                ) + (
                    "<br><b>Tau:</b> %{x}"
                    "<br><b>Rank:</b> %{y}"
                    "<br><b>Fitness:</b> %{customdata[0]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        # 右側の県名ラベル（色はregion色）
        fig.add_annotation(
            x=last_tau_index + label_buffer,
            y=last_point["Ranking"],
            text=ent,
            showarrow=False,
            font=dict(size=13, color=color),
            xanchor="left",
        )


# ======================
# 3) region8色分け版のcreate関数
# ======================
def create_fitness_bump_chart_region7(
    df: pd.DataFrame,
    *,
    entity_col: str = "prefecture",
    tau_col: str = "tau",
    fitness_col: str = "fitness",
    tie_break_col: Optional[str] = "GRP",
    rank_method: Literal["dense", "min", "average", "first", "max"] = "dense",
    top_k: Optional[int] = None,
    title: Optional[str] = None,
):
    # region8列を付与
    df2 = add_region7_column(df, prefecture_col=entity_col)

    # あなたの（後半の）prepare_bump_dataを使う想定：region8も残すため一旦マージ
    df_bump = prepare_bump_data(
        df2,
        entity_col=entity_col,
        tau_col=tau_col,
        value_col=fitness_col,
        tie_break_col=tie_break_col,
        rank_method=rank_method,
        ascending=False,
        top_k=top_k,
    )

    # df_bump には Entity/Tau しか残らないので region8 を付与し直す（Entity単位でOK）
    ent2region = df2[[entity_col, "region7"]].drop_duplicates().rename(columns={entity_col: "Entity"})
    df_bump = df_bump.merge(ent2region, on="Entity", how="left")

    tau_order = list(df_bump["Tau"].cat.categories)

    fig = go.Figure()

    add_entity_traces_region(
        fig,
        df_bump,
        tau_order,
        region_col="region7",
        palette=REGION7_PALETTE,
        marker_size=7,
        line_width=2,
        label_buffer=0.35,
        show_region_legend=True,
    )

    # 既存の customize_layout を流用（ただし凡例を出すので showlegend=True に上書き）
    customize_layout(
        fig,
        tau_order=tau_order,
        title=title,
        subtitle="Ranking changes by fitness across tau (colored by 7 regions)",
        footer="Note: Rank is computed within each tau. Ties in fitness are broken by GRP.",
    )
    fig.update_layout(showlegend=True)  # ★region凡例を表示

    # 凡例順序を固定したい場合（Plotlyは完全には保証しないが、概ね効く）
    fig.update_layout(legend=dict(title="Region (7)", traceorder="normal"))

    return fig

fig = create_fitness_bump_chart_region7(
    panel_df.reset_index(drop=False),
    top_k=None,
    title="Fitness bump chart by prefecture (7 regions)",
)
fig.show()

# %%
corr_df = panel_df.reset_index(drop=False)\
                  .filter(items=['prefecture', 'tau', 'fitness', 'g5_bar_pc_yen'])
corr_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =========================
# 相関 + 99% CI を計算
# =========================
def corr_with_ci(df: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    records = []

    z_crit = 2.576  # 99% CI

    for tau, g in df.groupby("tau"):
        g = g.dropna(subset=["fitness", "g5_bar_pc_yen"])
        n = len(g)

        if n < 4:
            continue

        r, _ = pearsonr(g["fitness"], g["g5_bar_pc_yen"])

        # Fisher z
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)

        z_low = z - z_crit * se
        z_high = z + z_crit * se

        records.append({
            "tau": tau,
            "corr": r,
            "ci_low": np.tanh(z_low),
            "ci_high": np.tanh(z_high),
            "n": n
        })

    return pd.DataFrame(records).sort_values("tau")


# =========================
# データ準備
# =========================
corr_df = (
    panel_df.reset_index(drop=False)
             .loc[:, ["prefecture", "tau", "fitness", "g5_bar_pc_yen"]]
)

corr_ts = corr_with_ci(corr_df)

# =========================
# plot
# =========================
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    corr_ts["tau"],
    corr_ts["corr"],
    label="Correlation",
    linewidth=2
)

ax.fill_between(
    corr_ts["tau"],
    corr_ts["ci_low"],
    corr_ts["ci_high"],
    alpha=0.25,
    label="99% CI"
)

ax.axhline(0, linestyle="--", linewidth=1)

ax.set_xlabel("tau")
ax.set_ylabel("Correlation (fitness vs g5_bar_pc_yen)")
ax.set_title("Cross-sectional correlation by tau (99% CI)")
ax.legend()

plt.tight_layout()
plt.show()

# %%
binned_scatter_df = panel_df.reset_index(drop=False)\
                            .filter(items=['prefecture', 'tau', 'fitness', 'g5_bar_pc_yen'])
binned_scatter_df
# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(binned_scatter_df['fitness'], binned_scatter_df['g5_bar_pc_yen'])
ax.set_xlabel('Fitness')
ax.set_ylabel('g5_bar_pc_yen')
ax.set_title('Binned scatter plot of fitness vs g5_bar_pc_yen')
# ax.set_xscale('log')
plt.show()
# %%
low_fitness_threshold = binned_scatter_df['fitness'].quantile(0.25)
high_fitness_threshold = binned_scatter_df['fitness'].quantile(0.75)

high_fitness_df = binned_scatter_df.query('fitness > @high_fitness_threshold')
mid_fitness_df = binned_scatter_df.query('fitness > @low_fitness_threshold and fitness < @high_fitness_threshold')
low_fitness_df = binned_scatter_df.query('fitness < @low_fitness_threshold')

high_fitness_corr = round(high_fitness_df['fitness'].corr(high_fitness_df['g5_bar_pc_yen']), 3)
mid_fitness_corr = round(mid_fitness_df['fitness'].corr(mid_fitness_df['g5_bar_pc_yen']), 3)
low_fitness_corr = round(low_fitness_df['fitness'].corr(low_fitness_df['g5_bar_pc_yen']), 3)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axes[2].scatter(high_fitness_df['fitness'], high_fitness_df['g5_bar_pc_yen'])
axes[2].set_xlabel('Fitness')
axes[0].set_ylabel('g5_bar_pc_yen')
axes[2].set_title(f'High fitness {high_fitness_corr} ')
axes[1].scatter(mid_fitness_df['fitness'], mid_fitness_df['g5_bar_pc_yen'])
axes[1].set_xlabel('Fitness')
axes[1].set_title(f'Mid fitness {mid_fitness_corr} ')
axes[0].scatter(low_fitness_df['fitness'], low_fitness_df['g5_bar_pc_yen'])
axes[0].set_xlabel('Fitness')
axes[0].set_title(f'Low fitness {low_fitness_corr} ')
axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[2].set_yscale('log')
plt.show()

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 0) 必要列だけ取り出し
# ======================
df = (
    panel_df.reset_index(drop=False)
    .filter(items=["prefecture", "tau", "fitness", "g5_bar_pc_yen", "GRP_yen"])
    .dropna(subset=["tau", "fitness", "g5_bar_pc_yen", "GRP_yen"])
    .copy()
)

# ======================
# 1) tau内で GRP 水準を Low/Mid/High に分割（25-50-25）
# ======================
def to_grp_group_within_tau(s: pd.Series) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return pd.cut(
        s,
        bins=[-np.inf, q1, q3, np.inf],
        labels=["Low GRP", "Mid GRP", "High GRP"],
        include_lowest=True,
    )

# df["grp_group"] = df.groupby("tau")["grp_fitness_1000"].transform(to_grp_group_within_tau)
df["grp_group"] = to_grp_group_within_tau(df["GRP_yen"])

# 念のため順序を固定
grp_order = ["Low GRP", "Mid GRP", "High GRP"]
df["grp_group"] = pd.Categorical(df["grp_group"], categories=grp_order, ordered=True)

# ======================
# 2) 3パネル scatter + OLS回帰線（傾き比較が主目的）
# ======================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

for ax, grp in zip(axes, grp_order):
    sub = df[df["grp_group"] == grp].copy()

    # scatter（分布感を残すため薄め）
    ax.scatter(sub["fitness"], sub["g5_bar_pc_yen"], s=18, alpha=0.25)

    # OLS直線（numpy polyfitで十分）
    x = sub["fitness"].to_numpy()
    y = sub["g5_bar_pc_yen"].to_numpy()

    # xが定数に近いと直線が引けないので保険
    if np.nanstd(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)  # y = b1*x + b0
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = b1 * x_line + b0
        ax.plot(x_line, y_line, linewidth=3)

        ax.set_title(f"{grp} (slope={b1:.3g}, n={len(sub)})")
    else:
        ax.set_title(f"{grp} (n={len(sub)})")

    ax.set_xlabel("Fitness")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Next 5y GRP pc growth (g5_bar_pc_yen)")
fig.suptitle("Fitness effect on next 5y GRP growth by GRP level (binned within tau)", y=1.02)
fig.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 0) 必要列
# ======================
df = (
    panel_df.reset_index(drop=False)
    .filter(items=["prefecture", "tau", "fitness", "g5_bar_pc_yen", "GRP_yen"])
    .dropna(subset=["tau", "fitness", "g5_bar_pc_yen", "GRP_yen"])
    .copy()
)

# ======================
# 1) tau内で GRP を 5 分割
# ======================
labels = ["Low", "Mid-Low", "Mid", "Mid-High", "High"]

def to_grp_quintile(s: pd.Series) -> pd.Series:
    return pd.qcut(
        s,
        q=5,
        labels=labels,
        duplicates="drop"
    )

df["grp_group"] = df.groupby("tau")["GRP_yen"].transform(to_grp_quintile)
df["grp_group"] = pd.Categorical(df["grp_group"], categories=labels, ordered=True)

# ======================
# 2) 5パネル scatter + OLS 傾き
# ======================
fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)

for ax, grp in zip(axes, labels):
    sub = df[df["grp_group"] == grp]

    ax.scatter(
        sub["fitness"],
        sub["g5_bar_pc_yen"],
        s=18,
        alpha=0.25
    )

    x = sub["fitness"].to_numpy()
    y = sub["g5_bar_pc_yen"].to_numpy()

    if np.nanstd(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, b1 * x_line + b0, linewidth=3)

        ax.set_title(f"{grp}\n slope={b1:.3f}, n={len(sub)}\n corr={round(sub['fitness'].corr(sub['g5_bar_pc_yen']), 3)}")
    else:
        ax.set_title(f"{grp}\n n={len(sub)}")

    ax.set_xlabel("Fitness")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Next 5y GRP pc growth (g5_bar_pc_yen)")
fig.suptitle(
    "Fitness effect on next 5y GRP growth by GRP level (quintiles within tau)",
    y=1.03
)

fig.tight_layout()
plt.show()

# %%
rank_df = panel_df.reset_index(drop=False)\
                  .filter(items=['prefecture', 'tau', 'fitness', 'g5_bar_pc_yen', 'grp_fitness_1000'])\
                  .assign(
                      rank_fitness = lambda x: x.groupby('tau')['fitness'].rank(method='min',
                                                                                ascending=False),
                      g,
                  )
rank_df
for tau, group in rank_df.groupby('tau'):
    plt.scatter(group['rank_fitness'], group['rank_g5_bar_pc_yen'])
    plt.title(f'tau={tau}')
    plt.show()
# %%
grp_rolling_df = pd.read_csv(
    '../../data/processed/external/grp/grp_capita.csv',
    encoding='utf-8',
    sep=',',
    )\
    .sort_values(by=['prefecture', 'year'], ascending=True, ignore_index=True)\
    .assign(
        grp_1000yen = lambda x: x['GRP'] / 1000,
        capita_density = lambda x: x['capita'] / x['area'],
        ln_capita_density = lambda x: np.log(x['capita_density']),
        ln_GRP = lambda x: np.log(x['GRP']),
        ln_GRP_t5 = lambda x: x.groupby('prefecture')['ln_GRP'].shift(-window_size),
        g5_bar = lambda x: (x['ln_GRP_t5'] - x['ln_GRP'])/window_size,
        ln_GRP_pc_yen = lambda x: np.log(x['GRP_per_capita_yen']), 
        ln_GRP_pc_yen_t5 = lambda x: x.groupby('prefecture')['ln_GRP_pc_yen'].shift(-window_size),
        g5_bar_pc_yen = lambda x: (x['ln_GRP_pc_yen_t5'] - x['ln_GRP_pc_yen'])/window_size,
        ln_capita = lambda x: np.log(x['capita']),
        ln_g5_bar = lambda x: np.log1p(x['g5_bar']),
        ln_g5_bar_pc_yen = lambda x: np.log1p(x['g5_bar_pc_yen']),
    )\
    .rename(columns={'year': 'tau'})\
    .drop_duplicates(keep='first', ignore_index=True)\
    .query('(1981 <= tau <= 2015)', engine='python')
grp_rolling_df


# %%
fitness_rolling_df = pref_df.copy()\
                    .assign(
                        tau = lambda x: x['app_nendo_period'].str[-4:].astype(np.int64),
                        ln_patent_count = lambda x: np.log1p(x['patent_count']), 
                        ln_fitness = lambda x: np.log(x['fitness']),
                        ln_mcp = lambda x: np.log(x['mpc']),
                        z_fitness = lambda x: (x['fitness'] - x['fitness'].mean()) / x['fitness'].std(),
                        fitness_lag1 = lambda x: x.groupby('prefecture')['fitness'].shift(1),
                        fitness_lag2 = lambda x: x.groupby('prefecture')['fitness'].shift(2),
                        fitness_lag3 = lambda x: x.groupby('prefecture')['fitness'].shift(3),
                        fitness_lag4 = lambda x: x.groupby('prefecture')['fitness'].shift(4),
                        fitness_lag5 = lambda x: x.groupby('prefecture')['fitness'].shift(5),
                        fitness_lead1 = lambda x: x.groupby('prefecture')['fitness'].shift(-1),
                        fitness_lead2 = lambda x: x.groupby('prefecture')['fitness'].shift(-2),
                        fitness_lead3 = lambda x: x.groupby('prefecture')['fitness'].shift(-3),
                        fitness_lead4 = lambda x: x.groupby('prefecture')['fitness'].shift(-4),
                        fitness_lead5 = lambda x: x.groupby('prefecture')['fitness'].shift(-5),
                    )
panel_rolling_df = pd.merge(
    grp_rolling_df,
    fitness_rolling_df,
    on=['prefecture', 'tau'],
    how='inner'
    )\
    .set_index(['prefecture', 'tau'])\
    .assign(
        grp_fitness_1000 = lambda x: x['fitness'] * x['GRP'] /1000,
        grp_fitness_100million = lambda x: x['fitness'] * x['GRP'] /100_000_000,
        pop_fitness = lambda x: x['fitness'] * x['capita_density'],
        patent_fitness = lambda x: x['fitness'] * x['patent_count'],
        ln_grp_fitness = lambda x: x['fitness'] * x['ln_GRP'],
        ln_pop_fitness = lambda x: x['fitness'] * x['ln_capita_density'],
        ln_patent_fitness = lambda x: x['fitness'] * x['ln_patent_count'],
    )
panel_rolling_df
# %%
panel_rolling_df.reset_index(drop=False)\
                .filter(items=['prefecture', 'tau', 'fitness', 'g5_bar_pc_yen', 'grp_1000yen'])


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BinnedScatterConfig:
    n_bins: int = 10
    by_tau: bool = True                 # True: tauごとにビン分け。False: 全期間で一括ビン分け
    bin_method: Literal["qcut", "cut"] = "qcut"  # qcut: 分位, cut: 等間隔
    min_obs_per_bin: int = 3            # ビン内サンプルが少なすぎる点は落とす
    add_errorbar: bool = True           # Trueなら平均の標準誤差をエラーバーで表示


def make_binned_scatter_df(
    df: pd.DataFrame,
    tau_col: str = "tau",
    x_col: str = "fitness",
    y_col: str = "g5_bar_pc_yen",
    group_col: str = "prefecture",
    cfg: BinnedScatterConfig = BinnedScatterConfig(),
) -> pd.DataFrame:
    """Create binned-scatter summary dataframe.

    Args:
        df: Panel dataframe containing tau, x, y, group columns.
        tau_col: Time-window identifier (e.g., rolling window index).
        x_col: Predictor variable (Fitness).
        y_col: Outcome variable (future GRPpc growth, etc.).
        group_col: Entity column (prefecture).
        cfg: Configuration for binning and summarization.

    Returns:
        A dataframe with columns:
            - tau (optional if cfg.by_tau=False)
            - bin_id
            - x_mean, y_mean
            - n
            - y_se (optional)
            - x_q_low, x_q_high (bin edges for interpretability when possible)
    """
    work = df[[c for c in [tau_col, group_col, x_col, y_col] if c in df.columns]].copy()

    # 欠損除外（最小限）
    work = work.dropna(subset=[x_col, y_col])
    if cfg.by_tau and tau_col not in work.columns:
        raise ValueError(f"cfg.by_tau=True but tau_col='{tau_col}' not in df.")
    if group_col not in work.columns:
        # group_col は必須ではないが、通常あるのでチェック
        pass

    def _assign_bins(s: pd.Series) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        # qcut は同値が多いと bin が潰れるので duplicates="drop" を使う
        if cfg.bin_method == "qcut":
            b = pd.qcut(s, q=cfg.n_bins, labels=False, duplicates="drop")
            # ビン境界も取れると嬉しいので、可能なら interval を作る
            try:
                intervals = pd.qcut(s, q=cfg.n_bins, duplicates="drop")
                edges = pd.DataFrame({
                    "bin_id": intervals.cat.codes,
                    "x_q_low": intervals.astype(str).str.extract(r"\((.*),")[0].astype(float),
                    "x_q_high": intervals.astype(str).str.extract(r",(.*)\]")[0].astype(float),
                }).drop_duplicates(subset=["bin_id"]).sort_values("bin_id")
            except Exception:
                edges = None
            return b, edges
        else:
            b = pd.cut(s, bins=cfg.n_bins, labels=False, duplicates="drop")
            return b, None

    if cfg.by_tau:
        out_list = []
        for tau, g in work.groupby(tau_col, sort=True):
            g = g.copy()
            g["bin_id"], edges = _assign_bins(g[x_col])

            agg = (
                g.groupby("bin_id", dropna=True)
                 .agg(
                     x_mean=(x_col, "mean"),
                     y_mean=(y_col, "mean"),
                     n=(y_col, "size"),
                     y_std=(y_col, "std"),
                 )
                 .reset_index()
            )
            if cfg.add_errorbar:
                agg["y_se"] = agg["y_std"] / np.sqrt(agg["n"])
            agg = agg.drop(columns=["y_std"])

            # サンプル数が少ないビンを落とす
            agg = agg[agg["n"] >= cfg.min_obs_per_bin].copy()
            agg[tau_col] = tau

            # ビン境界を付加（取れた場合のみ）
            if edges is not None:
                agg = agg.merge(edges, on="bin_id", how="left")

            out_list.append(agg)

        out = pd.concat(out_list, ignore_index=True) if out_list else pd.DataFrame()
        return out

    # 全期間で一括ビン分け
    work = work.copy()
    work["bin_id"], edges = _assign_bins(work[x_col])

    out = (
        work.groupby("bin_id", dropna=True)
            .agg(
                x_mean=(x_col, "mean"),
                y_mean=(y_col, "mean"),
                n=(y_col, "size"),
                y_std=(y_col, "std"),
            )
            .reset_index()
    )
    if cfg.add_errorbar:
        out["y_se"] = out["y_std"] / np.sqrt(out["n"])
    out = out.drop(columns=["y_std"])
    out = out[out["n"] >= cfg.min_obs_per_bin].copy()
    if edges is not None:
        out = out.merge(edges, on="bin_id", how="left")
    return out


def plot_binned_scatter(
    bdf: pd.DataFrame,
    tau_col: str = "tau",
    cfg: BinnedScatterConfig = BinnedScatterConfig(),
    title: str = "Binned scatter: Fitness (t) vs Future GRPpc growth (t+1)",
    xlabel: str = "Mean Fitness within bin",
    ylabel: str = "Mean future GRPpc growth within bin",
) -> plt.Axes:
    """Plot binned scatter.

    If cfg.by_tau=True, tauごとに薄い点（または色分け）にして重ね描きします。
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if cfg.by_tau and tau_col in bdf.columns:
        # tauごとに重ね描き（凡例がうるさければコメントアウト）
        for tau, g in bdf.groupby(tau_col, sort=True):
            if cfg.add_errorbar and "y_se" in g.columns:
                ax.errorbar(g["x_mean"], g["y_mean"], yerr=g["y_se"], fmt="o", capsize=2, alpha=0.7)
            else:
                ax.scatter(g["x_mean"], g["y_mean"], alpha=0.7)
    else:
        if cfg.add_errorbar and "y_se" in bdf.columns:
            ax.errorbar(bdf["x_mean"], bdf["y_mean"], yerr=bdf["y_se"], fmt="o", capsize=3)
        else:
            ax.scatter(bdf["x_mean"], bdf["y_mean"])

    ax.grid(True, alpha=0.3)
    return ax


# =========================
# 使い方（あなたのdfから）
# =========================
df_rolling = (
    panel_rolling_df.reset_index(drop=False)
    .filter(items=["prefecture", "tau", "fitness", "g5_bar_pc_yen", "coherence"])
)

cfg = BinnedScatterConfig(
    n_bins=10,
    by_tau=True,          # まずは tauごとに分位ビン
    bin_method="qcut",
    min_obs_per_bin=3,
    add_errorbar=True,
)

binned_df = make_binned_scatter_df(
    df_rolling,
    tau_col="tau",
    x_col="fitness",
    y_col="g5_bar_pc_yen",   # ←「向こう5年のGRPpc成長率」の列名に合わせる
    group_col="prefecture",
    cfg=cfg,
)

ax = plot_binned_scatter(
    binned_df,
    tau_col="tau",
    cfg=cfg,
    title="Binned scatter by tau: Fitness (t) vs GRPpc growth (t+1)",
    xlabel="Mean Fitness (bin average)",
    ylabel="Mean GRPpc per capita growth in next 5-year window",
)
plt.show()

# %%
cfg = BinnedScatterConfig(
    n_bins=10,
    by_tau=False,      # ← ここだけ変える
    bin_method="qcut",
    min_obs_per_bin=10,
    add_errorbar=True,
)

binned_df = make_binned_scatter_df(df_rolling, x_col="fitness", y_col="g5_bar_pc_yen", cfg=cfg)
ax = plot_binned_scatter(
    binned_df,
    cfg=cfg,
    title="Binned scatter (pooled): Fitness (t) vs GRPpc growth (t+1)",
)
plt.show()

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def _mean_ci(y: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Compute mean and (1-alpha) CI using normal approximation: mean ± z * SE.

    Args:
        y: 1D array of observations.
        alpha: Significance level. alpha=0.05 -> 95% CI.

    Returns:
        (mean, ci_low, ci_high)
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = y.size
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(y))
    if n == 1:
        return m, m, m

    s = float(np.std(y, ddof=1))
    se = s / np.sqrt(n)

    # z for common alphas without scipy
    # 90%: 1.645, 95%: 1.96, 99%: 2.576
    if np.isclose(alpha, 0.10):
        z = 1.645
    elif np.isclose(alpha, 0.05):
        z = 1.96
    elif np.isclose(alpha, 0.01):
        z = 2.576
    else:
        # fallback: approximate via 1.96
        z = 1.96

    return m, m - z * se, m + z * se


def _binned_stats_by_quantile(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Bin by quantiles of x_col, then compute:
      - x_mean (within bin)
      - y_mean and CI

    Args:
        df: DataFrame.
        x_col: Column to define quantile bins.
        y_col: Target column for mean/CI.
        n_bins: Number of quantile bins.
        alpha: CI alpha.

    Returns:
        DataFrame with columns:
          bin_idx, x_mean, y_mean, y_low, y_high, n
    """
    work = df[[x_col, y_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["bin_idx", "x_mean", "y_mean", "y_low", "y_high", "n"])

    # quantile bins (drop duplicates if too many ties)
    work["bin_idx"] = pd.qcut(work[x_col], q=n_bins, labels=False, duplicates="drop")
    g = work.groupby("bin_idx", as_index=False)

    rows = []
    for b, sub in g:
        x_mean = float(sub[x_col].mean())
        y_mean, y_low, y_high = _mean_ci(sub[y_col].to_numpy(), alpha=alpha)
        rows.append(
            {
                "bin_idx": int(b),
                "x_mean": x_mean,
                "y_mean": y_mean,
                "y_low": y_low,
                "y_high": y_high,
                "n": int(len(sub)),
            }
        )

    out = pd.DataFrame(rows).sort_values("bin_idx").reset_index(drop=True)
    return out


def plot_fitness_vs_growth_with_top_axis_grp(
    df: pd.DataFrame,
    fitness_col: str = "fitness",
    grp_col: str = "grp_1000yen",
    growth_col: str = "g5_bar_pc_yen",
    n_bins: int = 10,
    alpha: float = 0.05,
    log_base: str = "e",  # "e" or "10"
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Make a plot similar to the provided example:
      - bottom x-axis: log(Fitness)
      - top x-axis: log(GRP) (as coherence proxy)
      - y-axis: mean growth (g5_bar_pc_yen) with CI bands
      - two series: binned by fitness-quantiles vs binned by grp-quantiles
        but aligned by quantile position and drawn over the bottom axis.

    Args:
        df: Input DataFrame including fitness, grp, and growth.
        fitness_col: Column name for fitness.
        grp_col: Column name for GRP (coherence proxy).
        growth_col: Column name for 5-year forward growth.
        n_bins: Number of quantile bins.
        alpha: CI alpha (0.05 -> 95% CI).
        log_base: "e" for natural log, "10" for log10.
        figsize: Figure size.
        title: Optional title.

    Returns:
        Matplotlib Figure.
    """
    work = df[[fitness_col, grp_col, growth_col]].copy()

    # keep positive values for logs
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[fitness_col] > 0) & (work[grp_col] > 0)].copy()
    if work.empty:
        raise ValueError("No valid rows after filtering (need positive fitness and grp, and non-missing growth).")

    if log_base == "10":
        work["logF"] = np.log10(work[fitness_col].astype(float))
        work["logG"] = np.log10(work[grp_col].astype(float))
    else:
        work["logF"] = np.log(work[fitness_col].astype(float))
        work["logG"] = np.log(work[grp_col].astype(float))

    # binned stats separately
    stats_F = _binned_stats_by_quantile(work, x_col="logF", y_col=growth_col, n_bins=n_bins, alpha=alpha)
    stats_G = _binned_stats_by_quantile(work, x_col="logG", y_col=growth_col, n_bins=n_bins, alpha=alpha)

    if stats_F.empty or stats_G.empty:
        raise ValueError("Binning produced no results. Try reducing n_bins or check data distribution.")

    # Align by bin rank (quantile position). Use the smaller number of bins if duplicates were dropped.
    m = int(min(len(stats_F), len(stats_G)))
    stats_F = stats_F.iloc[:m].reset_index(drop=True)
    stats_G = stats_G.iloc[:m].reset_index(drop=True)

    # Use bottom axis x positions from fitness bins
    x_bottom = stats_F["x_mean"].to_numpy()

    # For top axis, show logG at the same quantile positions
    x_top_labels = stats_G["x_mean"].to_numpy()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Fitness-binned series (purple-ish default matplotlib color cycle will handle)
    ax.plot(x_bottom, stats_F["y_mean"], marker="o", label="log(Fitness)")
    ax.fill_between(x_bottom, stats_F["y_low"], stats_F["y_high"], alpha=0.25)

    # GRP-binned series aligned onto bottom x positions
    ax.plot(x_bottom, stats_G["y_mean"], marker="o", linestyle="--", label=f"log({grp_col})")
    ax.fill_between(x_bottom, stats_G["y_low"], stats_G["y_high"], alpha=0.25)

    ax.set_xlabel("log(Fitness)")
    ax.set_ylabel("Mean g5_bar_pc_yen")
    ax.axhline(work[growth_col].mean(), linestyle="--", linewidth=1)  # overall mean line

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="upper left")

    # Top axis for log(GRP)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    # Put a few ticks (not too dense)
    k = min(6, m)
    tick_idx = np.linspace(0, m - 1, k).round().astype(int)
    ax_top.set_xticks(x_bottom[tick_idx])
    ax_top.set_xticklabels([f"{v:.1f}" for v in x_top_labels[tick_idx]])
    ax_top.set_xlabel(f"log({grp_col})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ===== usage =====
df_in = df_rolling.reset_index(drop=False).filter(items=['prefecture','tau','fitness','g5_bar_pc_yen','coherence'])
fig = plot_fitness_vs_growth_with_top_axis_grp(
    df_in,
    fitness_col="fitness",
    grp_col="coherence",
    growth_col="g5_bar_pc_yen",
    n_bins=10,
    alpha=0.05,
    log_base="e",
    title=None,
)
plt.show()

# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


@dataclass(frozen=True)
class BinnedErrorbarResult:
    """Container for binned summary used for plotting."""
    df_summary: pd.DataFrame
    fig: plt.Figure
    ax: plt.Axes


def plot_binned_errorbar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    n_bins: int = 14,
    binning: Literal["quantile", "uniform"] = "quantile",
    ci: float = 0.95,
    min_n_per_bin: int = 5,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
) -> BinnedErrorbarResult:
    """
    Plot binned means with confidence-interval error bars (like the example image).

    Args:
        df: Input DataFrame.
        x_col: Column name for x-axis (continuous).
        y_col: Column name for y-axis.
        n_bins: Number of bins.
        binning: "quantile" makes bins with similar counts (おすすめ),
                 "uniform" makes equal-width bins.
        ci: Confidence level for the interval (e.g., 0.95).
        min_n_per_bin: Drop bins with fewer than this many observations.
        x_label: Label for x-axis. Defaults to x_col.
        y_label: Label for y-axis. Defaults to y_col.
        title: Optional title.

    Returns:
        BinnedErrorbarResult containing summary table and matplotlib objects.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing columns. need: {x_col}, {y_col}")

    d = df[[x_col, y_col]].dropna().copy()
    if len(d) == 0:
        raise ValueError("No data after dropping NA.")

    x = d[x_col].astype(float)
    y = d[y_col].astype(float)

    # ---- binning ----
    if binning == "quantile":
        # duplicates="drop" avoids errors when many identical x values exist
        d["bin"] = pd.qcut(x, q=n_bins, duplicates="drop")
    elif binning == "uniform":
        d["bin"] = pd.cut(x, bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'uniform'.")

    g = d.groupby("bin", observed=True)

    # ---- per-bin summary ----
    summary = g.agg(
        x_mean=(x_col, "mean"),
        y_mean=(y_col, "mean"),
        y_std=(y_col, "std"),
        n=(y_col, "size"),
    ).reset_index(drop=True)

    summary = summary[summary["n"] >= min_n_per_bin].copy()
    if len(summary) == 0:
        raise ValueError("All bins were dropped (min_n_per_bin too large or data too small).")

    # t-based CI around the mean
    alpha = 1.0 - ci
    summary["y_se"] = summary["y_std"] / np.sqrt(summary["n"])
    summary["t_crit"] = summary["n"].apply(lambda n: stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else np.nan)
    summary["y_err"] = summary["t_crit"] * summary["y_se"]

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        summary["x_mean"].to_numpy(),
        summary["y_mean"].to_numpy(),
        yerr=summary["y_err"].to_numpy(),
        fmt="o",
        markersize=6,
        capsize=5,
        elinewidth=2.5,
    )

    ax.set_xlabel(x_label or x_col, fontsize=22)
    ax.set_ylabel(y_label or y_col, fontsize=22)
    if title:
        ax.set_title(title, fontsize=18)

    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    # ax.set_xscale('log')
    return BinnedErrorbarResult(df_summary=summary, fig=fig, ax=ax)


# ======================
# 使い方（あなたの df_rolling から）
# ======================
df_plot = (
    df_rolling.reset_index(drop=False)
    .filter(items=["prefecture", "tau", "fitness", "g5_bar_pc_yen", "grp_1000yen"])\
    .assign(ln_fitness = lambda x: np.log(x['fitness']))
)

res = plot_binned_errorbar(
    df_plot,
    x_col="ln_fitness",
    y_col="g5_bar_pc_yen",
    n_bins=14,                 # 画像っぽい点の数
    binning="quantile",        # だいたい均等に点が並ぶのでおすすめ
    ci=0.95,
    min_n_per_bin=5,
    x_label="ln(Fitness)",
    y_label="GRPpc Growth (next 5y)",  # 好きに
)

plt.show()

# 必要なら集計表も確認できます
res.df_summary

#%%
panel_rolling_df.reset_index(drop=False)\
    .filter(items=['prefecture', 'tau', 'fitness', 'g5_bar_pc_yen', 'grp_1000yen'])\
    .to_clipboard(index=False)
# %%
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_fitness_and_future_growth_following(
    df: pd.DataFrame,
    *,
    tau_col: str = "tau",
    pref_col: str = "prefecture",
    fitness_col: str = "fitness",
    growth_col: str = "g5_bar_pc_yen",
    grp_col: str = "grp_1000yen",
    lead: int = 5,
    n_groups: int = 3,
    qcut_within_tau: bool = True,
    figsize: Tuple[float, float] = (12, 9),
    title: Optional[str] = None,
) -> plt.Figure:
    """Fitness の後を 5年先GRPpc成長率が追うように見せる最小構成プロット。

    手順:
      1) GRP水準 (grp_1000yen) を high/mid/low に分割（デフォルトは tau 内で分位分割）
      2) 各グループごとに tau ごとの平均系列（fitness と g5_bar_pc_yen）を作成
      3) 「追う」見え方のため、g5_bar_pc_yen を lead 年だけ右にシフトして (tau+lead) に配置
      4) 形状比較のため、各グループ内で z-score 標準化して同一軸に重ね描き

    Args:
        df: prefecture, tau, fitness, g5_bar_pc_yen, grp_1000yen を含むデータ。
        tau_col: 年度列。
        pref_col: 都道府県列（未使用だが存在チェック用）。
        fitness_col: Fitness列。
        growth_col: 5年先GRPpc成長率列。
        grp_col: GRP水準列（3分割に使用）。
        lead: 成長率系列を右にずらす年数（g5なら通常5）。
        n_groups: 分割数（3推奨）。
        qcut_within_tau: Trueなら tau 内で high/mid/low を作る。Falseなら全期間で一括分割。
        figsize: 図サイズ。
        title: 図タイトル。

    Returns:
        matplotlib Figure。
    """
    required = {tau_col, pref_col, fitness_col, growth_col, grp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    d = df[[pref_col, tau_col, fitness_col, growth_col, grp_col]].copy()
    d = d.dropna(subset=[tau_col, fitness_col, growth_col, grp_col])
    d[tau_col] = pd.to_numeric(d[tau_col], errors="coerce")
    d = d.dropna(subset=[tau_col])
    d[tau_col] = d[tau_col].astype(int)

    labels = ["low", "mid", "high"] if n_groups == 3 else [f"q{i+1}" for i in range(n_groups)]

    def _qcut_safe(s: pd.Series) -> pd.Series:
        # 同値が多くて qcut が落ちるケースに備えて rank で回避
        try:
            return pd.qcut(s, q=n_groups, labels=labels)
        except ValueError:
            return pd.qcut(s.rank(method="average"), q=n_groups, labels=labels)

    if qcut_within_tau:
        d["grp_level"] = d.groupby(tau_col)[grp_col].transform(_qcut_safe)
    else:
        d["grp_level"] = _qcut_safe(d[grp_col])

    # グループ×tau の平均系列を作る
    agg = (
        d.groupby(["grp_level", tau_col], observed=True)
        .agg(
            fitness_mean=(fitness_col, "mean"),
            growth_mean=(growth_col, "mean"),
            n=(pref_col, "nunique"),
        )
        .reset_index()
        .sort_values([ "grp_level", tau_col ])
    )

    # growth を右にずらして「fitness の後を追う」配置にする
    agg["tau_for_growth"] = agg[tau_col] + int(lead)

    # 図（3分割を縦に並べる）
    groups_in_data = [g for g in reversed(labels) if g in set(agg["grp_level"].astype(str))]
    nrows = len(groups_in_data)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True)
    if nrows == 1:
        axes = [axes]
    plt.rcParams['font.size'] = 17
    for ax, g in zip(axes, groups_in_data):
        sub = agg[agg["grp_level"].astype(str) == g].copy()
        if sub.empty:
            continue

        # fitness series (tau)
        s_fit = sub.set_index(tau_col)["fitness_mean"].sort_index()

        # growth series shifted (tau_for_growth)
        s_gro = sub.set_index("tau_for_growth")["growth_mean"].sort_index()

        # 形状比較のため z-score（平均0, 分散1）
        def _z(s: pd.Series) -> pd.Series:
            std = float(s.std(ddof=0))
            if std == 0.0 or np.isnan(std):
                return s * 0.0
            return (s - float(s.mean())) / std

        z_fit = _z(s_fit)
        z_gro = _z(s_gro)

        ax.plot(z_fit.index, z_fit.values, marker="o", linewidth=2.1, alpha=0.8,
                label="z-score of mean 5-year window Fitness")
        ax.plot(z_gro.index, z_gro.values, marker="o", linewidth=2.1, alpha=0.8,
                label=f"z-score of mean subsequent 5 years' GRPp.c. growth rate")
        ax.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=0.5, axis='x')
        # ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel(f"{g} {grp_col}", fontsize=21)
        ax.set_xticks(range(1985, 2020, 5))
        ax.set_xticklabels([f'{x}-{x+4}' for x in range(1981, 2015, 5)], rotation=90, 
                           fontsize=21)

        # 視認性のための注記（サンプル数の目安）
        ax.text(
            0.99,
            0.02,
            f"number of prefectures: {sub['n'].mean():.0f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
    axes[0].legend(loc="upper right")
    # axes[-1].set_xlabel(tau_col)
    fig.suptitle(title or f"Fitness and future GRPpc growth (shifted +{lead}) by GRP level", y=0.98)
    fig.tight_layout()
    return fig

fig = plot_fitness_and_future_growth_following(
    figsize=(12, 11),
    df=panel_rolling_df.reset_index(drop=False)\
    .filter(items=['prefecture', 'tau', 'fitness', 'g5_bar_pc_yen', 
                   'grp_1000yen', 'GRP_per_capita_1000yen'])\
    .rename(columns={'grp_1000yen': 'GRP'}),
    lead=0,
    grp_col='GRP',
    n_groups = 3,
    qcut_within_tau=True,  # tau内でhigh/mid/lowに分ける
)
plt.show()

# %%
