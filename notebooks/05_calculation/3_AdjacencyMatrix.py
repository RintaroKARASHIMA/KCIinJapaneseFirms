#! (root)/notebooks/3_calculate/1_AggregateWeight.py python3
# -*- coding: utf-8 -*-
#%%
import sys
import numpy as np
import pandas as pd
from IPython.display import display
from pathlib import Path
from scipy.stats import spearmanr

from ecomplexity import ecomplexity

sys.path.append(str(Path(__file__).resolve().parents[2]) + '/src/')
from initialize.config_loader import  load_filter_config, load_adj_config
from calculation import biadjm


agg_cfg = load_filter_config(str(Path(__file__).resolve().parents[2]) + '/config/reg_num_filter_agg.yaml')
cmp_cfg = load_adj_config(str(Path(__file__).resolve().parents[2]) + '/config/complexity.yaml')

#%%
agg_df = pd.read_csv(
    f'{agg_cfg.out_dir}{agg_cfg.out_file_name}.csv',
    encoding='utf-8',
    sep=',',
)

#%%
agg_df
#%%
# adj_df = pd.concat(
#     [biadjm.compute_pref_schmoch_lq(
#                              agg_df.query('period == @period'),
#                             producer_col = cmp_cfg.region_corporation,
#                             class_col = cmp_cfg.classification,
#                             count_col = 'weight',
#     ).assign(
#         period = lambda x: period
#     ) for period in agg_df.period.unique()],
#     axis='index',
#     ignore_index=True)
# adj_df
#%%
import numpy as np
import pandas as pd
from typing import Literal, List, Dict, Any


def compute_pref_schmoch_lq(
    df: pd.DataFrame,
    aggregate: Literal[True, False] = True,
    *,
    producer_col: str = "prefecture",
    class_col: str = "schmoch35",
    count_col: str = "weight",
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

#%%
compute_pref_schmoch_lq(agg_df, 
                        aggregate=True, 
                        producer_col=cmp_cfg.region_corporation, 
                        class_col=cmp_cfg.classification, 
                        count_col='weight')


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
        columns='schmoch35',
        values="mpc",
        aggfunc="sum",
        fill_value=0
    )
    fitness, complexity = Fitn_Comp(biadjm_presence.values)
    prefectures = biadjm_presence.index
    techs = biadjm_presence.columns
    result_df = pd.DataFrame(
        {
            'prefecture': prefectures,
            'fitness': fitness
        }
    ).sort_values(by='fitness', ascending=False, ignore_index=True)
    tech_result_df = pd.DataFrame(
        {
            'schmoch35': techs,
            'complexity': complexity
        }
    ).sort_values(by='complexity', ascending=False, ignore_index=True)
    return result_df, tech_result_df

#%%
mcp_df = compute_pref_schmoch_lq(agg_df, aggregate=True, producer_col=cmp_cfg.region_corporation, class_col=cmp_cfg.classification, 
                        count_col='weight')
f_df = define_mcp(mcp_df)[0]
c_df = define_mcp(mcp_df)[1]
c_df
#%%
adj_plot_df = pd.merge(mcp_df, f_df, on='prefecture', how='left')
adj_plot_df = pd.merge(adj_plot_df, c_df, on='schmoch35', how='left')\
                      .filter(items=['prefecture', 'schmoch35', 'mpc', 'fitness', 'complexity'])
adj_plot_df
schmoch35_df = pd.read_csv(
    '../../data/processed/external/schmoch/35.csv',
    encoding='utf-8',
    sep=',',
    ).filter(items=['Field_number', 'Field_en'])\
    .drop_duplicates(
        keep='first'
    )
schmoch35_df
adj_plot_df = pd.merge(adj_plot_df, schmoch35_df, left_on='schmoch35', right_on='Field_number', how='left')\
                      .drop(columns=['schmoch35', 'Field_number'])\
                      .rename(columns={'Field_en':'schmoch35'})
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_prefecture_schmoch_matrix(df: pd.DataFrame):
    """
    df: columns = ['prefecture', 'schmoch35', 'mpc', 'fitness', 'complexity']
    """

    # --------------------------
    # 1. prefecture をローマ字表記にする（例）
    # --------------------------
    # 辞書は必要に応じて追加してください
    romaji_map = {
        '北海道': 'Hokkaido', '青森県': 'Aomori', '岩手県': 'Iwate', '宮城県': 'Miyagi',
        '秋田県': 'Akita', '山形県': 'Yamagata', '福島県': 'Fukushima',
        '茨城県': 'Ibaraki', '栃木県': 'Tochigi', '群馬県': 'Gunma',
        '埼玉県': 'Saitama', '千葉県': 'Chiba', '東京都': 'Tokyo', '神奈川県': 'Kanagawa',
        '新潟県': 'Niigata', '富山県': 'Toyama', '石川県': 'Ishikawa', '福井県': 'Fukui',
        '山梨県': 'Yamanashi', '長野県': 'Nagano',
        '岐阜県': 'Gifu', '静岡県': 'Shizuoka', '愛知県': 'Aichi',
        '三重県': 'Mie', '滋賀県': 'Shiga', '京都府': 'Kyoto', '大阪府': 'Osaka',
        '兵庫県': 'Hyogo', '奈良県': 'Nara', '和歌山県': 'Wakayama',
        '鳥取県': 'Tottori', '島根県': 'Shimane', '岡山県': 'Okayama', '広島県': 'Hiroshima',
        '山口県': 'Yamaguchi',
        '徳島県': 'Tokushima', '香川県': 'Kagawa', '愛媛県': 'Ehime', '高知県': 'Kochi',
        '福岡県': 'Fukuoka', '佐賀県': 'Saga', '長崎県': 'Nagasaki',
        '熊本県': 'Kumamoto', '大分県': 'Oita', '宮崎県': 'Miyazaki', '鹿児島県': 'Kagoshima',
        '沖縄県': 'Okinawa'
    }

    df = df.copy()
    df['prefecture_romaji'] = df['prefecture'].map(romaji_map).fillna(df['prefecture'])

    # --------------------------
    # 2. prefecture を fitness 降順に
    # --------------------------
    order_pref = (
        df[['prefecture_romaji', 'fitness']]
        .drop_duplicates()
        .sort_values(by='fitness', ascending=False)
        ['prefecture_romaji']
    )

    # --------------------------
    # 3. schmoch35 を complexity 昇順に並べる
    # --------------------------
    order_schmoch = (
        df[['schmoch35', 'complexity']]
        .drop_duplicates()
        .sort_values(by='complexity')
        ['schmoch35']
        .tolist()
    )

    # --------------------------
    # 4. pivot（隣接行列 mpc）
    # --------------------------
    pivot_df = df.pivot_table(
        index='prefecture_romaji',
        columns='schmoch35',
        values='mpc',
        fill_value=0
    )

    # 並び順を適用
    pivot_df = pivot_df.loc[order_pref, order_schmoch]

    # --------------------------
    # 5. 描画
    # --------------------------
    plt.figure(figsize=(11, 13))
    plt.imshow(pivot_df, aspect='auto', cmap='Greys', interpolation='nearest')
    # plt.colorbar(label='mpc (0/1)')

    plt.xticks(
        ticks=np.arange(len(order_schmoch)),
        labels=order_schmoch,
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(len(order_pref)),
        labels=order_pref
    )

    plt.xlabel('Schmoch 35 Classes (sorted by complexity)')
    plt.ylabel('Prefecture (sorted by fitness)')
    plt.title('Prefecture × Schmoch35 Matrix (mpc adjacency)')

    plt.tight_layout()
    plt.show()


# --------------------------
# 使用例
# --------------------------
# plot_prefecture_schmoch_matrix(df)

plot_prefecture_schmoch_matrix(adj_plot_df)

#%%
# ここから技術分野ごとの特許数分布可視化
adj_plot_df.columns

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NestednessResult:
    """Container for nestedness (NODF) outputs."""
    nodf: float
    binary_matrix: pd.DataFrame  # rows=prefecture, cols=schmoch35, values in {0,1}
    rca_matrix: pd.DataFrame     # rows=prefecture, cols=schmoch35, RCA values
    sorted_binary_matrix: pd.DataFrame  # degree-sorted matrix used in NODF computation


def _compute_nodf_from_binary_matrix(A: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute NODF for a binary bipartite adjacency matrix.

    Args:
        A: Binary matrix of shape (R, C) with values 0/1.

    Returns:
        nodf: NODF in [0, 100].
        M: degree-sorted binary matrix used for computation.
        row_order: indices used to sort rows (descending degree).
        col_order: indices used to sort cols (descending degree).
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")
    M0 = (A > 0).astype(np.int8)

    row_deg = M0.sum(axis=1)
    col_deg = M0.sum(axis=0)

    # Stable sort for deterministic output when ties exist
    row_order = np.argsort(-row_deg, kind="mergesort")
    col_order = np.argsort(-col_deg, kind="mergesort")

    M = M0[row_order][:, col_order]
    R, C = M.shape

    row_deg_s = M.sum(axis=1)
    col_deg_s = M.sum(axis=0)

    row_sum = 0.0
    for i in range(R - 1):
        ki = row_deg_s[i]
        if ki == 0:
            continue
        for j in range(i + 1, R):
            kj = row_deg_s[j]
            if kj == 0:
                continue
            if ki > kj:
                overlap = int(np.dot(M[i], M[j]))
                row_sum += overlap / kj

    col_sum = 0.0
    for p in range(C - 1):
        kp = col_deg_s[p]
        if kp == 0:
            continue
        for q in range(p + 1, C):
            kq = col_deg_s[q]
            if kq == 0:
                continue
            if kp > kq:
                overlap = int(np.dot(M[:, p], M[:, q]))
                col_sum += overlap / kq

    n_row_pairs = R * (R - 1) / 2
    n_col_pairs = C * (C - 1) / 2
    denom = n_row_pairs + n_col_pairs

    nodf = 0.0 if denom == 0 else 100.0 * (row_sum + col_sum) / denom
    return nodf, M, row_order, col_order


def compute_nestedness_nodf(
    df: pd.DataFrame,
    *,
    prefecture_col: str = "prefecture",
    tech_col: str = "schmoch35",
    value_col: str = "mpc",
    rca_threshold: float = 1.0,
    drop_zero_rows_cols: bool = True,
) -> NestednessResult:
    """Compute nestedness (NODF) from a long-form dataframe with prefecture-tech values.

    Standard pipeline:
      1) Aggregate value_col by (prefecture, tech).
      2) Build a bipartite matrix X (prefecture x tech).
      3) Compute RCA:
            RCA_{p,t} = (X_{p,t} / sum_t X_{p,t}) / (sum_p X_{p,t} / sum_{p,t} X_{p,t})
         Then define adjacency as 1{RCA >= rca_threshold}.
      4) Compute NODF on the binary matrix.

    Args:
        df: DataFrame containing columns prefecture_col, tech_col, value_col.
        prefecture_col: Row entity column name.
        tech_col: Column entity column name.
        value_col: Weight/value column name (e.g., patent counts, mpc).
        rca_threshold: Threshold to binarize RCA into adjacency (default 1.0).
        drop_zero_rows_cols: If True, drop rows/cols that become all-zero after binarization.

    Returns:
        NestednessResult including NODF and matrices.
    """
    required = {prefecture_col, tech_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    work = df[[prefecture_col, tech_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[prefecture_col, tech_col, value_col])
    if work.empty:
        raise ValueError("No valid rows after dropping NaNs in required columns.")

    # 1) Aggregate
    agg = (
        work.groupby([prefecture_col, tech_col], as_index=False)[value_col]
        .sum()
    )

    # 2) Pivot to matrix X
    X = agg.pivot(index=prefecture_col, columns=tech_col, values=value_col).fillna(0.0)
    # If everything is zero, RCA is undefined
    total = float(X.to_numpy().sum())
    if total <= 0.0:
        raise ValueError("Total sum of the prefecture-tech matrix is zero. RCA cannot be computed.")

    row_sum = X.sum(axis=1)
    col_sum = X.sum(axis=0)

    # Guard against division by zero (rows/cols with zero activity)
    # We'll compute RCA only where row_sum>0 and col_sum>0; else RCA=0
    with np.errstate(divide="ignore", invalid="ignore"):
        # share within prefecture
        share_pt = X.div(row_sum.replace(0.0, np.nan), axis=0)
        # global tech share
        share_t = (col_sum / total).replace(0.0, np.nan)

        rca = share_pt.div(share_t, axis=1).fillna(0.0)

    # 3) Binarize by RCA threshold
    B = (rca >= float(rca_threshold)).astype(np.int8)

    if drop_zero_rows_cols:
        keep_rows = B.sum(axis=1) > 0
        keep_cols = B.sum(axis=0) > 0
        B = B.loc[keep_rows, keep_cols]
        rca = rca.loc[B.index, B.columns]

    if B.shape[0] < 2 or B.shape[1] < 2:
        # NODF is not meaningful with fewer than 2 rows/cols (pairs become too few)
        nodf = 0.0
        sorted_B = B.copy()
        return NestednessResult(
            nodf=nodf,
            binary_matrix=B,
            rca_matrix=rca,
            sorted_binary_matrix=sorted_B,
        )

    # 4) NODF computation
    A = B.to_numpy()
    nodf, M_sorted, row_order, col_order = _compute_nodf_from_binary_matrix(A)

    sorted_B = pd.DataFrame(
        M_sorted,
        index=B.index.to_numpy()[row_order],
        columns=B.columns.to_numpy()[col_order],
    )

    return NestednessResult(
        nodf=float(nodf),
        binary_matrix=B,
        rca_matrix=rca,
        sorted_binary_matrix=sorted_B,
    )


# -------------------------
# Example usage:
# -------------------------
result = compute_nestedness_nodf(adj_plot_df)
print("NODF:", result.nodf)
# display(result.sorted_binary_matrix.head())

#%%

import numpy as np
import pandas as pd

# 既に前の compute_nestedness_nodf を定義済みとして使う想定

def quick_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.5, 1.0, 1.5, 2.0]
    rows = []
    for th in thresholds:
        res = compute_nestedness_nodf(df, rca_threshold=th, drop_zero_rows_cols=True)
        B = res.binary_matrix
        R, C = B.shape
        ones = int(B.to_numpy().sum())
        density = float(ones / (R * C)) if R > 0 and C > 0 else np.nan
        rows.append(
            {
                "rca_threshold": th,
                "NODF": res.nodf,
                "R": R,
                "C": C,
                "ones": ones,
                "density": density,
            }
        )
    return pd.DataFrame(rows)

diag = quick_diagnostics(adj_plot_df)
print(diag)


#%%
trade_cols = {
    "time": "period",
    "loc": cmp_cfg.region_corporation,
    "prod": cmp_cfg.classification,
    "val": "weight",
}
c_df = ecomplexity(agg_df, trade_cols, rca_mcp_threshold=1)
c_df
# %%

#%%
# Update: remove trailing rank digits next to labels and ensure y-axis is reversed.
# - Added side-specific flags: `label_rank_left` and `label_rank_right` (default False).
# - y-axis is already reversed; keep it explicit.
from __future__ import annotations
from typing import Optional, Sequence, Mapping
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def _greedy_avoid_overlap(y_positions: np.ndarray, min_gap: float) -> np.ndarray:
    if len(y_positions) <= 1:
        return y_positions.copy()
    order = np.argsort(y_positions)
    y_sorted = y_positions[order].copy()
    for i in range(1, len(y_sorted)):
        if y_sorted[i] - y_sorted[i - 1] < min_gap:
            y_sorted[i] = y_sorted[i - 1] + min_gap
    for i in range(len(y_sorted) - 2, -1, -1):
        if y_sorted[i + 1] - y_sorted[i] < min_gap:
            y_sorted[i] = y_sorted[i + 1] - min_gap
    adjusted = np.empty_like(y_positions, dtype=float)
    adjusted[order] = y_sorted
    return adjusted

def _build_palette(colors_or_cmap: Optional[Sequence[str] | str], n: int) -> list[str]:
    """
    Create a list of n colors. If a list is provided, it will be cycled or truncated.
    If a string is provided, it's interpreted as a Matplotlib colormap name.
    Defaults to the first n colors of 'tab10'.
    """
    if colors_or_cmap is None:
        base = cm.get_cmap("tab10")
        return [base(i) for i in range(n)]
    if isinstance(colors_or_cmap, str):
        cmap = cm.get_cmap(colors_or_cmap)
        # Pick n evenly spaced colors
        return [cmap(i/(max(1, n-1))) for i in range(n)]
    # A list/tuple of hex or rgba
    colors = list(colors_or_cmap)
    if len(colors) >= n:
        return colors[:n]
    # cycle if shorter
    out = []
    i = 0
    while len(out) < n:
        out.append(colors[i % len(colors)])
        i += 1
    return out

def plot_bump_chart(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    rank_col: str,
    *,
    times: Optional[Sequence] = None,
    ax: Optional[plt.Axes] = None,
    linewidth: float = 2.5,
    marker: Optional[str] = "o",
    line_alpha: float = 0.95,
    label_fontsize: int = 10,
    label_pad_frac: float = 0.03,
    min_label_gap: float = 0.15,
    # Label controls
    left_labels: bool = True,
    right_labels: bool = True,
    label_rank_left: bool = False,
    label_rank_right: bool = False,
    # Style
    hide_spines: bool = True,
    show_ygrid: bool = True,
    ytick_multiple: int = 5,
    # NEW: color control
    ent_colors: Optional[Mapping[str, str]] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 15), dpi=150)
    if times is None:
        times = sorted(data[time_col].unique())
    time_to_x = {t: i for i, t in enumerate(times)}
    x_vals = np.arange(len(times), dtype=float)

    wide = (
        data[[entity_col, time_col, rank_col]]
        .dropna()
        .assign(_x=lambda d: d[time_col].map(time_to_x))
        .pivot_table(index=entity_col, columns="_x", values=rank_col)
        .reindex(columns=x_vals)
    )

    for ent, row in wide.iterrows():
        y = row.values.astype(float)
        isnan = np.isnan(y)
        if np.all(isnan):
            continue
        start = None
        color = ent_colors.get(ent) if ent_colors is not None else None
        for i in range(len(y)):
            if not isnan[i] and start is None:
                start = i
            if (isnan[i] or i == len(y) - 1) and start is not None:
                end = i if isnan[i] else i + 1
                ax.plot(
                    x_vals[start:end], y[start:end],
                    marker=marker, linewidth=linewidth, alpha=line_alpha,
                    color=color
                )
                start = None

    # y-axis (rank 1 is top) — set limits here; caller may still call invert_yaxis()
    max_rank = max(1, int(math.floor(np.nanmax(wide.values))))
    ax.set_ylim(0.5, max_rank + 0.5)

    # Only multiples of ytick_multiple
    if ytick_multiple is not None and ytick_multiple > 0:
        ticks = np.arange(0, max_rank + 1, ytick_multiple)
        ticks = ticks[(ticks >= 1) & (ticks <= max_rank)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(int(t)) for t in ticks])

    # X ticks
    ax.set_xlim(-0.5, len(times) - 0.5)
    ax.set_xticks(x_vals, labels=[str(t) for t in times], fontsize=18, rotation=90)

    if show_ygrid:
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    else:
        ax.grid(False)

    if hide_spines:
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    def _edge_labels(side: str) -> None:
        xi = 0 if side == "left" else len(times) - 1
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        x_pad = (-label_pad_frac if side == "left" else label_pad_frac) * x_span
        x_text = xi + x_pad

        ranks = []
        ents = []
        for ent, row in wide.iterrows():
            if not np.isnan(row.iloc[xi]):
                ranks.append(float(row.iloc[xi]))
                ents.append(ent)
        if not ranks:
            return
        y0 = np.array(ranks, dtype=float)
        y_adj = _greedy_avoid_overlap(y0, min_gap=min_label_gap)

        append_rank = label_rank_left if side == "left" else label_rank_right

        for ent, y_raw, y_lab in zip(ents, y0, y_adj):
            label = f"{ent} {int(y_raw)}" if (append_rank and not np.isnan(y_raw)) else f"{ent}"
            ha = "right" if side == "left" else "left"
            txt_color = ent_colors.get(ent) if ent_colors is not None else None
            ax.text(x_text, y_lab, label, va="center", ha=ha, fontsize=label_fontsize, color=txt_color)
            ax.plot([xi, x_text], [y_raw, y_lab], linewidth=1.0, alpha=0.5, color=txt_color)

    if left_labels:
        _edge_labels("left")
    if right_labels:
        _edge_labels("right")

    ax.set_ylabel("Rank")
    return ax

def bump_chart_from_pci(
    df: pd.DataFrame,
    *,
    period_col: str = "period",
    ipc_col: str = "ipc3",
    pci_col: str = "pci",
    ranking_col: Optional[str] = "ranking",
    times: Optional[Sequence] = None,
    top_k: Optional[int] = None,
    rank_method: str = "dense",
    hide_spines: bool = True,
    show_ygrid: bool = True,
    linewidth: float = 2.5,
    marker: Optional[str] = "o",
    label_fontsize: int = 10,
    min_label_gap: float = 0.15,
    ytick_multiple: int = 5,
    label_rank_left: bool = False,
    label_rank_right: bool = False,
    # NEW: group/color settings
    group_col: Optional[str] = None,              # e.g., "schmoch5"
    palette: Optional[Sequence[str] | str] = None # None→tab10先頭、または "tab10"/"Set2"/hexリスト等
) -> plt.Axes:
    d = df.copy()
    if (ranking_col is None) or (ranking_col not in d.columns):
        d["_rank"] = (
            d.groupby(period_col)[pci_col]
              .rank(method=rank_method, ascending=False)
              .astype(int)
        )
        use_rank_col = "_rank"
    else:
        use_rank_col = ranking_col
        d[use_rank_col] = pd.to_numeric(d[use_rank_col], errors="coerce")

    if top_k is not None:
        d = d[d[use_rank_col] <= int(top_k)].copy()

    # ---- build entity -> color (via group_col) ----
    ent_colors: Optional[dict[str, str]] = None
    if group_col is not None and group_col in d.columns:
        # 1) エンティティごとの代表グループ（最頻値）を決める
        def _mode_or_first(s: pd.Series):
            s = s.dropna()
            if s.empty:
                return np.nan
            m = s.mode()
            return m.iat[0] if not m.empty else s.iat[0]

        ent_to_group = (
            d.groupby(ipc_col)[group_col]
             .apply(_mode_or_first)
             .astype("category")
        )

        # 2) グループ→色を5色で割当て
        groups = list(ent_to_group.dropna().unique())
        n_groups = len(groups)
        if n_groups > 5:
            # 5色に収まらない場合は均等サンプリングで色を作る
            cols = _build_palette(palette or "tab10", n_groups)
        else:
            cols = _build_palette(palette or "tab10", max(1, n_groups or 5))[:n_groups]
        group_to_color = {g: c for g, c in zip(groups, cols)}

        # 3) エンティティ→色辞書
        ent_colors = {ent: group_to_color.get(grp) for ent, grp in ent_to_group.items()}

    tidy = d.rename(columns={period_col: "___period", ipc_col: "___ipc", use_rank_col: "___rank"})[
        ["___period", "___ipc", "___rank"]
    ].dropna()

    ax = plot_bump_chart(
        tidy.rename(columns={"___period": "year", "___ipc": "entity", "___rank": "rank"}),
        entity_col="entity",
        time_col="year",
        rank_col="rank",
        times=times,
        linewidth=linewidth,
        marker=marker,
        label_fontsize=label_fontsize,
        min_label_gap=min_label_gap,
        hide_spines=hide_spines,
        show_ygrid=show_ygrid,
        ytick_multiple=ytick_multiple,
        label_rank_left=label_rank_left,
        label_rank_right=label_rank_right,
        ent_colors=ent_colors,  # ← NEW
    )
    return ax


palette_s8 = {
    "C": "#FF0000",  # vermilion (red start)
    "A": "#E67E00",  # vivid orange
    "D": "#C8B100",  # deep amber (muted yellow)
    "B": "#2FA772",  # green
    "E": "#00A0B5",  # cyan / teal
    "F": "#3F8FD3",  # blue
    "G": "#6F66BD",  # indigo / blue-violet
    "H": "#999999",  # black
}

df_ranked = (
    c_df.assign(
        Ranking = c_df.groupby("period")["pci"].rank(ascending=False, method="dense")
    )
    .assign(Ranking=lambda d: d["Ranking"].astype(int))   # 見栄え用に整数化
).query('period != "1981-2010"')

df_income = (
    df_ranked.filter(items=['period', 'ipc3', 'pci', 'Ranking'])\
             .drop_duplicates()\
             .sort_values(['period', 'Ranking'], ignore_index=True)\
             .assign(
                 ipc1 = lambda x: x['ipc3'].str[:1],
             )
)
ax = bump_chart_from_pci(
    df_income,
    period_col="period",
    ipc_col="ipc3",
    pci_col="pci",
    ranking_col=None,
    ytick_multiple=5,
    label_rank_left=False,
    label_rank_right=False,
    hide_spines=True,
    group_col="ipc1",     # ← ここを指定
    palette=list(palette_s8.values()),          # ← 任意（Noneならtab10先頭から5色）
)
ax.invert_yaxis()  # 1位が上
plt.show()

# %%
