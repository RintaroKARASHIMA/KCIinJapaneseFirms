#! (root)/notebooks/05_calculation/3_Complexity.py python3
# -*- coding: utf-8 -*-
#%%
import sys
import numpy as np
import pandas as pd

from IPython.display import display
from pathlib import Path

from ecomplexity import ecomplexity

sys.path.append(str(Path(__file__).resolve().parents[2]) + '/src/')
from initialize.config_loader import  load_filter_config, load_adj_config
from calculation import biadjm
from visualize.bump_chart import *

adj_cfg = load_adj_config(str(Path(__file__).resolve().parents[2]) + '/config/complexity.yaml')

#%%
adj_df = pd.read_csv(
    f'{adj_cfg.in_dir}{adj_cfg.out_file_name}.csv',
    encoding='utf-8',
    sep=',',
)

# %%
trade_cols = {
    "time": "period",
    "loc": adj_cfg.region_corporation,
    "prod": adj_cfg.classification,
    "val": "weight",
}
c_df = ecomplexity(adj_df, trade_cols, rca_mcp_threshold=1)
c_df.filter(items=['period', adj_cfg.classification, 'pci'])
# %%
# %%
adj_df = c_df.query('period == "1981-2010"')\
        .filter(items=['corporation', 'schmoch35', 'mcp'])\
        .drop_duplicates()
adj_df.columns
schmoch35_df = pd.read_csv(
    f'../../data/processed/external/schmoch/35.csv',
    encoding='utf-8',
    sep=',',
)
schmoch35_df.columns
#%%
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


@dataclass(frozen=True)
class AdjMatrixPlotResult:
    """Return object for adjacency-matrix plotting."""
    mat: pd.DataFrame
    row_sums: pd.Series
    col_sums: pd.Series


def plot_adj_matrix_corp_schmoch35_bw(
    adj_df: pd.DataFrame,
    *,
    corp_col: str = "corporation",
    tech_col: str = "schmoch35",
    val_col: str = "mcp",
    figsize: Tuple[float, float] = (10, 10),
    dpi: int = 150,
    x_tick_max: int = 60,
    y_tick_max: int = 12,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> AdjMatrixPlotResult:
    """
    Plot adjacency matrix (rows=corporation, cols=schmoch35)
    with black-white colors and square cells.

    - 1 -> black, 0 -> white
    - columns sorted by number of 1s (descending, left is denser)
    - rows sorted by number of 1s (descending, top is denser)
    - x-axis: tick labels shown
    - y-axis: tick marks only (no labels)
    """

    # --- check columns ---
    required = {corp_col, tech_col, val_col}
    missing = required - set(adj_df.columns)
    if missing:
        raise ValueError(f"adj_df is missing columns: {sorted(missing)}")

    df = adj_df[[corp_col, tech_col, val_col]].copy()
    df[val_col] = (df[val_col].fillna(0).astype(float) > 0).astype(int)

    # --- pivot to adjacency matrix ---
    mat = (
        df.pivot_table(
            index=corp_col,
            columns=tech_col,
            values=val_col,
            aggfunc="max",
            fill_value=0,
        )
        .astype(int)
    )

    # --- sort columns and rows by density ---

    row_sums = mat.sum(axis=1).sort_values(ascending=False)
    mat = mat.loc[row_sums.index, :]
    
    col_sums = mat.sum(axis=0).sort_values(ascending=False)
    mat = mat.loc[:, col_sums.index]
    n_rows, n_cols = mat.shape

    # --- tick thinning ---
    def _pick_positions(n: int, max_ticks: int) -> np.ndarray:
        if n <= max_ticks:
            return np.arange(n)
        pos = np.linspace(0, n - 1, max_ticks)
        return np.unique(np.round(pos).astype(int))

    x_pos = _pick_positions(n_cols, x_tick_max)
    y_pos = _pick_positions(n_rows, y_tick_max)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bw_cmap = ListedColormap(["white", "black"])

    im = ax.imshow(
        mat.to_numpy(),
        cmap=bw_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",  # ← 正方形セル
    )

    # x-axis: labels shown
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(mat.columns[i]) for i in x_pos],
        rotation=90,
        ha="center",
        va="top",
    )

    # y-axis: tick marks only
    n_rows = mat.shape[0]
    step = 250

    # y 座標（下から）
    y_pos = np.arange(n_rows - 1, -1, -step)

    # 表示するラベル（下から何番目か）
    y_labels = [str(n_rows - y-1) for y in y_pos]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)

    # if title is None:
    #     title = "Adjacency matrix (1=black, 0=white)"
    # ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return AdjMatrixPlotResult(mat=mat, row_sums=row_sums, col_sums=col_sums)


# --- usage ---
result = plot_adj_matrix_corp_schmoch35_bw(
    (adj_df.merge(schmoch35_df, left_on='schmoch35', right_on='Field_number', how='left')\
          .drop(columns=['schmoch35', 'Field_number'])\
          .rename(columns={'Field_en':'schmoch35'})),
    corp_col='corporation',
    tech_col='schmoch35',
    val_col='mcp',
    figsize=(10, 11),
    dpi=150,
    x_tick_max=60,
    y_tick_max=12,
)

#%%
corp_df = c_df.query('period == "1981-2010"')\
            .filter(items=['corporation', 'diversity', 'eci', 'weight', 'ubiquity', 'mcp'])\
            .assign(
                patent_weight = lambda x: x.groupby('corporation')['weight'].transform('sum'),
                k_c1 = lambda x: x.query('mcp == 1').groupby('corporation')['ubiquity'].transform('sum') / x['diversity'],
            )\
            .drop(columns=['weight', 'ubiquity', 'mcp'])\
            .drop_duplicates().dropna()
corp_df
#%%
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.scatter(corp_df['diversity'], corp_df['eci'],
           s = 10,
           color='tab:blue')
ax.set_xlabel('Diversity')
ax.set_ylabel('KCI')
plt.show()
#%%
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.scatter(corp_df['diversity'], corp_df['k_c1'],
           s = 10,
           color='tab:blue')
ax.set_xlabel('Diversity')
ax.set_ylabel('$K_{C,1}$')
plt.show()
#%%
corp_df.corr()

#%%

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def bump_chart_from_eci(
    df: pd.DataFrame,
    *,
    corp_col: str = "corporation",
    period_col: str = "period",
    eci_col: str = "eci",
    times: Optional[Sequence] = None,
    top_k: int = 50,
    rank_method: str = "first",
    # plot style
    linewidth: float = 2.5,
    marker: Optional[str] = "o",
    label_fontsize: int = 10,
    min_label_gap: float = 0.15,
    ytick_multiple: int = 5,
    hide_spines: bool = True,
    show_ygrid: bool = True,
    left_labels: bool = True,
    right_labels: bool = True,
    label_rank_left: bool = False,
    label_rank_right: bool = False,
    # optional: 企業ごとの色指定
    ent_colors: Optional[Mapping[str, str]] = None,
) -> plt.Axes:
    """
    ['corporation','period','eci'] を持つDFから、各periodでECI降順の順位を作り、
    上位top_kのみのバンプチャートを描画する。
    """
    d = df[[corp_col, period_col, eci_col]].copy()
    d[eci_col] = pd.to_numeric(d[eci_col], errors="coerce")
    d = d.dropna(subset=[corp_col, period_col, eci_col])

    # period ごとに ECI 降順で rank（1が最上位）
    d["_rank"] = (
        d.groupby(period_col)[eci_col]
         .rank(method=rank_method, ascending=False)
         .astype(int)
    )

    # 上位 top_k のみ
    d = d[d["_rank"] <= int(top_k)].copy()

    # times 指定がなければ period の昇順（バンプは時系列順が基本）
    if times is None:
        times = sorted(d[period_col].unique())

    tidy = d.rename(
        columns={corp_col: "entity", period_col: "year", "_rank": "rank"}
    )[["entity", "year", "rank"]]

    ax = plot_bump_chart(
        tidy,
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
        left_labels=left_labels,
        right_labels=right_labels,
        label_rank_left=label_rank_left,
        label_rank_right=label_rank_right,
        ent_colors=ent_colors,
    )

    # 見た目として「1位が上」にしたいならこれをON（好み）
    ax.invert_yaxis()
    # ax.set_ylabel(f"Rank (ECI, top {top_k})")
    ax.set_xlabel(period_col)
    return ax
sep_year_corp_df = c_df.query('period != "1981-2010"')\
            .filter(items=['corporation', 'period', 'eci'])\
            .drop_duplicates().dropna()
sep_year_corp_df
#%%
plt.rcParams['font.family'] = 'Meiryo'
ax = bump_chart_from_eci(
    sep_year_corp_df,          # columns: corporation, period, eci
    top_k=50,
    ytick_multiple=1,
    right_labels=True,
    left_labels=True,
)
plt.tight_layout()
plt.show()



#%%
df_ranked = (
    df_income.assign(
        Ranking = df_income.groupby("period")["pci"].rank(ascending=False, method="dense")
    )
    .assign(Ranking=lambda d: d["Ranking"].astype(int))   # 見栄え用に整数化
)
# %%
# 1) 期ごとに tci の順位を作成（大きいほど1位）
df_ranked = (
    c_df.assign(
        Ranking = c_df.groupby("period")["pci"].rank(ascending=False, method="dense")
    )
    .assign(Ranking=lambda d: d["Ranking"].astype(int))   # 見栄え用に整数化
).query('period != "1981-2010"')

# 2) サンプルが期待する列名にあわせてリネーム
# df_income = (
#     df_ranked.rename(columns={
#         "period": "Year",
#         "ipc3": "District Name",
#         "pci": "Income"   # ← サンプルの hover が "Income" を参照するため合わせる
#     })
#     .sort_values(["Year", "Ranking"], ignore_index=True)\
#     .filter(items=['Year', 'District Name', 'Income', 'Ranking'])
# )
df_income = (
    df_ranked.filter(items=['period', 'ipc4', 'pci', 'Ranking'])\
             .drop_duplicates()\
             .sort_values(['period', 'Ranking'], ignore_index=True)\
             .assign(
                 ipc1 = lambda x: x['ipc4'].str[:1],
             )
)
#%%
display(df_income)

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


ax = bump_chart_from_pci(
    df_income,
    period_col="period",
    ipc_col="ipc4",
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

 #%%


# %%
custom_colors = get_custom_colors(background="light")
fig = go.Figure()

year_order = sorted(df_income["Year"].unique())
add_district_traces(fig, df_income.query('Ranking <= 30'), 
                    custom_colors, year_order)
add_ranking_annotations(fig, df_income, year_order)

add_subtitle(fig, "", subtitle_font_size=15, subtitle_color="grey", y_offset=1.05, x_offset=0.0)
add_footer(fig, "", footer_font_size=12, footer_color="grey", y_offset=-0.1, x_offset=0.35)

customize_layout(fig, year_order=year_order)
fig.show()

#%%
custom_colors = get_custom_colors(background="light")
fig = go.Figure()

year_order = sorted(df_income["Year"].unique())
add_district_traces(fig, df_income.query('31<= Ranking <= 60'), 
                    custom_colors, year_order)
add_ranking_annotations(fig, df_income, year_order)

add_subtitle(fig, "", subtitle_font_size=15, subtitle_color="grey", y_offset=1.05, x_offset=0.0)
add_footer(fig, "", footer_font_size=12, footer_color="grey", y_offset=-0.1, x_offset=0.35)

customize_layout(fig, year_order=year_order)
fig.show()
# %%
