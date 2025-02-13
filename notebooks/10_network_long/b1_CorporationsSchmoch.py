#! (root)/notebooks/08_producer_sep/r1_Prefecture.py python3
# -*- coding: utf-8 -*-

#%%
%run ../../src/initialize/load_libraries.py
%run ./0_LoadLibraries.py

#%%
data_dir = "../../data/processed/internal"

node_df = pd.read_csv(
                    f'{data_dir}/05_2_2_bipartite/{input_condition}_node.csv',
                    encoding='utf-8',
                    sep=',',
                    )#.query(f'{ar}_{year_style}_period == "{year_start}-{year_end}"')
edge_df = pd.read_csv(
                    f'{data_dir}/05_2_2_bipartite/{input_condition}_edge.csv',
                    encoding='utf-8',
                    sep=',',
                    )#.query(f'{ar}_{year_style}_period == "{year_start}-{year_end}"')

#%%
corporations_n = node_df.query('projected == "0"')['node_id'].nunique()
tech_n = node_df.query('projected == "1"')['node_id'].nunique()
adj_mat = pd.pivot_table(
                        edge_df,
                        index='Source',
                        columns='Target',
                        values='Weight',
                        aggfunc='sum',
                        fill_value=0,
                    ).values

#%%
# --- 企業(=projected=="0") と 技術(=projected=="1") のリスト抽出 ---
corp_nodes = node_df.query('projected == 0')['node_id'].unique()
tech_nodes = node_df.query('projected == 1')['node_id'].unique()

# --- まずは通常のピボットで全体の隣接行列を作る ---
adj_mat_df = pd.pivot_table(
    edge_df,
    index='Source',
    columns='Target',
    values='Weight',
    aggfunc='sum',
    fill_value=0,
)

# ----------------------------------------------------------------------
# 1) 企業ノード（行）の並べ替え
#    （例：企業ごとの総エッジ重みを合計し，小さい順にソート）
# ----------------------------------------------------------------------
corp_order = (
    edge_df.groupby('Source')['Weight']
    .sum()
    .sort_values(ascending=False)
    .index
)
# corp_order には企業のみ含めたいので、もし Source に技術が混在する可能性があればフィルタする
corp_order = [n for n in corp_order if n in corp_nodes]

# ----------------------------------------------------------------------
# 2) 技術ノード（列）の並べ替え
#    （例：技術ごとの総エッジ重みを合計し，小さい順にソート）
# ----------------------------------------------------------------------
tech_order = (
    edge_df.groupby('Target')['Weight']
    .sum()
    .sort_values(ascending=False)
    .index
)
# tech_order には技術のみ含めたいので、もし Target に企業が混在する可能性があればフィルタする
tech_order = [n for n in tech_order if n in tech_nodes]

# ----------------------------------------------------------------------
# 3) reindex で行列を再構築（ソート順に並べ替え）
# ----------------------------------------------------------------------
adj_mat_sorted_df = adj_mat_df.reindex(index=corp_order, columns=tech_order, fill_value=0)

# NumPy 配列に変換
adj_mat_sorted = adj_mat_sorted_df.values

# --- 0 以外のセルだけを可視化するためのマスクを作る ---
adj_mat_masked = np.ma.masked_where(adj_mat_sorted == 0, adj_mat_sorted)

# カラーマップ用に最小値・最大値を適当に指定（例）
w_min, w_max = 0.0, 2.0

# --- 可視化 ---
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    constrained_layout=True,
    figsize=(20, 10),
    dpi=100,
    facecolor="white",
)

# pcolorで行列を描画
c = ax.pcolor(adj_mat_masked.T, cmap="gray", vmin=w_min, vmax=w_max)

# 軸ラベルを設定（並べ替え後の順番に合わせたい場合は xticks, yticks を設定）
ax.set_xlabel(f"Corporations (N={len(corp_order)})")
ax.set_ylabel(f"Technological Fields (N={len(tech_order)})")

# y軸は行列の上から下へ描画されるため、企業を上から並べたいなら反転する
ax.invert_yaxis()

# 必要に応じて目盛りラベルを設定する例（企業名や技術名が多い場合は間引きや回転が必須）
# ax.set_xticks(np.arange(len(tech_order))+0.5)
# ax.set_xticklabels(tech_order, rotation=90, fontsize=8)
# ax.set_yticks(np.arange(len(corp_order))+0.5)
# ax.set_yticklabels(corp_order, fontsize=8)

# fig.colorbar(c, ax=ax)
ax.grid(False)  # pcolor 上に不要な格子線が重なる場合は消す

plt.show()

#%%
# 隣接区域以外を非表示化
adj_mat_masked = np.ma.masked_where(adj_mat == 0, adj_mat)

# 色の調整用
w_min, w_max = 0.0, 2.0

# 空間隣接行列を作図
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    constrained_layout=True,
    figsize=(10, 10),
    dpi=100,
    facecolor="white",
)
ax.pcolor(adj_mat_masked, cmap="grey", vmin=w_min, vmax=w_max)  # 行列
# ax.set_xticks(ticks=np.arange(N)+0.5)
# ax.set_xticklabels(labels=gdf_target.city2, size=9, rotation=90)
# ax.set_yticks(ticks=np.arange(N)+0.5)
# ax.set_yticklabels(labels=gdf_target.city2, size=9)
# ax.invert_yaxis() # 軸の反転
ax.set_ylabel("Corporations ($N=1938$)")
ax.set_xlabel("Schmoch Technological Fields ($N=35$)")
# ax.set_title(f'Osaka: $N = {len(gdf_target)}$', loc='left')
# fig.suptitle('spatial adjacency matrix', fontsize=20)
ax.grid()
# ax.set_aspect('equal', adjustable='box')
# fig.savefig(
#     f"{output_dir}adjacency_matrix_{output_condition}.png",
#     bbox_inches="tight",
#     pad_inches=0.1,
# )
plt.show()
