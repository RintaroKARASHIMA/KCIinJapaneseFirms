#! (root)/notebooks/08_producer_sep/r1_Prefecture.py python3
# -*- coding: utf-8 -*-

#%%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

#%%
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
