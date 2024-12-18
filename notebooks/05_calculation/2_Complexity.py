# Calculate Complexity

# %%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

# Import Libraries
import openpyxl

# %%
# Import Original Modules
from calculation import method_of_reflections as mor

# %%
# Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = "../../data/processed/internal/filtered_after_agg/"
OUTPUT_DIR = "../../data/processed/internal/"
EX_DIR = "../../data/processed/external/schmoch/"

## Name of Input and Output Files
input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
output_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"

## Check the condition
print(input_condition)
print(output_condition)


# %%


# %%
schmoch_df = pd.read_csv(
    f"{EX_DIR}35.csv",
    encoding="utf-8",
    sep=",",
    #  usecols=['Field_number', 'Field_en']
).drop_duplicates()

reg_num_top_df = pd.read_csv(
    f"{DATA_DIR}{input_condition}.csv", encoding="utf-8", sep=","
)
print(reg_num_top_df[region_corporation].nunique())

# %%
# pd.read_csv("../../data/interim/internal/filtered_before_agg/japan.csv").query(
#     "1981 <= app_nendo <= 2010"
# ).query(f"right_person_addr in {list(reg_num_top_df[region_corporation].unique())}")[
#     "reg_num"
# ].nunique()

# %%
trade_cols = {
    "time": f"{ar}_{year_style}_period",
    "loc": region_corporation,
    "prod": classification,
    "val": "reg_num",
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

c_df = ecomplexity(reg_num_top_df, cols_input=trade_cols, rca_mcp_threshold=1)
# prox_df = proximity(c_df, trade_cols)
# c_out_df = c_df.copy()
print(c_df.columns)

c_df = c_df[c_df["reg_num"] > 0].rename(columns=rename_col_dict)[col_order_list]
c_df = pd.concat(
    [
        mor.kh_ki(c_df[c_df[f"{ar}_{year_style}_period"] == period], classification)
        for period in c_df[f"{ar}_{year_style}_period"].unique()
    ],
    axis="index",
    ignore_index=True,
)

# for segment in c_df[f'{ar}_{year_style}_period'].unique():
#     display(c_df[c_df[f'{ar}_{year_style}_period'] == segment].head())
#     display(c_df[c_df[f'{ar}_{year_style}_period'] == segment].describe())
#     print(c_df[c_df[f'{ar}_{year_style}_period'] == segment].info())
#     print('\n')

# %%
# schmoch_df['ipc3'] = schmoch_df['IPC_code'].str[:3]
# schmoch_df = schmoch_df.drop_duplicates()
# schmoch_df
# c_df2 = pd.merge(c_df,
#                 schmoch_df,
#                 on=['ipc3'],
#                 how='left')\
#             .rename(columns={'Field_en':'schmoch35'})\
#             .drop(columns=['IPC_code', 'Field_number'])\
#             .drop_duplicates()
c_df2 = c_df.copy()
display(c_df2.head())

# %%
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 35
c_df

# 地区数を取得
N = c_df[region_corporation].nunique()

adj_mat = (
    pd.pivot_table(
        c_df2[
            c_df2[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}"
        ].sort_values(by=["ubiquity"], ascending=[False], ignore_index=True),
        index=region_corporation,
        columns=classification,
        values="mcp",
    )
    .fillna(0)
    .values
)
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
from matplotlib import colormaps

list(colormaps)

# %%

right_person_df = pd.merge(
    c_df.groupby([f"{ar}_{year_style}_period", region_corporation])[["reg_num"]]
    .sum()
    .reset_index(drop=False),
    c_df.groupby([f"{ar}_{year_style}_period", region_corporation])[[classification]]
    .nunique()
    .reset_index(drop=False),
    on=[f"{ar}_{year_style}_period", region_corporation],
    how="inner",
)
right_person_df = pd.merge(
    right_person_df,
    c_df[
        [f"{ar}_{year_style}_period", region_corporation, "diversity", "kci"]
        + [f"kh_{i}" for i in range(1, 20 + 1)]
    ].drop_duplicates(keep="first"),
    on=[f"{ar}_{year_style}_period", region_corporation],
    how="inner",
)
# for period in right_person_df[f'{ar}_{year_style}_period'].unique():
#     right_person_df

# for period in right_person_df[f'{ar}_{year_style}_period'].unique():
#     for i in range(1, 20+1):
#         value = right_person_df[right_person_df[f'{ar}_{year_style}_period']==period]
#         right_person_df[right_person_df[f'{ar}_{year_style}_period']==period][f'kh_{i}'] = (value[f'kh_{i}'] - value[f'kh_{i}'].mean()) / value[f'kh_{i}'].std()
#     display(right_person_df[right_person_df[f'{ar}_{year_style}_period'] == period].head())
#     display(right_person_df[right_person_df[f'{ar}_{year_style}_period'] == period].describe())
#     print(right_person_df[right_person_df[f'{ar}_{year_style}_period'] == period].info())
#     print('\n')
# right_person_df['reg_num'] = right_person_df['reg_num'].astype(np.int64)

right_person_df.to_csv(
    f"{output_dir}corporations/{output_condition}.csv",
    encoding="utf-8",
    sep=",",
    index=False,
)
# right_person_df.to_excel('../../output/tables/KCI.xlsx',
#                          index=False,
#                          sheet_name=output_condition)


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
display(classification_df)

#%%

# classification_df['reg_num'] = classification_df['reg_num'].astype(np.int64)
# classification_df = pd.merge(classification_df,
#                             schmoch_df.rename(columns={'Field_number':classification}),
#                             on=[classification],
#                             how='inner')\
#                             .drop(columns=[classification])\
#                             .rename(columns={'Field_en':classification})
# display(classification_df)
# schmoch_df['ipc3'] = schmoch_df['IPC_code'].str[:3]
# schmoch_df = schmoch_df.drop_duplicates()
# schmoch_df
# classification_df = pd.merge(classification_df,
#                                 schmoch_df,
#                                 # on=['ipc3'],
#                                 on=['schmoch35'],
#                                 how='left')\
#                             .rename(columns={'Field_en':'schmoch35'})\
#                             .drop(columns=['IPC_code', 'Field_number'])\
#                             .drop_duplicates()
# for period in classification_df[f'{ar}_{year_style}_period'].unique():
#     classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'] = (classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'] - classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'].min()) / (classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'].max() - classification_df[classification_df[f'{ar}_{year_style}_period']==period]['tci'].min())


fiveyears_df_dict = {
    f"{year}": classification_df[
        classification_df[f"{ar}_{year_style}_period"] == f"{year}"
    ][[f"{ar}_{year_style}_period", classification, "tci"]].drop_duplicates(
        keep="first"
    )
    for year in classification_df[f"{ar}_{year_style}_period"].unique()
    if year != f"{year_start}-{year_end}"
}

rank.rank_doubleaxis(
    fiveyears_df_dict,
    rank_num=124,
    member_col=classification,
    value_col="tci",
    prop_dict={
        "figsize": (16, 10),
        "xlabel": "Period",
        "ylabel": "Technological Fields",
        "title": "",
        "fontsize": 15,
        "year_range": 15,
        "ascending": False,
        "color": "default",
    },
)

classification_df.to_csv(
    f"{output_dir}tech/{output_condition}.csv", encoding="utf-8", sep=",", index=False
)
# classification_df[classification_df[f'{ar}_{year_style}_period']==f'{year_start}-{year_end}']\
#     [['schmoch35', 'reg_num', 'ubiquity', 'tci']]\
#     .rename(columns={'reg_num':'patent count', 'ubiquity':'degree centrality', 'tci':'TCI'})\
#     .to_excel('../../output/tables/TCI.xlsx',
#                          index=False,
#                          sheet_name=output_condition)

# %%
# openpyxl.Workbook()
# import openpyxl as xl
# from openpyxl.styles.borders import Border, Side

# wb1 = xl.load_workbook(filename='../../output/tables/TCI.xlsx')
# ws1 = wb1[output_condition]
# side = Side(style='thick', color='000000')

# border = Border(top=side, bottom=side, left=side, right=side)

# for row in ws1:
#     for cell in row:
#         ws1[cell.coordinate].border = border
# wb1.save('../../output/tables/TCI.xlsx')

# eneos_df = c_df[(c_df[f'{ar}_{year_style}_period']==f'{year_start}-{year_end}')&(c_df[region_corporation].str.contains('ＥＮＥＯＳ'))\
#                 &(c_df['mcp']==1)].copy()#[[region_corporation, 'reg_num', 'schmoch35']].copy()
# eneos_df = pd.merge(eneos_df,
#                     schmoch_df.rename(columns={'Field_number':'schmoch35'})\
#                               .drop_duplicates(keep='first'),
#                     on=['schmoch35'],
#                     how='inner')
# eneos_df[['ubiquity', 'Field_en', 'ki_1']]# eneos_df = c_df[(c_df[f'{ar}_{year_style}_period']==f'{year_start}-{year_end}')&(c_df[region_corporation].str.contains('ＥＮＥＯＳ'))\
#                 &(c_df['mcp']==1)].copy()#[[region_corporation, 'reg_num', 'schmoch35']].copy()
# eneos_df = pd.merge(eneos_df,
#                     schmoch_df.rename(columns={'Field_number':'schmoch35'})\
#                               .drop_duplicates(keep='first'),
#                     on=['schmoch35'],
#                     how='inner')
# eneos_df[['ubiquity', 'Field_en', 'ki_1']]
classification_df[
    classification_df[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}"
][["schmoch35", "reg_num", "ubiquity", "tci"]].drop_duplicates(
    keep="first"
).sort_values(
    by=["ubiquity"], ascending=[False], ignore_index=True
)

# %%
# Bipartite Graph
ps_df = classification_df[
    classification_df[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}"
][["schmoch35", "reg_num", "ubiquity", "tci", "ki_1"]].copy()
schmoch5_df = pd.read_csv(
    f"{ex_dir}35.csv",
    encoding="utf-8",
    sep=",",
    usecols=["Field_number", "Field_en", "schmoch5"],
).drop_duplicates(ignore_index=True)
ps_df = (
    pd.merge(ps_df, schmoch5_df, left_on="schmoch35", right_on="Field_en", how="inner")
    .drop(columns=["schmoch35"])
    .rename(columns={"Field_en": "label", "Field_number": "node_id"})
)
ps_df["tci"] = (
    (ps_df["tci"] - ps_df["tci"].min())
    / (ps_df["tci"].max() - ps_df["tci"].min())
    * 100
)
ps_df["node_id"] -= 1
ps_df[["node_id", "label", "reg_num", "ubiquity", "tci", "schmoch5"]].to_csv(
    f"{output_dir}product_space/{output_condition}_node.tsv",
    sep="\t",
    encoding="utf-8",
    index=False,
)
ps_df.drop_duplicates(keep="first", ignore_index=True).sort_values(
    by=["ki_1"], ascending=[False], ignore_index=True
)
ps_edge_df = (
    prox_df[prox_df[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}"]
    .rename(
        columns={
            "schmoch35_1": "source",
            "schmoch35_2": "target",
            "proximity": "weight",
        }
    )
    .drop(columns=[f"{ar}_{year_style}_period"])
    .drop_duplicates(subset=["source", "target"], ignore_index=True)
)
ps_edge_df["source"] -= 1
ps_edge_df["target"] -= 1
ps_edge_df.to_csv(
    f"{output_dir}product_space/{output_condition}_edge.tsv",
    sep="\t",
    encoding="utf-8",
    index=False,
)
ps_edge_df
graph_df = (
    pd.merge(
        c_df, schmoch_df, left_on=classification, right_on="Field_number", how="left"
    )
    .drop(columns=["Field_number", classification])
    .rename(columns={"Field_en": classification})
)
# graph_df = graph_df[graph_df['mcp']==1][[f'{ar}_{year_style}', region_corporation, 'ipc_class', 'mcp']]
graph_df
all_edge_df = (
    graph_df[
        (graph_df[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}")
        & (graph_df["mcp"] == 1)
    ]
    .copy()[[region_corporation, classification, "mcp"]]
    .rename(columns={"mcp": "Weight"})
)
all_edge_df["Type"] = "Undirected"
all_edge_df

all_node_list = list(all_edge_df[region_corporation].unique()) + list(
    all_edge_df[classification].unique()
)
all_flag_list = [0] * len(all_edge_df[region_corporation].unique()) + [1] * len(
    all_edge_df[classification].unique()
)
all_node_df = (
    pd.DataFrame(all_node_list, columns=["label"])
    .reset_index(drop=False)
    .rename(columns={"index": "node_id"})
)
all_node_df["projected"] = all_flag_list
all_node_df["node_id"] += 1

all_edge_df = pd.merge(
    all_edge_df, all_node_df, left_on=region_corporation, right_on="label", how="left"
).rename(columns={"node_id": "Source"})
all_edge_df = pd.merge(
    all_edge_df, all_node_df, left_on=classification, right_on="label", how="left"
).rename(columns={"node_id": "Target"})

all_edge_df = all_edge_df[["Source", "Target", "Type", "Weight"]]

all_node_df.to_csv(
    f"{output_dir}graph/{output_condition}_node.csv",
    encoding="utf-8",
    sep=",",
    index=False,
)
all_edge_df.to_csv(
    f"{output_dir}graph/{output_condition}_edge.csv",
    encoding="utf-8",
    sep=",",
    index=False,
)
# graph_df.to_csv(f'{output_dir}graph/{output_condition}.csv',
#                 encoding='utf-8',
#                 sep=',',
#                 index=False)
# graph_df
