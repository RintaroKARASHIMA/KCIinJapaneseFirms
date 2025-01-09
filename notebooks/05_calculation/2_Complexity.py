#! (root)/notebooks/05_calculation/2_Complexity.py python3
# -*- coding: utf-8 -*-

# %%
# %run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py
# Import Libraries
from ecomplexity import ecomplexity
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
classification = 'ipc3'
output_dir = '../../data/processed/internal/'

## Check the condition
print(input_condition)
print(output_condition)


# %%
schmoch_df = pd.read_csv(
    f"{EX_DIR}schmoch/35.csv",
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
#%%
reg_num_top_df

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
    f"{output_dir}05_2_3_corporations/{output_condition}.csv",
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

schmoch_df['ipc3'] = schmoch_df['IPC_code'].str[:3]
classification_df = pd.merge(classification_df, schmoch_df.drop(columns=['IPC_code', 'Field_number']), 
                             on='ipc3', 
                             how='left')
classification_df.to_csv(
    f"{output_dir}05_2_4_tech/{output_condition}.csv",
    encoding="utf-8",
    sep=",",
    index=False,
)

#%%

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

# %%
