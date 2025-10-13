#! (root)/notebooks/05_calculation/2_Complexity.py python3
# -*- coding: utf-8 -*-

#%%
## Load Global Settings
%run ../../src/initialize/load_libraries.py
%run ../../src/initialize/initial_conditions.py

## Load Local Settings
%run 0_LoadLibraries.py

# %%
# Import Original Modules
from calculation import method_of_reflections as mor
# reload(mor)

# %%
# Initialize Global Variables
in_dir = f'{IN_IN_DIR}filtered_after_agg/'
out_dir = f'{PRO_IN_DIR}'
ex_dir = f'{PRO_EX_DIR}'

## Check the condition
condition = 'app_nendo_1981_2010_5_all_p_3_right_person_name_fraction_schmoch35_fraction'
print(condition)


# %%
schmoch_df = pd.read_csv(
    f"{ex_dir}schmoch/35.csv",
    encoding="utf-8",
    sep=",",
    #  usecols=['Field_number', 'Field_en']
).drop_duplicates()

reg_num_top_df = pd.read_csv(
    f"{in_dir}{condition}.csv", 
    encoding="utf-8", 
    sep=","
)
print(reg_num_top_df[region_corporation].nunique())
display(reg_num_top_df.head())

#%%
plot_df = reg_num_top_df[reg_num_top_df[f'{ar}_{year_style}_period'] == f'{year_start}-{year_end}']\
                        [['right_person_name', 'reg_num']]\
                        .groupby('right_person_name', as_index=False)\
                        .sum()\
                        .assign(
                            reg_p = lambda x: x['reg_num']/x['reg_num'].sum()
                        )
plot_df['reg_p'].sum()


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

# %%
period_col = f"{ar}_{year_style}_period"
agg_dict = {
    "reg_num": ("reg_num", "sum"),
    "unique_classes": (classification, "nunique"),
    "diversity": ("diversity", "first"),
    "kci": ("kci", "first"),
    **{f"kh_{i}": (f"kh_{i}", "first") for i in range(1, 21)}
}
right_person_df = (
    c_df.groupby([period_col, region_corporation])
    .agg(**agg_dict)
    .reset_index()
)



right_person_df.to_csv(
    f"{out_dir}05_2_3_corporations/{condition}.csv",
    encoding="utf-8",
    sep=",",
    index=False,
)
# right_person_df.to_excel('../../output/tables/KCI.xlsx',
#                          index=False,
#                          sheet_name=output_condition)


# %%
# 各期間
period_col = f"{ar}_{year_style}_period"

agg_dict = {
    "reg_num": ("reg_num", "sum"),                      # reg_num の合計
    "unique_regions": (region_corporation, "nunique"),        # region_corporation のユニーク数
    "ubiquity": ("ubiquity", "first"),                        # グループごとの最初の値
    "tci": ("tci", "first"),                                  # 同上
    **{f"ki_{i}": (f"ki_{i}", "first") for i in range(1, 21)}  # ki_1–ki_20 を first
}

classification_df = (
    c_df.groupby([period_col, classification])
    .agg(**agg_dict)
    .reset_index()
)

classification_df
#%%
schmoch_df

#%%
classification_df = pd.merge(classification_df, 
                             schmoch_df.drop(columns=['IPC_code']).drop_duplicates().rename(columns={'Field_number':'schmoch35'}), 
                             on=classification, 
                             how='left')\
                        .drop(columns=[classification])\
                        .rename(columns={'Field_en':'schmoch35'})
classification_df
#%%
classification_df.to_csv(
    f"{out_dir}05_2_4_tech/{condition}.csv",
    encoding="utf-8",
    sep=",",
    index=False,
)
