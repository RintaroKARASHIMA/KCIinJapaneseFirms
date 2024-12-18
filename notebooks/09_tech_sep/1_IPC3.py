#! (root)/notebooks/00_template/1_sample.py python3
# -*- coding: utf-8 -*-

# %%
# Import Library
%run ../../src/load_libraries.py
%run 0_LoadLibraries.py

# Processing Data
from visualize import rank as vr

# Visualization

# Third Party
from ecomplexity import ecomplexity

# Set Visualization Parameters
pd.options.display.float_format = '{:.3f}'.format

# Import Original Modules
sys.path.append('../../src')
# from process import weight

# Import Initial Conditions
%run ../../src/initial_conditions.py

# Initialize Global Variables
global DATA_DIR, OUTPUT_DIR, EX_DIR
DATA_DIR = '../../data/processed/internal/filtered_after_agg/'
OUTPUT_DIR = '../../data/processed/internal/'
EX_DIR = '../../data/processed/external/schmoch/'

# Initialize Input and Output Conditions
input_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'
output_condition = f'{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}'

# Check the condition
print(input_condition)
print(output_condition)


# %%
def kh_ki(c_df, classification, n=19):
    kh1_ki1_df = pd.merge(
        c_df.copy(),
        c_df[c_df["mcp"] == 1]
        .groupby([region_corporation])[["ubiquity"]]
        .sum()
        .reset_index(drop=False)
        .copy()
        .rename(columns={"ubiquity": "kh_1"}),
        on=[region_corporation],
        how="left",
    )
    kh1_ki1_df = pd.merge(
        kh1_ki1_df.copy(),
        c_df[c_df["mcp"] == 1]
        .groupby([classification])[["diversity"]]
        .sum()
        .reset_index(drop=False)
        .copy()
        .rename(columns={"diversity": "ki_1"}),
        on=[classification],
        how="left",
    )
    kh1_ki1_df["kh_1"] = kh1_ki1_df["kh_1"] / kh1_ki1_df["diversity"]
    kh1_ki1_df["ki_1"] = kh1_ki1_df["ki_1"] / kh1_ki1_df["ubiquity"]
    kh_ki_df = kh1_ki1_df.copy()
    for i in range(n):
        kh_ki_df = pd.merge(
            kh_ki_df,
            kh_ki_df[kh_ki_df["mcp"] == 1]
            .groupby([region_corporation])[[f"ki_{i+1}"]]
            .sum()
            .reset_index(drop=False)
            .copy()
            .rename(columns={f"ki_{i+1}": f"kh_{i+2}"}),
            on=[region_corporation],
            how="left",
            copy=False,
        )
        kh_ki_df = pd.merge(
            kh_ki_df,
            kh_ki_df[kh_ki_df["mcp"] == 1]
            .groupby([classification])[[f"kh_{i+1}"]]
            .sum()
            .reset_index(drop=False)
            .copy()
            .rename(columns={f"kh_{i+1}": f"ki_{i+2}"}),
            on=[classification],
            how="left",
            copy=False,
        )
        kh_ki_df[f"kh_{i+2}"] = kh_ki_df[f"kh_{i+2}"] / kh_ki_df["diversity"]
        kh_ki_df[f"ki_{i+2}"] = kh_ki_df[f"ki_{i+2}"] / kh_ki_df["ubiquity"]
    return kh_ki_df


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

c_df = c_df[c_df["reg_num"] > 0].rename(
    columns=rename_col_dict)[col_order_list]
c_df = pd.concat(
    [
        kh_ki(c_df[c_df[f"{ar}_{year_style}_period"]
              == period], classification)
        for period in c_df[f"{ar}_{year_style}_period"].unique()
    ],
    axis="index",
    ignore_index=True,
)
# %%
# 各期間
classification_df = pd.merge(c_df.groupby([f'{ar}_{year_style}_period', classification])[['reg_num']].sum().reset_index(drop=False),
                             c_df.groupby([f'{ar}_{year_style}_period', classification])[
    [region_corporation]].nunique().reset_index(drop=False),
    on=[f'{ar}_{year_style}_period', classification],
    how='inner')
classification_df = pd.merge(classification_df,
                             c_df[[f'{ar}_{year_style}_period', classification, 'ubiquity', 'tci']
                                  + [f'ki_{i}' for i in range(1, 20+1)]]
                             .drop_duplicates(keep='first'),
                             on=[f'{ar}_{year_style}_period', classification],
                             how='inner')

fiveyears_df_dict = {
    f'{year}': classification_df[classification_df[f'{ar}_{year_style}_period'] == f'{year}'][[f'{ar}_{year_style}_period', classification, 'tci']].drop_duplicates(keep='first')
    for year in classification_df[f'{ar}_{year_style}_period'].unique() if year != f'{year_start}-{year_end}'
}

rank.rank_doubleaxis(fiveyears_df_dict,
                     rank_num=124,
                     member_col=classification,
                     value_col='tci',
                     prop_dict={
                         "figsize": (16, 16),
                         "xlabel": "Period",
                         "ylabel": "Technological Fields",
                         "title": "",
                         "fontsize": 15,
                         "year_range": 15,
                         "ascending": False,
                         "color": "default",
                     })

# %%
