import pandas as pd
import numpy as np
import sys
## Import Original Modules
sys.path.append("../../src")
from initialize import initial_conditions
from calculation import weight
from visualize import rank as vr

### Initialize Conditions
ar = initial_conditions.AR
year_style = initial_conditions.YEAR_STYLE

year_start = initial_conditions.YEAR_START
year_end = initial_conditions.YEAR_END
year_range = initial_conditions.YEAR_RANGE

extract_population = initial_conditions.EXTRACT_POPULATION
top_p_or_num = initial_conditions.TOP_P_OR_NUM
region_corporation = initial_conditions.REGION_CORPORATION
applicant_weight = initial_conditions.APPLICANT_WEIGHT

classification = initial_conditions.CLASSIFICATION
class_weight = initial_conditions.CLASS_WEIGHT

def kh_ki(c_df, region_corporation, classification, n=19):
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