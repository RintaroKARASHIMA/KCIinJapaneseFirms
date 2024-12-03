#! (root)/notebooks/06_producer_long/c1_Statics.py python3
# -*- coding: utf-8 -*-

# %%
%run 0_LoadLibraries.py
%run ../../src/initial_condition.py


# %%
## Set Global Variables
global DATA_DIR, EX_DIR, OUTPUT_DIR
DATA_DIR = "../../data/processed/internal/corporations/"
EX_DIR = "../../data/processed/external/schmoch/"
OUTPUT_DIR = "../../output/figures/"

## Arrange Conditons
top_p_or_num = ("p", 100)
region_corporation = "right_person_addr"

input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
fig_name_base = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}.png"

print(input_condition)
print(fig_name_base)

# %%
period_order_dict = {
    f"{period_start}-{period_start+year_range-1}": i
    for i, period_start in enumerate(range(year_start, year_end + 1, year_range))
}
period_order_dict[f"{year_start}-{year_end}"] = len(period_order_dict)
period_order_dict

# %%
df = pd.read_csv(f"{DATA_DIR}{ar}_{year_style}_{top_p_or_num[0]}_{top_p_or_num[1]}.csv")
df

# %%
filtered_df = pd.read_csv(
    "../../data/interim/internal/filtered_before_agg/addedclassification.csv", sep=","
)

# %%
filtered_df  # filtered_df['right_person_name'].nunique()
filtered_df[filtered_df[f"{ar}_{year_style}"].isin(range(year_start, year_end + 1))][
    "right_person_name"
].nunique()

filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))&(filtered_df['right_person_name'].isin(df['right_person_name']))]['right_person_name'].nunique()

filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))&(filtered_df['right_person_name'].isin(df['right_person_name']))]['right_person_name'].nunique()*100 / filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))]['right_person_name'].nunique()

filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))&(filtered_df['right_person_name'].isin(df['right_person_name']))]['reg_num'].nunique()


filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))]['reg_num'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))]['reg_num'].nunique()