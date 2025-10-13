#! (root)/notebooks/06_producer_long/r2_Coarse.py python3
# -*- coding: utf-8 -*-

#%%
# %load 0_LoadLibraries.py 
## Import Library
### Processing Data
import pandas as pd
import numpy as np
import sys

### Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import seaborn as sns
import cv2
from PIL import Image
import io

### Network Analysis
import networkx as nx
import networkx.algorithms.bipartite as bip

### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 20
pd.options.display.float_format = "{:.3f}".format

## Import Original Modules
sys.path.append("../../src")
import initial_condition
from process import weight
from visualize import rank as vr

# %%
## Set Global Variables
global DATA_DIR, EX_DIR, OUTPUT_DIR
DATA_DIR = "../../data/processed/internal/corporations/"
EX_DIR = "../../data/processed/external/schmoch/"
OUTPUT_DIR = "../../output/figures/"

## Initial Conditons
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE

extract_population = initial_condition.EXTRACT_POPULATION
top_p_or_num = ("p", 100)
region_corporation = "right_person_addr"
applicant_weight = initial_condition.APPLICANT_WEIGHT

classification = initial_condition.CLASSIFICATION
class_weight = initial_condition.CLASS_WEIGHT

color_list = initial_condition.COLOR_LIST

input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
fig_name_base = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}.png"

print(input_condition)

# %%
period_order_dict = {
    f"{period_start}-{period_start+year_range-1}": i
    for i, period_start in enumerate(range(year_start, year_end + 1, year_range))
}
period_order_dict[f"{year_start}-{year_end}"] = len(period_order_dict)
print(period_order_dict)

df = pd.read_csv(
    f"{DATA_DIR}{input_condition}.csv",
    encoding="utf-8",
    engine="python",
    sep=",",
    index_col=0,
)
display(df)

filtered_df = pd.read_csv(
    "../../data/interim/internal/filtered_before_agg/addedclassification.csv", sep=","
)
print(filtered_df)

# %%
##
df_dict = {}
combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    1: [
        "reg_num",
        "diversity",
        "特許数と法人次数（Diversity）の相関",
        "特許数（対数スケール）",
        "法人次数（Diversity）",
        "lower right",
    ],
    2: [
        "reg_num",
        "kci",
        "特許数とKCIの相関",
        "特許数（対数スケール）",
        "KCI",
        "lower right",
    ],
    3: [
        "diversity",
        "kci",
        "法人次数（Diversity）とKCIの相関",
        "法人次数（Diversity）",
        "KCI",
        "lower right",
    ],
    4: [
        "diversity",
        "kh_1",
        "法人次数（Diversity）と法人平均近傍次数（kh_1）の相関",
        "法人次数（Diversity）",
        "法人平均近傍次数（kh_1）",
        "lower right",
    ],
}

for i, combi in combi_dict.items():
    fig, ax = plt.subplots(figsize=(8, 8))
    period = f"{year_start}-{year_end}"
    corr_num = round(
        df[df[f"{ar}_{year_style}_period"] == period][combi[0]].corr(
            df[df[f"{ar}_{year_style}_period"] == period][combi[1]]
        ),
        3,
    )
    print(period, corr_num)
    ax.scatter(
        df[df[f"{ar}_{year_style}_period"] == period][combi[0]],
        df[df[f"{ar}_{year_style}_period"] == period][combi[1]],
        s=20,
        alpha=0.8,
        label=f"{period}年度（{corr_num}）",
        color="black",
    )
    if i == 4:
        ax.axvline(
            x=df[df[f"{ar}_{year_style}_period"] == period][combi[0]].mean(),
            color="red",
            linestyle="--",
        )
        ax.axhline(
            y=df[df[f"{ar}_{year_style}_period"] == period][combi[1]].mean(),
            color="red",
            linestyle="--",
        )
    ax.set_title(combi[2])
    if combi[0] in ["reg_num"]:
        ax.set_xscale("log")
    if combi[1] in ["reg_num"]:
        ax.set_yscale("log")
    x_min = df[(df[f"{ar}_{year_style}_period"] == period)][combi[0]].min()
    x_2smallest = (
        df[(df[f"{ar}_{year_style}_period"] == period)][combi[0]].nsmallest(2).iloc[1]
    )
    y_2smallest = (
        df[(df[f"{ar}_{year_style}_period"] == period)][combi[1]].nsmallest(2).iloc[1]
    )
    if i == 4:
        # ax.text(x_min-1,
        #         df[(df[f'{ar}_{year_style}_period']==period)&(df[combi[0]]==x_min)][combi[1]].values[0]-0.5,
        #         df[(df[f'{ar}_{year_style}_period']==period)&(df[combi[0]]==x_min)]['right_person_name'].values[0],
        #         fontsize=15, color='red')
        # ax.text(x_min-1,
        #         df[(df[f'{ar}_{year_style}_period']==period)&(df[combi[0]]==x_2smallest)][combi[1]].values[0]-0.5,
        #         df[(df[f'{ar}_{year_style}_period']==period)&(df[combi[0]]==x_2smallest)]['right_person_name'].values[0],
        #         fontsize=15, color='red')
        # ax.text(df[(df[f'{ar}_{year_style}_period']==period)&(df[combi[1]]==y_2smallest)][combi[0]].values[0]-1,
        #         y_2smallest-0.5,
        #         df[(df[f'{ar}_{year_style}_period']==period)&(df[combi[1]]==y_2smallest)]['right_person_name'].values[0],
        #         fontsize=15, color='red')
        for i, row in (
            df[(df[f"{ar}_{year_style}_period"] == period)]
            .sort_values("kci", ascending=False)
            .reset_index(drop=True)
            .iterrows()
        ):
            # 2つで68分かかる
            # if row['right_person_name'] in df[(df[f'{ar}_{year_style}_period']==period)].sort_values('kci', ascending=False).iloc[25:51,:]['right_person_name'].values:
            #     # ax.text(row[combi[0]], row[combi[1]]-0.5, i+1, fontsize=12, color='orange')
            #     if i+1!=51: ax.scatter(row[combi[0]], row[combi[1]], s=20, color='orange')
            # if row['right_person_name'] in df[(df[f'{ar}_{year_style}_period']==period)].sort_values('kci', ascending=False).iloc[51:76,:]['right_person_name'].values:
            #     # ax.text(row[combi[0]], row[combi[1]]-0.5, i+1, fontsize=12, color='green')
            #     ax.scatter(row[combi[0]], row[combi[1]], s=20, color='green')
            if (
                row["right_person_addr"]
                in df[(df[f"{ar}_{year_style}_period"] == period)]
                .sort_values("kci", ascending=False)
                .tail(25)["right_person_addr"]
                .values
            ):
                ax.text(
                    row[combi[0]], row[combi[1]] - 0.5, i + 1, fontsize=10, color="blue"
                )
                ax.scatter(row[combi[0]], row[combi[1]], s=20, color="blue")
            if (
                row["right_person_addr"]
                in df[(df[f"{ar}_{year_style}_period"] == period)]
                .sort_values("kci", ascending=False)
                .head(25)["right_person_addr"]
                .values
            ):
                ax.text(
                    row[combi[0]], row[combi[1]] - 0.5, i + 1, fontsize=12, color="red"
                )
                ax.scatter(row[combi[0]], row[combi[1]], s=20, color="red")

    ax.set_ylabel(combi[4])
    ax.set_xlabel(combi[3])
    ax.legend(loc=combi[5], fontsize=20)
    plt.show()

# %%
k_trans_df = (
    df[(df[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}")]
    .sort_values("kci", ascending=False)
    .copy()
)
k_trans_df_dict = {
    "0": k_trans_df.melt(
        id_vars=["right_person_name"], value_vars=[f"diversity"]
    ).sort_values("value", ascending=False)
}
fs = (32, 40)
for i in range(2, 20 + 1, 2):
    k_trans_df_dict[f"{i}"] = k_trans_df.melt(
        id_vars=["right_person_name"], value_vars=[f"kh_{i}"]
    ).sort_values("value", ascending=False)
    # if i >= 12: display(sample_df_dict[f'{i}'].head(10))
k_trans = vr.rank_doubleaxis(
    k_trans_df_dict,
    version="kci",
    rank_num=100,
    member_col="right_person_name",
    num_col="value",
    prop_dict={
        "figsize": fs,
        "xlabel": "N",
        "ylabel": "",
        "title": "",
        "fontsize": 24,
        "year_range": 2,
        "ascending": False,
        # 'color': color_dict
        "color": "default",
    },
)

plt.savefig(f"{OUTPUT_DIR}co_ranking/ktrans_{fig_name_base}", bbox_inches="tight")
plt.show()

# %%
df_dict = {}
combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    1: [
        "reg_num",
        "diversity",
        "特許数と法人次数（Diversity）の相関",
        "特許数（対数スケール）",
        "法人次数（Diversity）",
        "upper left",
    ],
    2: [
        "reg_num",
        "kci",
        "特許数とKCIの相関",
        "特許数（対数スケール）",
        "KCI",
        "lower left",
    ],
    3: [
        "diversity",
        "kci",
        "法人次数（Diversity）とKCIの相関",
        "法人次数（Diversity）",
        "KCI",
        "lower right",
    ],
    4: [
        "diversity",
        "kh_1",
        "法人次数（Diversity）と法人平均近傍次数（kh_1）の相関",
        "法人次数（Diversity）",
        "法人平均近傍次数（kh_1）",
        "lower right",
    ],
}

for i, combi in combi_dict.items():
    fig, ax = plt.subplots(figsize=(12, 12))
    color_count = 0
    for period in range(year_start, year_end + 1, year_range):
        period = f"{period}-{period+year_range-1}"
        corr_num = round(
            df[df[f"{ar}_{year_style}_period"] == period][combi[0]].corr(
                df[df[f"{ar}_{year_style}_period"] == period][combi[1]]
            ),
            3,
        )
        print(period, corr_num)
        ax.scatter(
            df[df[f"{ar}_{year_style}_period"] == period][combi[0]],
            df[df[f"{ar}_{year_style}_period"] == period][combi[1]],
            s=20,
            alpha=0.8,
            label=f"{period}年度（{corr_num}）",
            color=color_list[color_count],
        )
        if i == 4:
            ax.axvline(
                x=df[df[f"{ar}_{year_style}_period"] == period][combi[0]].mean(),
                color=color_list[color_count],
                linestyle="--",
            )
            ax.axhline(
                y=df[df[f"{ar}_{year_style}_period"] == period][combi[1]].mean(),
                color=color_list[color_count],
                linestyle="--",
            )
        ax.set_title(combi[2])
        if combi[0] in ["reg_num"]:
            ax.set_xscale("log")
        if combi[1] in ["reg_num"]:
            ax.set_yscale("log")
        ax.set_ylabel(combi[4])
        ax.set_xlabel(combi[3])
        ax.legend(loc=combi[5])
        color_count += 1

    plt.savefig(
        f"{OUTPUT_DIR}co_corr/{combi[0]}_{combi[1]}_{fig_name_base}",
        bbox_inches="tight",
    )
    plt.show()
# %%
df_dict = {}
combi_dict = {  # 1: ['right_person_name', 'diversity', f'特許権者次数(=diversity，k_h0)ランキング 値が小さいものTop15の推移（出願期間：{year_start}-{year_end}年度）', True],
    # 1: ['right_person_name', '', f'特許権者次数(=diversity，k_h0)ランキング 値が小さいものTop15の推移（出願期間：{year_start}-{year_end}年度）', True],
    2: [
        "right_person_name",
        "diversity",
        f"特許権者次数(=diversity，k_h0)ランキング 値が大きいものTop15の推移（出願期間：{year_start}-{year_end}年度）",
        False,
    ],
    #   3: ['right_person_name', 'eci', f'KCIランキング 値が小さいものTop15の推移（出願期間：{year_start}-{year_end}年度）', True],
    # 3: ['right_person_name', 'ipc_class_num', f'IPC数ランキング 値が大きいものTop15の推移（出願期間：{year_start}-{year_end}年度）', False],
    4: [
        "right_person_name",
        "kci",
        f"KCIランキング 値が大きいものTop15の推移（出願期間：{year_start}-{year_end}年度）",
        False,
    ],
}

for i, combi in combi_dict.items():
    div_df = (
        df[[f"{ar}_{year_style}_period"] + combi[: 1 + 1]]
        .sort_values(
            by=[f"{ar}_{year_style}_period", combi[1]], ascending=[True, False]
        )
        .copy()
    )
    div_df = div_df.drop_duplicates(keep="first")
    div_df_dict = {}
    for year in range(year_start, year_end + 1, year_range):
        period = f"{year}-{year+year_range-1}"
        div_df_dict[period] = div_df[
            div_df[f"{ar}_{year_style}_period"] == period
        ].copy()
        # display(div_df_dict[f'{year}-{year+year_range-1}'].head(15))
    if i == 4:
        df["kci_rank"] = (
            df.groupby(f"{ar}_{year_style}_period")[["kci"]]
            .rank(ascending=False, method="first")
            .reset_index(drop=False)["kci"]
        )
        # display(c_df[c_df['right_person_name'].str.contains('三菱重工業')])
    if i > 4:
        fs = (12, 15)
    else:
        fs = (10, 12)
    # display(div_df)
    sample = vr.rank_doubleaxis(
        div_df_dict,
        version="kci",
        rank_num=15,
        member_col=combi[0],
        num_col=combi[1],
        prop_dict={
            "figsize": fs,
            "xlabel": "期間",
            "ylabel": "",
            "title": combi[2],
            "fontsize": 24,
            "year_range": year_range,
            "ascending": combi[3],
            # 'color': color_dict
            "color": "default",
        },
    )
    plt.savefig(
        f"{OUTPUT_DIR}co_ranking/{combi[1]}_{fig_name_base}", bbox_inches="tight"
    )
    plt.show()

# %%
color_list = ["red"] + [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "grey",
    "violet",
    "indigo",
    "turquoise",
    "gold",
    "lime",
    "coral",
    "navy",
    "skyblue",
    "tomato",
    "olive",
    "cyan",
    "darkred",
    "darkgreen",
    "darkblue",
    "darkorange",
    "darkviolet",
    "deeppink",
    "firebrick",
    "darkcyan",
    "darkturquoise",
    "darkslategray",
    "darkgoldenrod",
    "mediumblue",
    "mediumseagreen",
    "mediumpurple",
    "mediumvioletred",
    "midnightblue",
    "saddlebrown",
    "seagreen",
    "sienna",
    "steelblue",
][10:]
color_count = 0
fig, ax = plt.subplots(figsize=(10, 10))
for s in list(right_person_df["segment"].unique())[0:]:

    x = (
        right_person_df[right_person_df["segment"] == s][["reg_num"]]
        .rank(ascending=False, method="first")
        .sort_values("reg_num", ascending=True)["reg_num"]
    )
    # y = 1 - np.cumsum(right_person_df[right_person_df['segment']==s][['reg_num']].sort_values('reg_num',ascending=False)['reg_num'] / right_person_df[right_person_df['segment']==s]['reg_num'].sum())
    y = np.cumsum(
        right_person_df[right_person_df["segment"] == s][["reg_num"]].sort_values(
            "reg_num", ascending=False
        )["reg_num"]
        / right_person_df[right_person_df["segment"] == s]["reg_num"].sum()
    )
    # y = [1] + list(y)[:-1]
    y = list(y)[:-1] + [1]
    # ccdf_array = ccdf()
    # ax.plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=8,
    #                 color=color_list[color_count], label=s+'年度', alpha=0.6)
    ax.plot(
        x,
        y,
        "o",
        markersize=8,
        color=color_list[color_count],
        label=s + "年度",
        alpha=0.6,
    )
    ax.axvline(len(x) * 3 // 100, color=color_list[color_count], linestyle="--")
    color_count += 1
# ax.legend(loc='lower left', fontsize=18)
ax.legend(loc="upper left", fontsize=18)

ax.set_title(
    "各期間における特許権者の累積特許数分布（両対数スケール）" + "\n", fontsize=20
)
ax.set_xlabel("特許数", fontsize=18)
ax.set_ylabel("ccdf", fontsize=18)

ax.set_xscale("log")
# ax.set_yscale('log')

ax.tick_params(labelsize=18)
# ax.set_xlim(0.8, 300)

# x軸の指数表記を普通に戻す魔法
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# ax.set_xlim(prop_dict['xlim'])
# ax.set_ylim(prop_dict['ylim'])

ax.grid(
    axis="both", which="major", alpha=1, linestyle="--", linewidth=0.6, color="gray"
)

plt.show()
