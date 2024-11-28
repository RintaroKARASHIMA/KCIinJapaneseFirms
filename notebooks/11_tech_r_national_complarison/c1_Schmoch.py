#! (root)/notebooks/00_template/1_sample.py python3
# -*- coding: utf-8 -*-

# %%
# %load 0_LoadLibraries.py
## Import Library
### Processing Data
import sys
import pandas as pd
import numpy as np

### Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import seaborn as sns
import cv2
from PIL import Image
import io
import networkx as nx
import networkx.algorithms.bipartite as bip


### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format
plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 20

## Import Original Modules
sys.path.append("../../src")
import initial_condition
from process import weight
from visualize import rank as vr

## Initialize Global Variables
global DATA_DIR, EX_DIR, OUTPUT_DIR
DATA_DIR = "../../data/processed/internal/tech/"
EX_DIR = "../../data/processed/external/"
OUTPUT_DIR = "../../output/figures/"

## Initialize Input and Output Conditions
### Import Initial Conditions
ar = initial_condition.AR
year_style = initial_condition.YEAR_STYLE

year_start = initial_condition.YEAR_START
year_end = initial_condition.YEAR_END
year_range = initial_condition.YEAR_RANGE

extract_population = initial_condition.EXTRACT_POPULATION
top_p_or_num = initial_condition.TOP_P_OR_NUM
region_corporation = initial_condition.REGION_CORPORATION
applicant_weight = initial_condition.APPLICANT_WEIGHT

classification = initial_condition.CLASSIFICATION
class_weight = initial_condition.CLASS_WEIGHT

input_condition = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}"
fig_name_base = f"{ar}_{year_style}_{extract_population}_{top_p_or_num[0]}_{top_p_or_num[1]}_{region_corporation}_{applicant_weight}_{classification}_{class_weight}.png"

### Check the condition
print(input_condition)
print(fig_name_base)


# %%
period_order_dict = {
    f"{period_start}-{period_start+year_range-1}": i
    for i, period_start in enumerate(range(year_start, year_end + 1, year_range))
}

period_order_dict[f"{year_start}-{year_end}"] = len(period_order_dict)
period_order_dict
jp_df = pd.read_csv(f"{data_dir}{input_condition}.csv", encoding="utf-8", sep=",")

# schmoch_df = pd.read_csv(f'{ex_dir}schmoch/35.csv',
#                          encoding='utf-8',
#                          sep=',',
#                          usecols=['Field_number', 'Field_en']).drop_duplicates()

# jp_df = pd.merge(jp_df, schmoch_df, left_on=classification, right_on='Field_number', how='left')\
#         .drop(columns=['Field_number', classification])\
#         .rename(columns={'Field_en': classification})\
#         .sort_values(f'{ar}_{year_style}_period', key=lambda col: col.map(period_order_dict))
jp_df = jp_df.sort_values(
    f"{ar}_{year_style}_period", key=lambda col: col.map(period_order_dict)
)

eu_df = pd.read_csv(f"{ex_dir}abroad/eu.csv", encoding="utf-8", sep=",")
eu_df

# %%
eu_jp_df = pd.merge(
    jp_df[jp_df[f"{ar}_{year_style}_period"] == f"{year_start}-{year_end}"][
        [classification, "tci", "ubiquity", "ki_1", "reg_num"]
    ],
    eu_df[[classification, "schmoch5", "TCI_eu", "reg_num_eu"]],
    on=classification,
    how="outer",
).rename(columns={"tci": "TCI_jp", "reg_num": "reg_num_jp"})
eu_jp_df["TCI_jp"] = (
    (eu_jp_df["TCI_jp"] - eu_jp_df["TCI_jp"].min())
    / (eu_jp_df["TCI_jp"].max() - eu_jp_df["TCI_jp"].min())
    * 100
)
eu_jp_df = eu_jp_df.sort_values("TCI_jp", ascending=False).reset_index(drop=True)
eu_jp_df["schmoch5"] = eu_jp_df["schmoch5"].replace(
    "Mechanical engineering", "Mechanical engineering, machinery"
)
eu_jp_df["schmoch5"] = eu_jp_df["schmoch5"].replace(
    "Chemistry", "Chemistry, pharmaceuticals"
)
display(jp_df[~jp_df["schmoch35"].isin(eu_df["schmoch35"])]["schmoch35"].unique())

eu_jp_df[['TCI_rank_jp', 'TCI_rank_eu']] = eu_jp_df[['TCI_jp', 'TCI_eu']].rank(ascending=False)
display(eu_jp_df)

#%%
df_dict = {}
tech_color = {
        'Chemistry, pharmaceuticals': 'red',
        'Electrical engineering': 'blue',
        'Instruments': 'green', 
        'Mechanical engineering, machinery': 'orange',
        'Other fields': 'gray'
    }
combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    # 1: ["TCI_jp", "TCI_eu", "relation between the TCIs in Japanese corporation and EU regions", "Japanese Corporations（period：1981-2010 fiscal year）", "EU Regions（period：1985-2009 year）", "center", ],
    # 2: ["TCI_rank_jp", "TCI_rank_eu", "relation between the TCIs in Japanese corporation and EU regions", "Japanese Corporations ranking（period：1981-2010 fiscal year）", "EU Regions ranking（period：1985-2009 year）", "center", ],
    # 2: ["reg_num_jp", "reg_num_eu", "corr between the patent amounts in Japan and EU", "Japan（period：1981-2010 fiscal year）", "EU（period：1985-2009 year）", "center", ],
    # 3: ["reg_num_jp", "TCI_jp", "relation between the patent counts and the TCIs in Japan", "Patent Counts", "TCIs", "center left", ],
    # 4: ["TCI_jp", "reg_num_jp", "relation between the patent counts and the TCIs in Japan", "TCIs", "Patent Counts", "center left", ],
    # 6: ["TCI_jp", "ubiquity", "relation between the ubiquity and the TCIs in Japan", "TCIs", "Ubiquity", "center left", ],
    7: ["ubiquity", "TCI_jp", "", "Ubiquity $K_{T, 0}$", "TCI", "center left", ],
    8: ["ubiquity", "ki_1", "", "Ubiquity $K_{T, 0}$", "The Average Diversity $K_{T, 1}$", "center left", ],
    # 7: ["ubiquity", "TCI_jp", "", "Degree centrality $k_{t, 0}$", "TCIs", "center left", ],
    # 8: ["ubiquity", "ki_1", "", "Degree centrality $k_{t, 0}$", "the average nearest neighbor degree $k_{t, 1}$", "center left", ],
    # 5: ["reg_num_eu", "TCI_eu", "corr between the patent amounts in EU and TCI in EU", "EU（period：1985-2009 year）", "EU（period：1985-2009 year）", "center", ],
    # 2: ["TCI_eu", "TCI_jp", "corr between the TCIs in Japan and EU", "EU（period：1985-2009 year）", "Japan（period：1981-2010 fiscal year）", "center", ],
}
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Meiryo'
for i, combi in combi_dict.items():
    fig, ax = plt.subplots(figsize=(6, 6))
    period = f"{year_start}-{year_end}"
    corr_num = round(eu_jp_df[combi[0]].corr(eu_jp_df[combi[1]]), 3)
    print(period, corr_num)
    # ax.scatter(eu_jp_df[combi[0]], eu_jp_df[combi[1]],
    #            s=20, alpha=0.8, color="black", )
    # if i == 4:
    ax.axvline(x=eu_jp_df[combi[0]].mean(), color="black", )
    ax.axhline(y=eu_jp_df[combi[1]].mean(), color="black", )
    # ax.axvline(x=eu_jp_df[combi[0]].mean(), color="gray", linestyle="--", )
    # ax.axhline(y=eu_jp_df[combi[1]].mean(), color="gray", linestyle="--", )
    ax.set_title(combi[2]+'(corr=' + r"$\bf{" + str(corr_num)+ "}$" +')\n')
    if combi[0] in ["reg_num"]: ax.set_xscale("log")
    if combi[1] in ["reg_num"]: ax.set_yscale("log")
    x_min = eu_jp_df[combi[0]].min()
    x_2smallest = (eu_jp_df[combi[0]].nsmallest(2).iloc[1])
    y_2smallest = (eu_jp_df[combi[1]].nsmallest(2).iloc[1])
    head_df = eu_jp_df.head(5)
    between_df = eu_jp_df.iloc[5:len(eu_jp_df)-5, :]
    tail_df = eu_jp_df.tail(5)
    if i != 5:
        # display(eu_jp_df)
        # for i, row in head_df.iterrows():
        #     ax.text(row[combi[0]], row[combi[1]], f'{i+1} {row["schmoch35"]}', fontsize=18, color="red")
        #     ax.scatter(row[combi[0]], row[combi[1]], s=20, color="red")
        # for i, row in between_df.iterrows():
        #     ax.text(row[combi[0]], row[combi[1]], f'{i+1} {row["schmoch35"]}', fontsize=15, color="black")
        #     ax.scatter(row[combi[0]], row[combi[1]], s=20, color="black")
        # for i, row in tail_df.iterrows():
        #     ax.text(row[combi[0]], row[combi[1]], f'{i+1} {row["schmoch35"]}', fontsize=18, color="blue", )
        #     ax.scatter(row[combi[0]], row[combi[1]], s=20, color="blue")
        # for i, row in head_df.iterrows():
        #     ax.text(row[combi[0]], row[combi[1]], f'{i+1} {row["schmoch35"]}', fontsize=18, color="red")
            
            # if i == 4: ax.scatter(row[combi[0]], row[combi[1]], s=40, color=tech_color[row['schmoch5']], label=row['schmoch5'])
            # else: ax.scatter(row[combi[0]], row[combi[1]], s=40, color=tech_color[row['schmoch5']])
        # for i, row in between_df.iterrows():
        #     # ax.text(row[combi[0]], row[combi[1]], i+1, fontsize=15, color="black")
        #     if i == 7: ax.scatter(row[combi[0]], row[combi[1]], s=40, color=tech_color[row['schmoch5']], label=row['schmoch5'])
        #     else: ax.scatter(row[combi[0]], row[combi[1]], s=40, color=tech_color[row['schmoch5']])
            
        # for i, row in tail_df.iterrows():
        #     # ax.text(row[combi[0]], row[combi[1]], i+1, fontsize=18, color="blue")
        #     ax.scatter(row[combi[0]], row[combi[1]], s=40, color="blue", label=f'{i+1} {row["schmoch35"]}')
        for tech_color_key in tech_color.keys():
            ax.scatter(eu_jp_df[eu_jp_df['schmoch5']==tech_color_key][combi[0]], eu_jp_df[eu_jp_df['schmoch5']==tech_color_key][combi[1]], 
                       color=tech_color[tech_color_key], label=tech_color_key, 
                       s=60)
        # for i, row in tail_df.iterrows():
            # ax.text(row[combi[0]]-len(row['schmoch35'])*10, row[combi[1]], row['schmoch35'], fontsize=18, color=tech_color[row['schmoch5']])
            # ax.text(row[combi[0]], row[combi[1]], row['schmoch35'], fontsize=18, color=tech_color[row['schmoch5']])
        # for ind, row in head_df.iterrows():
        #     if ind == 1: ax.text(row[combi[0]]+1, row[combi[1]]-2, f'\n{ind+1} {row["schmoch35"]}', fontsize=20, color=tech_color[row['schmoch5']])
        #     else: ax.text(row[combi[0]]+1, row[combi[1]]-1, f'{ind+1} {row["schmoch35"]}', fontsize=20, color=tech_color[row['schmoch5']])
    # elif i == 2:
    #     for i, row in head_df.iterrows():
    #         ax.text(row[combi[0]], row[combi[1]], i+1, fontsize=18, color="red")
    #         ax.scatter(row[combi[0]], row[combi[1]], s=20, color="red")
    #     for i, row in between_df.iterrows():
    #         ax.text(row[combi[0]], row[combi[1]], i+1, fontsize=15, color="black")
    #         ax.scatter(row[combi[0]], row[combi[1]], s=20, color="black")
    #     for i, row in tail_df.iterrows():
    #         ax.text(row[combi[0]], row[combi[1]], i+1, fontsize=18, color="blue", )
    #         ax.scatter(row[combi[0]], row[combi[1]], s=20, color="blue")
    ax.set_ylabel(combi[4])
    ax.set_xlabel(combi[3])
    # ax.set_xscale('log')
    ax.legend(loc=combi[5], fontsize=20, bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    # if i == 7: ax.legend(loc='lower right', prop={'weight': 'bold', 'size': 15}, labelspacing=1.25, borderaxespad=0, bbox_to_anchor=(1.25, 0.05))
    fig.savefig(f'{output_dir}{fig_name_base}', bbox_inches='tight')
    # fig.savefig(f'{output_dir}{fig_name_base.replace(".png", f"_{i}.eps")}', bbox_inches='tight')
    plt.show()

#%%
sample_df = eu_jp_df.copy()
sample_df_dict = {
    "0": sample_df.melt(id_vars=[classification], value_vars=["TCI_jp", "TCI_eu"]).sort_values(
        ["variable", "value"], ascending=[False, False])
}
sample_df_dict['0']

#%%
sample_df = eu_jp_df.copy()
sample_melt_df = sample_df.melt(id_vars=[classification], value_vars=["reg_num_jp", "reg_num_eu"]).sort_values(
        ["variable", "value"], ascending=[False, False])
sample_df_dict = {
    "0": sample_melt_df[sample_melt_df['variable']=='reg_num_jp'], 
    "1": sample_melt_df[sample_melt_df['variable']=='reg_num_eu'],
}



fs = (30, 30)

sample = vr.rank_doubleaxis(
    sample_df_dict,
    rank_num=33,
    member_col=classification,
    value_col="value",
    prop_dict={
        "figsize": fs,
        "xlabel": "",
        "ylabel": "",
        "title": "",
        "fontsize": 40,
        "year_range": 1,
        "ascending": False,
        # 'color': color_dict
        "color": "default",
    },
)
plt.title("Japan vs EU patent amount", fontsize=40)
plt.xticks(range(0, 2), ['Japan', 'EU'], rotation=90)

#%%
sample_df = eu_jp_df.copy()
sample_melt_df = sample_df.melt(id_vars=[classification], value_vars=["TCI_jp", "TCI_eu"]).sort_values(
        ["variable", "value"], ascending=[False, False])
sample_df_dict = {
    "0": sample_melt_df[sample_melt_df['variable']=='TCI_jp'], 
    "1": sample_melt_df[sample_melt_df['variable']=='TCI_eu'],
}

fs = (30, 30)

sample = vr.rank_doubleaxis(
    sample_df_dict,
    rank_num=33,
    member_col=classification,
    value_col="value",
    prop_dict={
        "figsize": fs,
        "xlabel": "",
        "ylabel": "",
        "title": "",
        "fontsize": 40,
        "year_range": 1,
        "ascending": False,
        # 'color': color_dict
        "color": "default",
    },
)
plt.title("Japan vs EU TCI", fontsize=40)

plt.xticks(range(0, 2), ['Japan', 'EU'], rotation=90)

#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import squarify

sns.set_style('darkgrid')
df_2col = df[df[f'{ar}_{year_style}_period'] == f'{year_start}-{year_end}'][
    ['schmoch35', 'reg_num']
].sort_values('reg_num', ascending=False)
# df_raw = pd.read_csv('https://github.com/selva86/datasets/raw/master/import squarify.csv')

# df_raw = pd.read_csv('https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv')
# display(df_raw)
# display(df_raw.groupby('class').size().reset_index(name='counts'))
labels = df_2col.apply(lambda x: str(x[0]) + '\n (' + str(round(x[1], 1)) + ')', axis=1)
sizes = df_2col['reg_num'].values.tolist()
colors = [plt.cm.Spectral(i / float(len(labels))) for i in range(len(labels))]
# colors = color_list
# display(sizes)
plt.figure(figsize=(40, 30), dpi=120)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8)

plt.title('Treemap of Vechile Class')
plt.axis('off')
plt.show()