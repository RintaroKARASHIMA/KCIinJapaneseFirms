#! (root)/notebooks/05_calculation/2_Complexity.py python3
# -*- coding: utf-8 -*-

# %%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py
# Import Libraries
from ecomplexity import ecomplexity
from ecomplexity import proximity
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
# classification = 'ipc3'
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
    f"{DATA_DIR}app_nendo_1981_2010_5_all_p_3_right_person_name_fraction_ipc3_fraction.csv", encoding="utf-8", sep=","
)
display(reg_num_top_df.head())
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

p_df = proximity(reg_num_top_df, cols_input=trade_cols, rca_mcp_threshold=1)
display(p_df.head())

#%%
p_df['ipc3_1'].nunique()
classification_p_df = p_df.groupby('ipc3_1', as_index=False).sum()[['ipc3_1', 'proximity']]\
                          .assign(
                              proximity=lambda x: x['proximity']/123
                          )
classification_p_df

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
# import matplotlib.pyplot as plt
# tp_df = pd.merge(classification_df[classification_df[f"{ar}_{year_style}_period"] == '1981-2010'], 
#                  classification_p_df, 
#                  left_on=classification, right_on='ipc3_1', how='inner', 
#                  copy=False)\
#                  [['ipc3', 'tci', 'proximity']].drop_duplicates(keep='first')
# plt.scatter(tp_df['tci'], tp_df['proximity'])
# print(tp_df['tci'].corr(tp_df['proximity']))

data_dir = '../../data/interim/internal/filtered_before_agg/'
filter_dir = '../../data/interim/internal/filter_after_agg/'
output_dir = '../../data/interim/internal/filtered_after_agg/'
all_df = pd.read_csv(f'{data_dir}japan.csv', 
                     encoding='utf-8', 
                     sep=',', 
                     usecols=['right_person_name',
                               'app_nendo',
                               'right_person_addr'],
                     dtype={
                            'right_person_name': str,
                            'app_nendo': np.int64,
                            'right_person_addr': str
                     })\
            .query('1981 <= app_nendo <= 2010')\
            .query('right_person_addr in ("東京都", "大阪府", "愛知県")')
all_df

trade_cols = {
    "time": f"{ar}_{year_style}_period",
    "loc": region_corporation,
    "prod": 'ipc3',
    "val": "reg_num",
}
rename_col_dict = {"eci": "kci", "pci": "tci"}
col_order_list = [
    f"{ar}_{year_style}_period",
    region_corporation,
    'ipc3',
    "reg_num",
    "rca",
    "mcp",
    "diversity",
    "ubiquity",
    "kci",
    "tci",
]
# %%
all_df = all_df[all_df['right_person_name'].isin(c_df['right_person_name'].unique())]
tokyo_df = all_df.query('right_person_addr == "東京都"')
osaka_df = all_df.query('right_person_addr == "大阪府"')
aichi_df = all_df.query('right_person_addr == "愛知県"')

#%%
display(reg_num_top_df.head())
#%%
tokyo_df['right_person_name'].nunique(), osaka_df['right_person_name'].nunique()
#%%
tokyo_c_df = ecomplexity(reg_num_top_df[reg_num_top_df['right_person_name'].isin(tokyo_df['right_person_name'].unique())], 
                        cols_input=trade_cols, rca_mcp_threshold=1)\
                        .sort_values(by=['pci'], ascending=False)
osaka_c_df = ecomplexity(reg_num_top_df[reg_num_top_df['right_person_name'].isin(osaka_df['right_person_name'].unique())],
                        cols_input=trade_cols, rca_mcp_threshold=1)\
                        .sort_values(by=['pci'], ascending=False)
aichi_c_df = ecomplexity(reg_num_top_df[reg_num_top_df['right_person_name'].isin(aichi_df['right_person_name'].unique())],
                        cols_input=trade_cols, rca_mcp_threshold=1)\
                        .sort_values(by=['pci'], ascending=False)
#%%
display(tokyo_c_df.head())
# osaka_c_df= osaka_c_df[osaka_c_df['pci']!=osaka_c_df['pci'].min()]
# %%
import matplotlib.pyplot as plt
tokyo_osaka_df = pd.merge(tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
                          osaka_c_df[osaka_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
                          on='ipc3', how='inner', copy=False)

tokyo_osaka_df = pd.merge(tokyo_osaka_df,
                          aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
                          on='ipc3', how='inner', copy=False)\
                          .rename(columns={'pci_x': 'pci_x', 'pci_y': 'pci_y', 'pci': 'pci_z'})\
                          .query('ipc3 not in ("A63", "G07")')

# %%
aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')
# %%
all_vs_3pref_df = pd.merge(tokyo_osaka_df,
                           c_df[c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'tci']].drop_duplicates(keep='first').sort_values(by='tci', ascending=False),
                           on='ipc3', how='inner', copy=False)\
                         .assign(pci_x=lambda x: (x['pci_x']-x['pci_x'].min())/(x['pci_x'].max()-x['pci_x'].min())*100,
                                 pci_y=lambda x: (x['pci_y']-x['pci_y'].min())/(x['pci_y'].max()-x['pci_y'].min())*100,
                                 pci_z=lambda x: (x['pci_z']-x['pci_z'].min())/(x['pci_z'].max()-x['pci_z'].min())*100,
                                 tci=lambda x: (x['tci']-x['tci'].min())/(x['tci'].max()-x['tci'].min())*100)\
                         .drop_duplicates(keep='first')

         
#%%
aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')
# tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')

#%%
plt.rcParams['font.size'] = 25
fig, (ax_tokyo, ax_osaka, ax_aichi) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
ax_tokyo.scatter(all_vs_3pref_df['pci_x'], all_vs_3pref_df['tci'], color='black', s=60)
ax_osaka.scatter(all_vs_3pref_df['pci_y'], all_vs_3pref_df['tci'], color='black', s=60)
ax_aichi.scatter(all_vs_3pref_df['pci_z'], all_vs_3pref_df['tci'], color='black', s=60)
ax_tokyo.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')
ax_osaka.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')
ax_aichi.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')
ax_tokyo.set_xlabel('Corporate TCI in Tokyo')
ax_osaka.set_xlabel('Corporate TCI in Osaka')
ax_aichi.set_xlabel('Corporate TCI in Aichi')
ax_tokyo.set_ylabel('Corporate TCI in Japan')
plt.show()

#%%
all_df.query('right_person_addr == "愛知県"')['right_person_name'].nunique()
# %%
combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    # 3: ["reg_num_jp", "TCI_jp", "relation between the patent counts and the TCIs in Japan", "Patent Counts", "TCIs", "center left", ],
    # 4: ["TCI_jp", "reg_num_jp", "relation between the patent counts and the TCIs in Japan", "TCIs", "Patent Counts", "center left", ],
    # 6: ["TCI_jp", "ubiquity", "relation between the ubiquity and the TCIs in Japan", "TCIs", "Ubiquity", "center left", ],
    # 7: ["ubiquity", "tci", "", "Degree Centrality $K_{T, 0}$", "TCI", "center left", ],
    7: ["pci_x", "pci", "", "TOKYO", "OSAKA", "center left", ],
    8: ["pci_y", "pci", "", "TOKYO", "ALL PREFECTURE", "center left", ], 
}
tokyo_aichi_df = pd.merge(
    aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
    tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first'), 
    on=['ipc3'], 
    how='inner'
).rename(
    columns={'pci_x':'aichi', 'pci_y':'tokyo'}
).query('aichi not in aichi.nsmallest(2)', engine='python')\
.assign(
    aichi = lambda df: ((df['aichi']-df['aichi'].min())/(df['aichi'].max()-df['aichi'].min()))*100,
    tokyo = lambda df: ((df['tokyo']-df['tokyo'].min())/(df['tokyo'].max()-df['tokyo'].min()))*100,
)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(
    tokyo_aichi_df['aichi'], 
    tokyo_aichi_df['tokyo'], 
    c='black'
)
ax.set_aspect('equal')
ax.set_xticks(range(0, 100+1, 20))
ax.set_xticklabels(range(0,100+1, 20))
ax.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--')
ax.set_xlabel('Corporate TCI in Aichi')
ax.set_ylabel('Corporate TCI in Tokyo')
plt.show()
# display(tokyo_aichi_df)
# display(aichi_c_df[aichi_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')['pci'])
# display(tokyo_c_df[tokyo_c_df[f'{ar}_{year_style}_period']=='1981-2010'][['ipc3', 'pci']].drop_duplicates(keep='first')['pci'])
#%%
tokyo_aichi_df.query('tokyo < 40')\
    .query('aichi > tokyo').to_clipboard()

#%%
for i, combi in combi_dict.items():
    plot_df = df[[combi[0], combi[1]]].drop_duplicates()
    fig, ax = plt.subplots(figsize=(8, 8))
    period = f'{year_start}-{year_end}'
    corr_num = round(plot_df[combi[0]].corr(plot_df[combi[1]]), 3)
    print(period, corr_num)

    # ax.set_title(combi[2]+'(corr=' + r"$\bf{" + str(corr_num)+ "}$" +')\n')
    
    # scale if necessary
    if combi[0] in ["reg_num"]: ax.set_xscale('log')
    if combi[1] in ["reg_num"]: ax.set_yscale('log')

    x_min = plot_df[combi[0]].min()
    x_2smallest = (plot_df[combi[0]].nsmallest(2).iloc[1])
    y_2smallest = (plot_df[combi[1]].nsmallest(2).iloc[1])
    head_df = plot_df.head(5)
    between_df = plot_df.iloc[5:len(df)-5, :]
    tail_df = plot_df.tail(5)
    
    ax.scatter(
                x=combi[0], y=combi[1], 
                data=df,
                color='black', 
                s=60)
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    min_limit = min(x_limits[0], y_limits[0])
    max_limit = max(x_limits[1], y_limits[1])
    ax.plot([min_limit, max_limit], [min_limit, max_limit], color='red', linewidth=2, linestyle='--')
    
    # plot the mean values
    # ax.axvline(x=df[combi[0]].mean(), color='black', )
    # ax.axhline(y=df[combi[1]].mean(), color='black', )

    ax.set_ylabel(combi[4])
    ax.set_xlabel(combi[3])
    # ax.legend(loc=combi[5], fontsize=20, bbox_to_anchor=(1.05, 1.5), borderaxespad=0)
    # if i == 7: ax.legend(loc='lower right', prop={'weight': 'bold', 'size': 15}, labelspacing=1.25, borderaxespad=0, bbox_to_anchor=(1.25, 0.05))
    # fig.savefig('../../outputs/charts/', bbox_inches='tight')
    # fig.savefig(f'{output_dir}{fig_name_base.replace(".png", f"_{i}.eps")}', bbox_inches='tight')
    # plt.tight_layout(pad=1.08, h_pad=1.08, w_pad=1.08)
    plt.show()


# %%
