#! (root)/notebooks/08_producer_sep/r1_Prefecture.py python3
# -*- coding: utf-8 -*-

# %%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py
DATA_DIR = '../../data/processed/internal/tech_comparison/'

# %%
path_list = glob(DATA_DIR + '/*')
name_df = pd.read_csv(path_list[0],
                      encoding='utf-8',
                      sep=',')\
            .sort_values('schmoch35_tci', ascending=False)
addr_df = pd.read_csv(path_list[1],
                      encoding='utf-8',
                      sep=',')\
            .sort_values('schmoch35_tci', ascending=False)

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
    1: ["co_tci", "pre_tci", "relation between the TCIs in Japanese Corporations and Prefectures", "Corporations（period：1981-2010 fiscal year）", "Prefectures（period：1981-2010 fiscal year）", "center", ],
    2: ["co_tci_rank", "pre_tci_rank", "relation between the TCI rankings in Japanese Corporations and Prefectures", "Corporations（period：1981-2010 fiscal year）", "Prefectures（period：1981-2010 fiscal year）", "center", ],
    # 2: ["TCI_rank_jp", "TCI_rank_eu", "relation between the TCIs in Japanese corporation and EU regions", "Japanese Corporations ranking（period：1981-2010 fiscal year）", "EU Regions ranking（period：1985-2009 year）", "center", ],
    # 2: ["reg_num_jp", "reg_num_eu", "corr between the patent amounts in Japan and EU", "Japan（period：1981-2010 fiscal year）", "EU（period：1985-2009 year）", "center", ],
    # 3: ["reg_num_jp", "TCI_jp", "relation between the patent counts and the TCIs in Japan", "Patent Counts", "TCIs", "center left", ],
    # 4: ["TCI_jp", "reg_num_jp", "relation between the patent counts and the TCIs in Japan", "TCIs", "Patent Counts", "center left", ],
    # 5: ["reg_num_eu", "TCI_eu", "corr between the patent amounts in EU and TCI in EU", "EU（period：1985-2009 year）", "EU（period：1985-2009 year）", "center", ],
    # 2: ["TCI_eu", "TCI_jp", "corr between the TCIs in Japan and EU", "EU（period：1985-2009 year）", "Japan（period：1981-2010 fiscal year）", "center", ],
}
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Meiryo'
for i, combi in combi_dict.items():
    fig, ax = plt.subplots(figsize=(8, 8))
    period = f"{year_start}-{year_end}"
    corr_num = round(pre_co_df[combi[0]].corr(pre_co_df[combi[1]]), 3)
    print(period, corr_num)
    # ax.scatter(pre_co_df[combi[0]], pre_co_df[combi[1]],
    #            s=20, alpha=0.8, color="black", )
    # if i == 4:
    ax.axvline(x=pre_co_df[combi[0]].mean(), color="gray", linestyle="--", )
    ax.axhline(y=pre_co_df[combi[1]].mean(), color="gray", linestyle="--", )
    ax.set_title(combi[2]+'(corr=' + r"$\bf{" + str(corr_num)+ "}$" +')\n')
    if combi[0] in ["reg_num"]: ax.set_xscale("log")
    if combi[1] in ["reg_num"]: ax.set_yscale("log")
    x_min = pre_co_df[combi[0]].min()
    x_2smallest = (pre_co_df[combi[0]].nsmallest(2).iloc[1])
    y_2smallest = (pre_co_df[combi[1]].nsmallest(2).iloc[1])
    head_df = pre_co_df.head(5)
    between_df = pre_co_df.iloc[5:len(pre_co_df)-5, :]
    tail_df = pre_co_df.tail(5)
    if i != 5:
        # display(pre_co_df)
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
            ax.scatter(pre_co_df[pre_co_df['schmoch5']==tech_color_key][combi[0]], pre_co_df[pre_co_df['schmoch5']==tech_color_key][combi[1]], 
                       color=tech_color[tech_color_key], label=tech_color_key, 
                       s=60)
        # for i, row in pre_co_df.iterrows():
        #     ax.text(row[combi[0]], row[combi[1]], f'{i+1} {row["schmoch35"]}', fontsize=18, color="black")
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
    ax.legend(loc=combi[5], fontsize=20, bbox_to_anchor=(1.65, 0.5), borderaxespad=0, prop={'weight': 'bold'})
    plt.show()
