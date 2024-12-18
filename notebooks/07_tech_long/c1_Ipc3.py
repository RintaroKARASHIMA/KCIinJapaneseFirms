#! (root)/notebooks/08_producer_sep/r1_Prefecture.py python3
# -*- coding: utf-8 -*-

#%%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

#%%
global data_dir, ex_dir, output_dir
data_dir = '../../data/processed/internal/tech/'
ex_dir = '../../data/processed/external/'
output_dir = '../../output/figures/'

jp_df = pd.read_csv(f'{data_dir}{input_condition}.csv', 
                    encoding='utf-8',
                    sep=','
                    )

# schmoch_df = pd.read_csv(f'{ex_dir}schmoch/35.csv', 
#                          encoding='utf-8', 
#                          sep=',', 
#                          usecols=['Field_number', 'Field_en']).drop_duplicates()

# jp_df = pd.merge(jp_df, schmoch_df, left_on=classification, right_on='Field_number', how='left')\
#         .drop(columns=['Field_number', classification])\
#         .rename(columns={'Field_en': classification})\
#         .sort_values(f'{ar}_{year_style}_period', key=lambda col: col.map(period_order_dict))
jp_df = jp_df.sort_values(f'{ar}_{year_style}_period', key=lambda col: col.map(period_order_dict))
jp_df
jp_df['schmoch5'] = jp_df['schmoch5'].replace('Mechanical engineering', 'Mechanical engineering, machinery')
jp_df['schmoch5'] = jp_df['schmoch5'].replace('Chemistry', 'Chemistry, pharmaceuticals')
jp_df = jp_df[jp_df[f'{ar}_{year_style}_period'] == '1981-2010']
jp_df['tci'] = 100 * (jp_df['tci'] - jp_df['tci'].min()) / (jp_df['tci'].max() - jp_df['tci'].min())

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
    7: ["ubiquity", "tci", "", "Degree centrality $k_{t, 0}$", "TCIs", "center left", ],
    8: ["ubiquity", "ki_1", "", "Degree centrality $k_{t, 0}$", "the average nearest neighbor degree $k_{t, 1}$", "center left", ],
    # 5: ["reg_num_eu", "TCI_eu", "corr between the patent amounts in EU and TCI in EU", "EU（period：1985-2009 year）", "EU（period：1985-2009 year）", "center", ],
    # 2: ["TCI_eu", "TCI_jp", "corr between the TCIs in Japan and EU", "EU（period：1985-2009 year）", "Japan（period：1981-2010 fiscal year）", "center", ],
}
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Meiryo'
for i, combi in combi_dict.items():
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
    period = f"{year_start}-{year_end}"
    count = 0
    for s5, color in tech_color.items():
        target_df = jp_df[(jp_df['app_nendo_period']==period)&(jp_df['schmoch5'] == s5)]\
                    .drop_duplicates(subset=[combi[0], combi[1]])
        ax[count//3, count%3].scatter(target_df[combi[0]], target_df[combi[1]], color=color, label=s5)
        ax[1, 2].scatter(target_df[combi[0]], target_df[combi[1]], color='black')
        
        corr_num = round(target_df[combi[0]].corr(target_df[combi[1]]), 3)
        print(period, corr_num)
        
        ax[count//3, count%3].axvline(x=target_df[combi[0]].mean(), color="gray", linestyle="--", )
        ax[count//3, count%3].axhline(y=target_df[combi[1]].mean(), color="gray", linestyle="--", )
        if i == 7: ax[count//3, count%3].text(250, 0, f'corr={corr_num}', fontsize=20)
        else: ax[count//3, count%3].text(250, 5, f'corr={corr_num}', fontsize=20)
        ax[count//3, count%3].set_xlim(-100, 600)
        if i == 7: ax[count//3, count%3].set_ylim(-10, 110)
        else: ax[count//3, count%3].set_ylim(0, 30)
        # ax.set_title(combi[2]+'(corr=' + r"$\bf{" + str(corr_num)+ "}$" +')\n')
        if combi[0] in ["reg_num"]: ax.set_xscale("log")
        if combi[1] in ["reg_num"]: ax.set_yscale("log")
        count += 1
    ax[1, 2].axvline(x=jp_df[(jp_df['app_nendo_period']==period)]\
                            .drop_duplicates(subset='ipc3', keep='first')\
                            [combi[0]].mean(), color="gray", linestyle="--", )
    ax[1, 2].axhline(y=jp_df[(jp_df['app_nendo_period']==period)]\
                            .drop_duplicates(subset='ipc3', keep='first')\
                            [combi[1]].mean(), color="gray", linestyle="--", )
    corr_num = round(jp_df[(jp_df['app_nendo_period']==period)]\
                            .drop_duplicates(subset='ipc3', keep='first')\
                            [combi[0]].corr(jp_df[(jp_df['app_nendo_period']==period)]\
                                            .drop_duplicates(subset='ipc3', keep='first')\
                                            [combi[1]]), 
                            3)
    if i == 7:ax[1, 2].text(250, 0, f'corr={corr_num}', fontsize=20)
    else:ax[1, 2].text(250, 5, f'corr={corr_num}', fontsize=20)
    
    # fig.delaxes(ax[1, 2])
    fig.text(0.5, 0, 'Degree centrality $k_{t, 0}$', ha='center', va='center', fontsize=30)
    if i == 7:
        fig.text(0.05, 0.5, 'TCIs', ha='center', va='center', rotation='vertical', fontsize=30)
    else:
        fig.text(0.05, 0.5, 'Average neighbor degree $k_{t, 1}$', ha='center', va='center', rotation='vertical', fontsize=30)
    # ax.set_xscale('log')
    # ax.legend(loc=combi[5], fontsize=20, bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    # if i == 7: ax.legend(loc='lower right', prop={'weight': 'bold', 'size': 15}, labelspacing=1.25, borderaxespad=0, bbox_to_anchor=(1.25, 0.05))
    # fig.savefig(f'{output_dir}{fig_name_base.replace(".png", f"_{i}.eps")}', bbox_inches='tight')
    plt.show()

