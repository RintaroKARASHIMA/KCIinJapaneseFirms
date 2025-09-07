#! (root)/notebooks/07_tech_long/c1_Schmoch.py python3
# -*- coding: utf-8 -*-

#%%
%run ../../src/initialize/load_libraries.py
%run ../../src/initialize/initial_conditions.py
%run 0_LoadLibraries.py

import scipy.stats as stats

DATA_DIR = '../../data/processed/internal/05_2_4_tech/'
EX_DIR = '../../data/processed/external/'

# print(condition)
condition = 'app_nendo_1981_2010_5_all_p_3_right_person_name_fraction_schmoch35_fraction'

# %%
period_order_dict = {
    f'{period_start}-{period_start+year_range-1}': i
    for i, period_start in enumerate(range(year_start, year_end + 1, year_range))
}

period_order_dict[f'{year_start}-{year_end}'] = len(period_order_dict)
period_order_dict

# %%
df = pd.read_csv(
                    f'{DATA_DIR}{condition}.csv', 
                    encoding='utf-8',
                    sep=','
                    )

schmoch_df = pd.read_csv(
                         f'{EX_DIR}schmoch/35.csv', 
                         encoding='utf-8', 
                         sep=',', 
                         )\
                         .drop_duplicates()

# df = pd.merge(df, schmoch_df, 
#               left_on=classification, right_on='Field_en', 
#               how='left'
#               )\
#             .drop(columns=['Field_number', classification])\
#             .rename(columns={'Field_en': classification})\
#             .query(f'{ar}_{year_style}_period == "{year_start}-{year_end}"')\
#             .assign(
#             tci = lambda x: ((x['tci']-x['tci'].min())/(x['tci'].max()-x['tci'].min())) * 100,
#             )\
#             .sort_values(by='tci', ascending=False)
sep_df = df.copy().query(f'{ar}_{year_style}_period != "{year_start}-{year_end}"')\
            .drop_duplicates()
df = df.query(f'{ar}_{year_style}_period == "{year_start}-{year_end}"')\
            .drop_duplicates()\
            .sort_values(by='tci', ascending=False)\
            # .assign(
            # tci = lambda x: ((x['tci']-x['tci'].min())/(x['tci'].max()-x['tci'].min())) * 100,
            # )
display(df)

# %%
fig2, ax2 = plt.subplots(figsize=(10, 12))

tech_color = {
        'Chemistry, pharmaceuticals': 'red',
        'Electrical engineering': 'blue',
        'Instruments': 'green', 
        'Mechanical engineering, machinery': 'orange',
        'Other fields': 'gray'
    }
tech_shape = {
        'Chemistry, pharmaceuticals': 'o',
        'Electrical engineering': 's',
        'Instruments': '^', 
        'Mechanical engineering, machinery': '*',
        'Other fields': '|'
    }

combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    # 3: ["reg_num_jp", "TCI_jp", "relation between the patent counts and the TCIs in Japan", "Patent Counts", "TCIs", "center left", ],
    # 4: ["TCI_jp", "reg_num_jp", "relation between the patent counts and the TCIs in Japan", "TCIs", "Patent Counts", "center left", ],
    # 6: ["TCI_jp", "ubiquity", "relation between the ubiquity and the TCIs in Japan", "TCIs", "Ubiquity", "center left", ],
    # 7: ["ubiquity", "tci", "", "Degree Centrality $K_{T, 0}$", "TCI", "center left", ],
    6: ["reg_num", "tci", "", "Patent Counts", "TCI", "center left", ],
    7: ["ubiquity", "tci", "", "Ubiquity $K_{T, 0}$", "TCI", "center left", ],
    # 8: ["ubiquity", "ki_1", "", "Ubiquity $K_{T, 0}$", "The Average Diversity $K_{T, 1}$", "center left", ],
    8: ["ubiquity", "ki_1", "", "Degree Centrality $K_{T, 0}$\n\n", "The Average Diversity $K_{T, 1}$", "upper left", ],
    9: ["tci", "ki_1", "", "TCI","The Average Diverstity $K_{T, 1}$",  "upper left", ],
    # 7: ["ubiquity", "TCI_jp", "", "Degree centrality $k_{t, 0}$", "TCIs", "center left", ],
    # 8: ["ubiquity", "ki_1", "", "Degree centrality $k_{t, 0}$", "the average nearest neighbor degree $k_{t, 1}$", "center left", ],
}
for i, combi in combi_dict.items():
    plot_df = df[[combi[0], combi[1], 'schmoch5']].drop_duplicates()
    fig, ax = plt.subplots(figsize=(6, 6))
    period = f'{year_start}-{year_end}'
    # corr_num = round(plot_df[combi[0]].corr(plot_df[combi[1]]), 3)
    # print(period, corr_num)

    # 相関係数とp値を計算
    correlation, p_value = stats.pearsonr(plot_df[combi[0]], plot_df[combi[1]])

    print('Correlation:', correlation)
    print('p-value:', p_value)


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
    
    if i != 5:
        for tech_color_key in tech_shape.keys():
            ax.scatter(
                        x=combi[0], y=combi[1], 
                        data=df[df['schmoch5']==tech_color_key],
                        # color=tech_color[tech_color_key], 
                        color='black',
                        label=tech_color_key, 
                        marker=tech_shape[tech_color_key],
                        s=60)

    
    # plot the mean values
    if i != 9: ax.axvline(x=df[combi[0]].mean(), color='black', )
    else: ax.axhline(y=df[combi[1]].mean(), color='black', )
    if i in [6, 7]: ax.axhline(y=0, color='black', )
    elif i in [9]: ax.axvline(x=0, color='black', )
    else: ax.axhline(y=df[combi[1]].mean(), color='black', )

    ax.set_ylabel(combi[4])
    ax.set_xlabel(combi[3])
    ax.legend(loc=combi[5], fontsize=20, bbox_to_anchor=(1.05, 1.5), borderaxespad=0)
    # if i == 7: ax.legend(loc='lower right', prop={'weight': 'bold', 'size': 15}, labelspacing=1.25, borderaxespad=0, bbox_to_anchor=(1.25, 0.05))
    # fig.savefig('../../outputs/charts/', bbox_inches='tight')
    # fig.savefig(f'{output_dir}{fig_name_base.replace(".png", f"_{i}.eps")}', bbox_inches='tight')
    # plt.tight_layout(pad=1.08, h_pad=1.08, w_pad=1.08)
    # plt.show()

    kt0_kt1_corr_list = []
    kt0_tci_corr_list = []
    kt1_tci_corr_list = []
    period_list = []
    for period in sep_df[f'{ar}_{year_style}_period'].unique():
        plot_df = sep_df.query(f'{ar}_{year_style}_period == "{period}"').drop_duplicates(subset=classification)
        corr_num = round(plot_df[combi[0]].corr(plot_df[combi[1]]), 3)
        print(period, corr_num)
        if i == 8: kt0_kt1_corr_list.append(corr_num)
        elif i == 7: kt0_tci_corr_list.append(corr_num)
        elif i == 9: kt1_tci_corr_list.append(corr_num)
        period_list.append(period)
        # 相関係数とp値を計算
        # correlation, p_value = stats.pearsonr(plot_df['ubiquity'], plot_df['tci'])

        # print('Correlation:', correlation)
        # print('p-value:', p_value)
    ax2.plot(
        kt0_kt1_corr_list, 
        label='$K_{T, 0}$ and $K_{T, 1}$' if i == 7 else None, 
        color='red',
        linewidth=3
        )
    ax2.plot(
            kt0_tci_corr_list, 
            label='$K_{T, 0}$ and TCI' if i == 8 else None, 
            color='blue', 
            linewidth=3
            )
    ax2.plot(
            kt1_tci_corr_list, 
            label='TCI and $K_{T, 1}$' if i == 9 else None,
            color='green',
            linewidth=3
            )
    ax2.axhline(y=correlation, color='black')
    
ax2.axhspan(0.2, 0.4, color='lightgray', alpha=0.3)
ax2.axhspan(-0.2, -0.4, color='lightgray', alpha=0.3)
ax2.axhspan(0.4, 0.7, color='gray', alpha=0.25)
ax2.axhspan(-0.7, -0.4, color='gray', alpha=0.25)


ax2.set_ylim(-0.7, 0.7)
ax2.set_xticks(range(len(period_list)))
ax2.set_xticklabels(period_list, rotation=45, size=20)
ax2.set_yticks([_/10 for _ in range(-7,7+1) if _ not in [-7, -5, -3, -1, 1, 3, 5, 7]])
ax2.set_yticklabels([_/10 for _ in range(-7,7+1) if _ not in [-7, -5, -3, -1, 1, 3, 5, 7]], size=20)
ax2.set_ylabel('Correlation Coefficient', size=20)
ax2.set_xlabel('Period', size=20)
ax2.legend(loc='center left', fontsize=20, bbox_to_anchor=(1.05, 1.5), borderaxespad=0)

plt.show()

#%%
# df[df['schmoch35'] == 'Biotechnology']
df[(df['ubiquity'] <400) & (df['ubiquity'] >300)]
# %%
