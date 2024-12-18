#! (root)/notebooks/07_tech_long/r1_Schmoch.py python3
# -*- coding: utf-8 -*-

#%%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

#%%
global data_dir, ex_dir, output_dir
data_dir = '../../data/processed/internal/tech/'
ex_dir = '../../data/processed/external/'
output_dir = '../../output/figures/'

#%%
df = pd.read_csv(
                    f'{DATA_DIR}{input_condition}.csv', 
                    encoding='utf-8',
                    sep=','
                    )

schmoch_df = pd.read_csv(
                         f'{EX_DIR}schmoch/35.csv', 
                         encoding='utf-8', 
                         sep=',', 
                         )\
                         .drop_duplicates()

df = pd.merge(df, schmoch_df, 
              left_on=classification, right_on='Field_en', 
              how='left'
              )\
            .drop(columns=['Field_number', classification])\
            .rename(columns={'Field_en': classification})\
            .query(f'{ar}_{year_style}_period == "{year_start}-{year_end}"')\
            .assign(
            tci = lambda x: minmax_scale(x['tci']) * 100,
            )\
            .sort_values(by='tci', ascending=False)

display(df)

# %%
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
    7: ["ubiquity", "tci", "", "Ubiquity $K_{T, 0}$", "TCI", "center left", ],
    # 8: ["ubiquity", "ki_1", "", "Ubiquity $K_{T, 0}$", "The Average Diversity $K_{T, 1}$", "center left", ],
    8: ["ubiquity", "ki_1", "", "Degree Centrality $K_{T, 0}$\n\n", "The Average Nearest Neighbor Degree $K_{T, 1}$", "upper left", ],
    # 7: ["ubiquity", "TCI_jp", "", "Degree centrality $k_{t, 0}$", "TCIs", "center left", ],
    # 8: ["ubiquity", "ki_1", "", "Degree centrality $k_{t, 0}$", "the average nearest neighbor degree $k_{t, 1}$", "center left", ],
}
for i, combi in combi_dict.items():
    plot_df = df[[combi[0], combi[1], 'schmoch5']].drop_duplicates()
    fig, ax = plt.subplots(figsize=(6, 6))
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
    
    if i != 5:
        for tech_color_key in tech_color.keys():
            ax.scatter(
                        x=combi[0], y=combi[1], 
                        data=df[df['schmoch5']==tech_color_key],
                        color=tech_color[tech_color_key], 
                        label=tech_color_key, 
                        s=60)

    
    # plot the mean values
    ax.axvline(x=df[combi[0]].mean(), color='black', )
    ax.axhline(y=df[combi[1]].mean(), color='black', )

    ax.set_ylabel(combi[4])
    ax.set_xlabel(combi[3])
    ax.legend(loc=combi[5], fontsize=20, bbox_to_anchor=(1.05, 1.5), borderaxespad=0)
    # if i == 7: ax.legend(loc='lower right', prop={'weight': 'bold', 'size': 15}, labelspacing=1.25, borderaxespad=0, bbox_to_anchor=(1.25, 0.05))
    # fig.savefig('../../outputs/charts/', bbox_inches='tight')
    # fig.savefig(f'{output_dir}{fig_name_base.replace(".png", f"_{i}.eps")}', bbox_inches='tight')
    # plt.tight_layout(pad=1.08, h_pad=1.08, w_pad=1.08)
    plt.show()
