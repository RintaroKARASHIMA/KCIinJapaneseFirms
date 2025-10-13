#! (root)/notebooks/08_producer_sep/c1_Trends.py python3
# -*- coding: utf-8 -*-

# %%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py
# %%
global data_dir, ex_dir, output_dir
data_dir = "../../data/processed/internal/corporations/"
ex_dir = "../../data/processed/external/schmoch/"
output_dir = "../../output/figures/"


df = pd.read_csv(f'{data_dir}{input_condition}.csv')
df

# %%
df_dict = {}
combi_dict = {  # ind: [x, y, title, xlabel, ylabel, legend_loc]
    1: ['reg_num', 'diversity', '特許数と法人次数（Diversity）の相関', '特許数（対数スケール）', '法人次数（Diversity）', 'upper left'],
    2: ['reg_num', 'kci', '特許数とKCIの相関', '特許数（対数スケール）', 'KCI', 'lower left'],
    3: ['diversity', 'kci', '法人次数（Diversity）とKCIの相関', '法人次数（Diversity）', 'KCI', 'lower right'],
    4: ['diversity', 'kh_1', '法人次数（Diversity）と法人平均近傍次数（kh_1）の相関', '法人次数（Diversity）', '法人平均近傍次数（kh_1）', 'lower right']
}

for i, combi in combi_dict.items():
    fig, ax = plt.subplots(figsize=(12, 12))
    color_count = 0
    for period in range(year_start, year_end+1, year_range):
        period = f'{period}-{period+year_range-1}'
        corr_num = round(df[df[f'{ar}_{year_style}_period'] == period][combi[0]].corr(
            df[df[f'{ar}_{year_style}_period'] == period][combi[1]]), 3)
        print(period, corr_num)
        ax.scatter(df[df[f'{ar}_{year_style}_period'] == period][combi[0]],
                   df[df[f'{ar}_{year_style}_period']
                       == period][combi[1]], s=20,
                   alpha=0.8, label=f'{period}年度（{corr_num}）', color=color_list[color_count])
        if i == 4:
            ax.axvline(x=df[df[f'{ar}_{year_style}_period'] == period][combi[0]].mean(
            ), color=color_list[color_count], linestyle='--')
            ax.axhline(y=df[df[f'{ar}_{year_style}_period'] == period][combi[1]].mean(
            ), color=color_list[color_count], linestyle='--')
        ax.set_title(combi[2])
        if combi[0] in ['reg_num']:
            ax.set_xscale('log')
        if combi[1] in ['reg_num']:
            ax.set_yscale('log')
        ax.set_ylabel(combi[4])
        ax.set_xlabel(combi[3])
        ax.legend(loc=combi[5])
        color_count += 1

    plt.savefig(
        f'{output_dir}co_corr/{combi[0]}_{combi[1]}_{fig_name_base}', bbox_inches="tight")
    plt.show()

# %%
filtered_df = pd.read_csv(
    '../../data/interim/internal/filtered_before_agg/addedclassification.csv', sep=',')
filtered_df
# filtered_df['right_person_name'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(
    range(year_start, year_end+1))]['right_person_name'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1)) & (
    filtered_df['right_person_name'].isin(df['right_person_name']))]['right_person_name'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1)) & (filtered_df['right_person_name'].isin(df['right_person_name']))
            ]['right_person_name'].nunique()*100 / filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))]['right_person_name'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1)) & (
    filtered_df['right_person_name'].isin(df['right_person_name']))]['reg_num'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(
    range(year_start, year_end+1))]['reg_num'].nunique()
filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1)) & (filtered_df['right_person_name'].isin(
    df['right_person_name']))]['reg_num'].nunique() / filtered_df[filtered_df[f'{ar}_{year_style}'].isin(range(year_start, year_end+1))]['reg_num'].nunique()
