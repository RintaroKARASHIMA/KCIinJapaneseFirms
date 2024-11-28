#! (root)/notebooks/10_network_long/p1_CorporationsFiner.py python3
# -*- coding: utf-8 -*-

#%%
# %load 0_LoadLibraries.py

#%%
# color_count = 0
# fig, ax = plt.subplots(figsize=(8, 8))
# for s in list(right_person_df['segment'].unique())[0:1]:
#     ccdf_array = ccdf(right_person_df[right_person_df['segment']==s]['diversity'].to_list())
#     ax.plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=7, 
#                     color=color_list[color_count], label=s+'年度', alpha=0.6)
#     color_count += 1
#     # print(right_person_df[right_person_df['segment']==s]['diversity'].mean())
#     # print(right_person_df[right_person_df['segment']==s]['right_person_name'].nunique())
#     # print(right_person_df[right_person_df['segment']==s]['diversity'].mean() * right_person_df[right_person_df['segment']==s]['right_person_name'].nunique())
#     # print(right_person_df[right_person_df['segment']==s]['diversity'].mean() * right_person_df[right_person_df['segment']==s]['right_person_name'].nunique()/627)
#     # print(right_person_df[right_person_df['segment']==s]['diversity'].mean())
# ax.legend(loc='lower left', fontsize=18)
# # ax.legend(loc='upper right', fontsize=18)

# # ax.set_title('各期間における特許権者の補累積次数（Diversity）分布（両対数スケール）'+'\n', fontsize=20)
# ax.set_xlabel('特許権者次数（Diversity）', fontsize=18)
# ax.set_ylabel('ccdf', fontsize=18)

# ax.set_xscale('log')
# ax.set_yscale('log')

# ax.tick_params(labelsize=18)
# ax.set_xlim(0.8, 300)

# # x軸の指数表記を普通に戻す魔法
# ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# # ax.set_xlim(prop_dict['xlim'])
# # ax.set_ylim(prop_dict['ylim'])

# ax.grid(axis='both', 
#         which='major', 
#         alpha=1, 
#         linestyle='--', 
#         linewidth=0.6, 
#         color='gray')
    
# plt.show()
