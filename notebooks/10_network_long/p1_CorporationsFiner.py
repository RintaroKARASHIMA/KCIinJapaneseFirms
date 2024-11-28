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

#%%
all_c_df
def ccdf(diversity_col: list):
        freq_array = np.array(np.bincount(diversity_col))
        p_list = []
        cumsum = 0.0
        s = float(freq_array.sum())
        for freq in freq_array:
            if freq != 0:
                cumsum += freq / s
                p_list.append(cumsum)
            else:
                p_list.append(1.0)
                
        ccdf_array = 1 - np.array(p_list)
        if ccdf_array[0] == 0:
            ccdf_array[0] = 1.0
        return ccdf_array

color_list = [
    'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
    'grey', 'violet', 'indigo', 'turquoise', 'gold', 'lime', 'coral',
    'navy', 'skyblue', 'tomato', 'olive', 'cyan', 'darkred', 'darkgreen',
    'darkblue', 'darkorange', 'darkviolet', 'deeppink', 'firebrick', 'darkcyan',
    'darkturquoise', 'darkslategray', 'darkgoldenrod', 'mediumblue', 'mediumseagreen',
    'mediumpurple', 'mediumvioletred', 'midnightblue', 'saddlebrown', 'seagreen',
    'sienna', 'steelblue'
    ][10:]

color_count = 0
fig, ax = plt.subplots(figsize=(10, 10))
for s in all_c_df['segment'].unique():
    ccdf_array = ccdf(all_c_df[all_c_df['segment']==s]['diversity'].to_list())
    ax.plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=8, 
                    color='red', label=s, alpha=0.6)
    color_count += 1
ax.legend(loc='upper right', fontsize=18)

ax.set_title('特許権者の補累積次数（Diversity）分布（両対数スケール）'+'\n', fontsize=20)
ax.set_xlabel('特許権者次数（Diversity）', fontsize=18)
ax.set_ylabel('ccdf', fontsize=18)

ax.set_xscale('log')
ax.set_yscale('log')

# x軸の指数表記を普通に戻す魔法
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# ax.set_xlim(prop_dict['xlim'])
# ax.set_ylim(prop_dict['ylim'])

ax.grid(axis='both', 
        which='major', 
        alpha=1, 
        linestyle='--', 
        linewidth=0.8, 
        color='gray')
    
plt.show()

#%%
color_list = ['red']+[
    'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
    'grey', 'violet', 'indigo', 'turquoise', 'gold', 'lime', 'coral',
    'navy', 'skyblue', 'tomato', 'olive', 'cyan', 'darkred', 'darkgreen',
    'darkblue', 'darkorange', 'darkviolet', 'deeppink', 'firebrick', 'darkcyan',
    'darkturquoise', 'darkslategray', 'darkgoldenrod', 'mediumblue', 'mediumseagreen',
    'mediumpurple', 'mediumvioletred', 'midnightblue', 'saddlebrown', 'seagreen',
    'sienna', 'steelblue'
    ][10:]
color_count = 0
fig, ax = plt.subplots(figsize=(10, 10))
for s in list(right_person_df['segment'].unique())[0:]:
    
    x = right_person_df[right_person_df['segment']==s][['reg_num']].rank(ascending=False, method='first').sort_values('reg_num', ascending=True)['reg_num']
    # y = 1 - np.cumsum(right_person_df[right_person_df['segment']==s][['reg_num']].sort_values('reg_num',ascending=False)['reg_num'] / right_person_df[right_person_df['segment']==s]['reg_num'].sum())
    y = np.cumsum(right_person_df[right_person_df['segment']==s][['reg_num']].sort_values('reg_num',ascending=False)['reg_num'] / right_person_df[right_person_df['segment']==s]['reg_num'].sum())
    # y = [1] + list(y)[:-1]
    y = list(y)[:-1] + [1]
    # ccdf_array = ccdf()
    # ax.plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=8, 
    #                 color=color_list[color_count], label=s+'年度', alpha=0.6)
    ax.plot(x, y, 'o', markersize=8, 
                    color=color_list[color_count], label=s+'年度', alpha=0.6)
    ax.axvline(len(x)*3//100, color=color_list[color_count], linestyle='--')
    color_count += 1
# ax.legend(loc='lower left', fontsize=18)
ax.legend(loc='upper left', fontsize=18)

ax.set_title('各期間における特許権者の累積特許数分布（両対数スケール）'+'\n', fontsize=20)
ax.set_xlabel('特許数', fontsize=18)
ax.set_ylabel('ccdf', fontsize=18)

ax.set_xscale('log')
# ax.set_yscale('log')

ax.tick_params(labelsize=18)
# ax.set_xlim(0.8, 300)

# x軸の指数表記を普通に戻す魔法
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))

# ax.set_xlim(prop_dict['xlim'])
# ax.set_ylim(prop_dict['ylim'])

ax.grid(axis='both', 
        which='major', 
        alpha=1, 
        linestyle='--', 
        linewidth=0.6, 
        color='gray')
    
plt.show()
