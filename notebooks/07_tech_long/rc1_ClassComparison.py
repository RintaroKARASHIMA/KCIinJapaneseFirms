#! (root)/notebooks/08_producer_sep/r1_Prefecture.py python3
# -*- coding: utf-8 -*-

# %%
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py
from scipy.stats import wilcoxon, mannwhitneyu
DATA_DIR = '../../data/processed/internal/05_2_7_tech_comparison/'

# %%
path_list = glob(DATA_DIR + '/*')
name_df = pd.read_csv(path_list[0],
                      encoding='utf-8',
                      sep=',')\
            .sort_values('schmoch35_tci', ascending=True)
addr_df = pd.read_csv(path_list[1],
                      encoding='utf-8',
                      sep=',')\
            .sort_values('schmoch35_tci', ascending=True)


tech_color = {
    'Chemistry, pharmaceuticals': 'tab:red',
    'Electrical engineering': 'tab:blue',
    'Instruments': 'tab:green',
    'Mechanical engineering, machinery': 'tab:orange',
    'Other fields': 'tab:gray'
}
#%%
name_df

# %%
## Scatter plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, 
                               figsize=(15, 15), 
                               sharey=True)

ax1.scatter(
            x='schmoch35_tci', y='schmoch35',
            data=name_df, 
            color='black', 
            marker='o',
            s=100,
            label='Schmoch（N=35）'
            )
ax1.scatter(
            x='ipc3_tci', y='schmoch35',
            data=name_df, 
            color='red', 
            alpha=1, 
            marker='+',
            s=200,
            label='IPC Class（N=124）'
            )
ax1.set_xlabel(
               'Corporate TCI', 
               fontsize=24, 
               fontweight='bold'
               )
ax1.set_xticks(
               range(0, 100+1, 25), 
               range(0, 100+1, 25)
               )
ax1.set_xticklabels(
                    [_ for _ in range(0, 100+1, 25)], 
                    rotation=90
                    )

ax1.grid(
         True, linestyle='--', which='major', axis='y'
         )


ax2.set_xlabel('Regional TCI', fontsize=24, fontweight='bold')
ax2.grid(True, linestyle='--', which='major', axis='y')
# ax2.legend(loc='upper left', fontsize=15, prop={'weight': 'bold'},bbox_to_anchor=(1.05, 0.5), borderaxespad=0)

ax2.scatter(
            x='schmoch35_tci', y='schmoch35', 
            data=addr_df,
            color='black', 
            marker='o',
            s=100,
            label='Schmoch（N=35）'
            )
ax2.scatter(
            x='ipc3_tci', y='schmoch35',
            data=addr_df,
            color='blue', 
            alpha=1, 
            marker='+',
            s=200,
            label='IPC Class（N=124）'
            )

# ax1.legend(loc='upper left', fontsize=15, prop={'weight': 'bold'},bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
# ax2.legend(
#            loc='upper left', 
#            fontsize=15, 
#            prop={
#                  'weight': 'bold'
#                  }, 
#            bbox_to_anchor=(-0.55, 0.5), 
#            borderaxespad=0
#            )

# ax1.text(1.1375, 0.75, 'Regional', fontsize=32, fontweight='bold', transform=ax2.transAxes)

ax2.set_yticklabels(
                    name_df['schmoch35'].drop_duplicates(), 
                    rotation=90
                    )
ax1.set_ylabel(
               'Schmoch', 
               fontsize=24, 
               fontweight='bold', 
            #    rotation=180
               )
ax2.set_xticks(
               range(0, 100+1, 25), 
               range(0, 100+1, 25)
               )
ax2.set_xticklabels(
                    [_ for _ in range(0, 100+1, 25)], 
                    rotation=90
                    )
ax1.xaxis.tick_top()#x軸を上側に
ax1.xaxis.set_label_position('top')#x軸ラベルを上側に
ax2.xaxis.tick_top()#x軸を上側に
ax2.xaxis.set_label_position('top')#x軸ラベルを上側に

lines = []
labels = []

for i, ax in enumerate(fig.axes):
    if i != 2:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)


fig.legend([_ for i, _ in enumerate(lines) if i != 2], 
           [_ for i, _ in enumerate(labels) if i != 2], 
           loc="upper right", 
           bbox_to_anchor=(1.275, 0.92), 
           borderaxespad=0,
           fontsize=21,
           prop={'weight': 'bold'}
           )

# ax2.text(1.125, 0.75, 'Corporate', fontsize=32, fontweight='bold', transform=ax2.transAxes)
# ax2.xaxis.set_major_locator(MultipleLocator(1))
# ax2.xaxis.set_major_locator(FixedLocator(name_df['schmoch35'].index.to_list()))

# ax.yaxis.set_major_formatter(
#     FixedFormatter(
#         [name_conv_dict[name] for name in first_top_sources[member_col].to_list()]
#     )
# )
# ax2.xaxis.set_major_formatter(
#     FixedFormatter(
#         name_df['schmoch35'].to_list()
#     )
# )

# ax.set_xscale('log')
# ax.legend(loc='center left', fontsize=20, bbox_to_anchor=(1.5, 0.5), borderaxespad=0, prop={'weight': 'bold'})
# ax.legend(loc='lower left', fontsize=20, prop={'weight': 'bold'})
# plt.tight_layout()
# fig.savefig(output_dir+'schmoch35_ipc3.png', dpi=400, bbox_inches='tight')
# plt.suptitle('TCI in Each Approach', fontsize=32, fontweight='bold')
plt.show()

# %%
## 
name_df['tci_abs'] = abs(name_df['schmoch35_tci'] - name_df['ipc3_tci'])
addr_df['tci_abs'] = abs(addr_df['schmoch35_tci'] - addr_df['ipc3_tci'])
name_df['schmoch35-ipc3'] = name_df['schmoch35'] + '-' + name_df['ipc3']
addr_df['schmoch35-ipc3'] = addr_df['schmoch35'] + '-' + addr_df['ipc3']
name_addr_df = pd.merge(name_df[['schmoch35-ipc3', 'tci_abs']].rename(columns={'tci_abs':'tci_abs_name'}), addr_df[['schmoch35-ipc3', 'tci_abs']].rename(columns={'tci_abs':'tci_abs_addr'}), on='schmoch35-ipc3', how='inner')
# ウィルコクソンの符号つき順位検定
statistic, p_value = wilcoxon(name_addr_df['tci_abs_name'], name_addr_df['tci_abs_addr'])
print(statistic, p_value)
# マンホイットニーのU検定
statistic, p_value = mannwhitneyu(name_addr_df['tci_abs_name'], name_addr_df['tci_abs_addr'], 
                                  alternative='two-sided')
print(statistic, p_value)
# display(name_addr_df)

#%%
name_addr_df

#%%
fig, ax = plt.subplots(figsize=(11, 6))
plot_df = pd.concat(
                    [
                    name_addr_df.filter(['tci_abs_name'])\
                                .rename(columns={'tci_abs_name':'tci_abs'})\
                                .assign(producer='Corporate'), 
                    name_addr_df.filter(['tci_abs_addr'])\
                                .rename(columns={'tci_abs_addr':'tci_abs'})\
                                .assign(producer='Regional'),
                    ], 
                    axis='index'
                    )
sns.violinplot(
                y='producer', x='tci_abs', 
                data=plot_df, 
                ax=ax,
                hue='producer',
                palette=['red', 'blue'],
                cut=0,
                inner_kws=dict(box_width=15, whis_width=2,
                                markerfacecoloralt='black', 
                                markeredgecolor='black', 
                                markerfacecolor='black', 
                                markersize=11, 
                                color='#000000'
                                )
                )
# sns.violinplot(
#                 x='producer', y='tci_abs', 
#                 data=plot_df, 
#                 ax=ax,
#                 hue='producer',
#                 palette=['red', 'blue'],
#                 cut=0,
#                 inner_kws=dict(box_width=15, whis_width=2,
#                                 markerfacecoloralt='black', 
#                                 markeredgecolor='black', 
#                                 markerfacecolor='black', 
#                                 markersize=11, 
#                                 color='#000000'
#                                 )
#                 )
ax.set_ylabel('Approach')
ax.set_xlabel('Absolute TCI Difference')
# ax.xaxis.tick_top()#x軸を上側に
# ax.xaxis.set_label_position('top')#x軸ラベルを上側に
ax.grid(True, linestyle='--', which='major', axis='x')


#%%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, 
                               figsize=(15, 15), 
                               sharey=True)

ax2.scatter(
            x='schmoch35', y='schmoch35_tci',
            data=name_df, 
            color='black', 
            marker='o',
            s=100,
            label='Schmoch（N=35）'
            )
ax2.scatter(
            x='schmoch35', y='ipc3_tci',
            data=name_df, 
            color='red', 
            alpha=1, 
            marker='+',
            s=200,
            label='IPC Class（N=124）'
            )
ax2.set_ylabel(
               'Corporate TCI', 
               fontsize=24, 
               fontweight='bold'
               )
ax1.set_yticks(
               range(0, 100+1, 25), 
               range(0, 100+1, 25)
               )
ax1.set_yticklabels(
                    [_ for _ in range(0, 100+1, 25)], 
                    rotation=90
                    )
ax1.set_xticklabels(
                    []
                )
ax1.grid(
         True, linestyle='--', which='major', axis='x'
         )


ax1.set_ylabel(
              'Regional TCI', fontsize=24, fontweight='bold'
              )
ax1.grid(
        True, linestyle='--', which='major', axis='x'
        )
# ax2.legend(loc='upper left', fontsize=15, prop={'weight': 'bold'},bbox_to_anchor=(1.05, 0.5), borderaxespad=0)

ax1.scatter(
            x='schmoch35', y='schmoch35_tci', 
            data=addr_df,
            color='black', 
            marker='o',
            s=100,
            label='Schmoch（N=35）'
            )
ax1.scatter(
            x='schmoch35', y='ipc3_tci',
            data=addr_df,
            color='blue', 
            alpha=1, 
            marker='+',
            s=200,
            label='IPC Class（N=124）'
            )

# ax1.legend(loc='upper left', fontsize=15, prop={'weight': 'bold'},bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
# ax2.legend(
#            loc='upper left', 
#            fontsize=15, 
#            prop={
#                  'weight': 'bold'
#                  }, 
#            bbox_to_anchor=(-0.55, 0.5), 
#            borderaxespad=0
#            )

# ax1.text(1.1375, 0.75, 'Regional', fontsize=32, fontweight='bold', transform=ax2.transAxes)

ax2.set_xticklabels(
                    name_df['schmoch35'].drop_duplicates(), 
                    rotation=90
                    )
ax2.set_xlabel(
               'Schmoch', 
               fontsize=24, 
               fontweight='bold', 
            #    rotation=180
               )
ax2.set_yticks(
               range(0, 100+1, 25), 
               range(0, 100+1, 25)
               )
ax2.set_yticklabels(
                    [_ for _ in range(0, 100+1, 25)], 
                    rotation=90
                    )
# ax1.xaxis.tick_top()#x軸を上側に
# ax1.xaxis.set_label_position('top')#x軸ラベルを上側に
# ax2.xaxis.tick_top()#x軸を上側に
# ax2.xaxis.set_label_position('top')#x軸ラベルを上側に

lines = []
labels = []

for i, ax in enumerate(fig.axes):
    if i != 2:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)


fig.legend(
           [_ for i, _ in enumerate(lines) if i != 2], 
           [_ for i, _ in enumerate(labels) if i != 2], 
           loc="upper right", 
           bbox_to_anchor=(1.35, 0.2), 
           borderaxespad=0,
           prop={'weight': 'bold', 'size': 27}
           )

#%%

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=True)
plot_df = pd.concat(
                    [
                    name_addr_df.filter(['tci_abs_name'])\
                                .rename(columns={'tci_abs_name':'tci_abs'})\
                                .assign(producer='Corporate'), 
                    name_addr_df.filter(['tci_abs_addr'])\
                                .rename(columns={'tci_abs_addr':'tci_abs'})\
                                .assign(producer='Regional'),
                    ], 
                    axis='index'
                    )
sns.violinplot(
                y='producer', x='tci_abs', 
                data=plot_df[plot_df['producer'] == 'Corporate'], 
                ax=ax1,
                hue='producer',
                palette=['red', 'blue'],
                cut=0,
                inner_kws=dict(box_width=15, whis_width=2,
                                markerfacecoloralt='black', 
                                markeredgecolor='black', 
                                markerfacecolor='black', 
                                markersize=11, 
                                color='#000000'
                                )
                )
sns.violinplot(
                y='producer', x='tci_abs', 
                data=plot_df[plot_df['producer'] == 'Regional'], 
                ax=ax2,
                hue='producer',
                palette=['blue'],
                cut=0,
                inner_kws=dict(box_width=15, whis_width=2,
                                markerfacecoloralt='black', 
                                markeredgecolor='black', 
                                markerfacecolor='black', 
                                markersize=11, 
                                color='#000000'
                                )
                )
# sns.violinplot(
#                 x='producer', y='tci_abs', 
#                 data=plot_df, 
#                 ax=ax,
#                 hue='producer',
#                 palette=['red', 'blue'],
#                 cut=0,
#                 inner_kws=dict(box_width=15, whis_width=2,
#                                 markerfacecoloralt='black', 
#                                 markeredgecolor='black', 
#                                 markerfacecolor='black', 
#                                 markersize=11, 
#                                 color='#000000'
#                                 )
#                 )
ax1.set_ylabel('')
ax2.set_ylabel('')
ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax2.set_xlabel('Absolute TCI Difference')
ax1.grid(True, linestyle='--', which='major', axis='x')
ax2.grid(True, linestyle='--', which='major', axis='x')
# ax.xaxis.tick_top()#x軸を上側に
# ax.xaxis.set_label_position('top')#x軸ラベルを上側に


# %%
addr_df[addr_df['schmoch35_tci']>=75]['schmoch35'].nunique()
name_df[name_df['schmoch35_tci']>=75]['schmoch35'].nunique()
# %%
