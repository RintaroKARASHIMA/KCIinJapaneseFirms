import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator

# original module
import conv, pallet

# plt.rcParams['font.family'] = 'Meiryo'
# plt.rcParams['font.size'] = 15
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.bottom'] = False
# plt.rcParams['axes.spines.left'] = False
# plt.rcParams['axes.spines.right'] = False

# plt.rcParams['axes.grid'] = True
# plt.rcParams['axes.grid.axis'] = 'both'
# plt.rcParams['axes.grid.which'] = 'major'
# plt.rcParams['grid.alpha'] = 1.0
# plt.rcParams['grid.color'] = 'gray' # '#b0b0b0'
# plt.rcParams['grid.linestyle'] = '--'
# plt.rcParams['grid.linewidth'] = 0.8

global original_colors_list, lighter_colors_list, darker_colors_list, name_conv_dict, tech_colors_dict
original_colors_list = pallet.ORIGINAL_COLORS_LIST
lighter_colors_list = pallet.LIGHTER_COLORS_LIST
darker_colors_list = pallet.DARKER_COLORS_LIST
name_conv_dict = conv.NAME_CONV_DICT
tech_colors_dict = pallet.TECH_COLORS_DICT

def rank_doubleaxis(
                    df_dict: dict,
                    rank_num: int = 15,
                    member_col: str = 'right_person_name',
                    value_col: str = 'reg_num',
                    prop_dict: dict = {
                                        'figsize': (16, 10),
                                        'xlabel': 'Segment',
                                        'ylabel': 'y軸のラベル',
                                        'title': 'タイトル',
                                        'fontsize': 15,
                                        'year_range': 15,
                                        'ascending': False,
                                        'color': 'default',
                                        },
                    ):
    '''
    Args:
        df_dict: dict
            key: str, 区間名
            value: pd.DataFrame, データ
        rank_num: int, 上位何人を表示するか
        member_col: str, ランク付けする対象の列名
        value_col: str, ランク付けの基準となる列名
        prop_dict: dict
            figsize: tuple, 描画サイズ
            xlabel: str, 'Segment'
            ylabel: str, 'y軸のラベル'
            title: str, 'タイトル'
            fontsize: int, 15
            year_range: int, 15
            ascending: bool, False
            color: str, 'default'
    
    '''
    plt.rcParams['font.size'] = prop_dict['fontsize']
    plt.rcParams['font.family'] = 'Meiryo'
    
    rank_dfs_dict = df_dict.copy()
    segments_list = list(rank_dfs_dict.keys())

    for period, rank_df in rank_dfs_dict.items():
        rank_df = rank_df[[member_col, value_col]].copy()
        rank_df['rank'] = rank_df[value_col].rank(method='first', ascending=prop_dict['ascending']).astype(np.int64)
        rank_df['segment'] = period
        # try: rank_df['segment'] = int(period[:4])
        # except ValueError: rank_df['segment'] = period
        rank_df = rank_df.sort_values(by=['rank'], ascending=True)
        rank_dfs_dict[period] = rank_df[[member_col, 'rank', 'segment']].copy()
    rank_df = pd.concat(list(rank_dfs_dict.values()), ignore_index=True, axis='index')
    rank_df['segment'] = pd.Categorical(rank_df['segment'], categories=segments_list, ordered=True)
    
    # 左軸の上位
    first_top_sources = rank_df[(rank_df['segment'] == segments_list[0]) & (rank_df['rank'] <= rank_num)]
    # 右軸の上位
    last_top_sources = rank_df[(rank_df['segment'] == segments_list[-1]) & (rank_df['rank'] <= rank_num)]

    hr_list = rank_df[rank_df['rank'] <= rank_num][member_col].unique().tolist()

    color_list = (lighter_colors_list + original_colors_list + darker_colors_list) * (
        len(hr_list) // 60 + 1
    )

    hr_color_dict = {hr: 'gray' for hr in rank_df[member_col].unique().tolist()}
    for i, hr in enumerate(hr_list):
        hr_color_dict[hr] = color_list[i]
    if prop_dict['color'] == 'default':
        for i, hr in enumerate(hr_list):
            hr_color_dict[hr] = color_list[i]
    else:
        for k, v in prop_dict['color'].items():
            hr_color_dict[k] = v
    hr_color_dict = {**hr_color_dict, **tech_colors_dict}
    # hr_color_dict = dict(zip(hr_list, color_list[:len(hr_list)]))
    # キャンバスの生成
    fig, ax = plt.subplots(
        figsize=prop_dict['figsize'], subplot_kw=dict(ylim=(0.5, 0.5 + rank_num))
    )

    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(FixedLocator(first_top_sources['rank'].to_list()))
    
    # ax.yaxis.set_major_formatter(
    #     FixedFormatter(
    #         [name_conv_dict[name] for name in first_top_sources[member_col].to_list()]
    #     )
    # )
    ax.yaxis.set_major_formatter(
        FixedFormatter(
            first_top_sources[member_col].to_list()
        )
    )

    # 右側の軸
    yax2 = ax.secondary_yaxis('right')
    yax2.yaxis.set_major_locator(FixedLocator(last_top_sources['rank'].to_list()))
    # yax2.yaxis.set_major_formatter(
    #     FixedFormatter(
    #         [name_conv_dict[name] for name in last_top_sources[member_col].to_list()]
    #     )
    # )
    yax2.yaxis.set_major_formatter(
        FixedFormatter(
            last_top_sources[member_col].to_list()
        )
    )

    i = 0
    for member, rank in rank_df[rank_df['rank'] <= 10000].groupby(member_col):
        ax.plot(
            'segment',
            'rank',
            'o-',
            data=rank,
            linewidth=7,
            markersize=10,
            # color=hr_color_dict[member],
            color=hr_color_dict[member],
            alpha=0.6, 
            label=member
        )

    # 降順で描画しようね
    ax.invert_yaxis()

    # 軸ラベルとタイトル
    ax.set(
        xlabel='\n' + prop_dict['xlabel'],
        ylabel=prop_dict['ylabel'] + '\n',
        title=prop_dict['title'],
    )

    # 補助線
    ax.grid(axis='both', linestyle='--', c='lightgray')

    # 枠線消えちゃえ
    [s.set_visible(False) for s in ax.spines.values()]
    [s.set_visible(False) for s in yax2.spines.values()]

    # x軸の目盛り
    plt.xticks(
        range(
            0,
            len(segments_list),
            1,
            # prop_dict['year_range'],
        ),
        segments_list,
        rotation=90
    )

    # 収まるように描画しようね
    fig.tight_layout()

    return rank_df


# if __name__ == '__main__':
#     rank()
#     network()
