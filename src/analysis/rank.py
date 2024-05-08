import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator

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
    original_colors_list = [
        '#ff0000',
        '#ff5000',
        '#ffa100',
        '#fff100',
        '#bbff00',
        '#6bff00',
        '#1aff00',
        '#00ff35',
        '#00ff86',
        '#00ffd6',
        '#00d6ff',
        '#0086ff',
        '#0035ff',
        '#1a00ff',
        '#6b00ff',
        '#bb00ff',
        '#ff00f1',
        '#ff00a1',
        '#ff0050',
        '#ff0000',
    ]
    lighter_colors_list = [
        '#ff0000',
        '#ff5000',
        '#ffa100',
        '#fff100',
        '#bbff00',
        '#6bff00',
        '#1aff00',
        '#00ff35',
        '#00ff85',
        '#00ffd6',
        '#00d5ff',
        '#0085ff',
        '#0035ff',
        '#1900ff',
        '#6b00ff',
        '#bb00ff',
        '#ff00f1',
        '#ff00a0',
        '#ff0050',
        '#ff0000',
    ]
    darker_colors_list = [
        '#7f0000',
        '#7f2800',
        '#7f5000',
        '#7f7800',
        '#5d7f00',
        '#357f00',
        '#0d7f00',
        '#007f1a',
        '#007f42',
        '#007f6b',
        '#006a7f',
        '#00427f',
        '#001a7f',
        '#0c007f',
        '#35007f',
        '#5d007f',
        '#7f0078',
        '#7f0050',
        '#7f0028',
        '#7f0000',
    ]
    # 左側の軸
    name_conv_dict = {
        'ＪＦＥエンジニアリング株式会社': 'JFEエンジニアリング株式会社',
        'ＭｅｉｊｉＳｅｉｋａフアルマ株式会社': 'MeijiSeikaファルマ株式会社',
        'キツコーマン株式会社': 'キッコーマン株式会社',
        'キツセイ薬品工業株式会社': 'キッセイ薬品株式会社',
        'キヤノン株式会社': 'キヤノン株式会社',
        'コニカミノルタ株式会社': 'コニカミノルタ株式会社',
        'シヤープ株式会社': 'シャープ株式会社',
        'セイコーエプソン株式会社': 'セイコーエプソン株式会社',
        'ソニーグループ株式会社': 'ソニーグループ株式会社',
        'ソニー株式会社': 'ソニー株式会社',
        'ダイスタージヤパン株式会社': 'ダイスタージャパン株式会社',
        'トヨタ自動車株式会社': 'トヨタ自動車株式会社',
        'パナソニツクホールデイングス株式会社': 'パナソニックホールディングス株式会社',
        'パナソニツク株式会社': 'パナソニック株式会社',
        'メルシヤン株式会社': 'メルシャン株式会社',
        'ルネサスエレクトロニクス株式会社': 'ルネサンスエレクトロニクス株式会社',
        '旭化成株式会社': '旭化成株式会社',
        '一丸フアルコス株式会社': '一丸ファルコス株式会社',
        '株式会社ＩＨＩ': '株式会社IHI',
        '株式会社デンソー': '株式会社デンソー',
        '株式会社ノエビア': '株式会社ノエビア',
        '株式会社フアンケル': '株式会社ファンケル',
        '株式会社リコー': '株式会社リコー',
        '株式会社三井Ｅ＆Ｓホールデイングス': '株式会社三井E&Sホールディングス',
        '株式会社神戸製鋼所': '株式会社神戸製鋼所',
        '株式会社東芝': '株式会社東芝',
        '株式会社日立製作所': '株式会社日立製作所',
        '株式会社豊田中央研究所': '株式会社豊田中央研究所',
        '株式会社明治': '株式会社明治',
        '株式会社林原': '株式会社林原',
        '丸善製薬株式会社': '丸善製薬株式会社',
        '協和キリン株式会社': '協和キリン株式会社',
        '高砂香料工業株式会社': '高砂香料工業株式会社',
        '国立研究開発法人産業技術総合研究所': '国立研究開発法人産業技術総合研究所',
        '国立大学法人東京工業大学': '国立大学法人東京工業大学',
        '国立大学法人東京大学': '国立大学法人東京大学',
        '財団法人微生物化学研究会': '財団法人微生物化学研究会',
        '三井化学アグロ株式会社': '三井化学アグロ株式会社',
        '三栄源エフ・エフ・アイ株式会社': '三栄源エフ・エフ・アイ株式会社',
        '三共株式会社': '三井株式会社',
        '三省製薬株式会社': '三省製薬株式会社',
        '三菱ケミカル株式会社': '三菱ケミカル株式会社',
        '三菱重工業株式会社': '三菱重工業株式会社',
        '三菱商事ライフサイエンス株式会社': '三菱商事ライフサイエンス株式会社',
        '三菱電機株式会社': '三菱電機株式会社',
        '三洋電機株式会社': '三洋電機株式会社',
        '住友フアーマ株式会社': '住友ファーマ株式会社',
        '住友重機械工業株式会社': '住友重機械工業株式会社',
        '小川香料株式会社': '小川香料株式会社',
        '松谷化学工業株式会社': '松谷化学工業株式会社',
        '森永乳業株式会社': '森永乳業株式会社',
        '雪印メグミルク株式会社': '雪印メグミルク株式会社',
        '川崎重工業株式会社': '川崎重工業株式会社',
        '太陽化学株式会社': '太陽化学株式会社',
        '大阪瓦斯株式会社': '大阪瓦斯株式会社',
        '大塚製薬株式会社': '大塚製薬株式会社',
        '大鵬薬品工業株式会社': '大鵬薬品工業株式会社',
        '中外製薬株式会社': '中外製薬株式会社',
        '中国電力株式会社': '中国電力株式会社',
        '長谷川香料株式会社': '長谷川香料株式会社',
        '天野エンザイム株式会社': '天野エンザイム株式会社',
        '田辺三菱製薬株式会社': '田辺三菱製薬株式会社',
        '独立行政法人産業技術総合研究所': '独立行政法人産業技術総合研究所',
        '日産自動車株式会社': '日立自動車株式会社',
        '日清オイリオグループ株式会社': '日清オイリオグループ株式会社',
        '日本ケミフア株式会社': '日本ケミファ株式会社',
        '日本新薬株式会社': '日本新薬株式会社',
        '日本水産株式会社': '日本水産株式会社',
        '日本製鉄株式会社': '日本製鉄株式会社',
        '日本電気株式会社': '日本電気株式会社',
        '日本電信電話株式会社': '日本電信電話株式会社',
        '日立造船株式会社': '日立造船株式会社',
        '不二製油グループ本社株式会社': '不二製油グループ本社株式会社',
        '不二製油株式会社': '不二製油株式会社',
        '富士通株式会社': '富士通株式会社',
        '富士電機株式会社': '富士電機株式会社',
        '本田技研工業株式会社': '本田技研工業株式会社',
        '味の素株式会社': '味の素株式会社',
        '理研ビタミン株式会社': '理研ビタミン株式会社'
    }

    tech_colors_dict = {
        'Digital communication': ['Electrical engineering', '#6bff00'],
        'Telecommunications': ['Electrical engineering', '#6bff00'],
        'Computer technology': ['Electrical engineering', '#6bff00'],
        'Audio-visual technology': ['Electrical engineering', '#6bff00'],
        'IT methods for management': ['Electrical engineering', '#6bff00'],
        'Pharmaceuticals': ['Chemistry', '#ff0000'],
        'Organic fine chemistry': ['Chemistry', '#ff0000'],
        'Basic communication processes': ['Electrical engineering', '#6bff00'],
        'Optics': ['Instruments', '#00ffd6'],
        'Semiconductors': ['Electrical engineering', '#6bff00'],
        'Biotechnology': ['Instruments', '#00ffd6'],
        'Medical technology': ['Electrical engineering', '#6bff00'],
        'Micro-structural and nano-technology': ['Chemistry', '#ff0000'],
        'Measurement': ['Instruments', '#00ffd6'],
        'Food chemistry': ['Chemistry', '#ff0000'],
        'Control': ['Instruments', '#00ffd6'],
        'Furniture, games': ['Other fields', '#ff0050'],
        'Basic materials chemistry': ['Chemistry', '#ff0000'],
        'Chemical engineering': ['Chemistry', '#ff0000'],
        'Environmental technology': ['Chemistry', '#ff0000'],
        'Macromolecular chemistry, polymers': ['Chemistry', '#ff0000'],
        'Engines, pumps, turbines': ['Mechanical engineering', '#bb00ff'],
        'Electrical machinery, apparatus, energy': ['Electrical engineering',
        '#6bff00'],
        'Textile and paper machines': ['Mechanical engineering', '#bb00ff'],
        'Other consumer goods': ['Other fields', '#ff0050'],
        'Civil engineering': ['Other fields', '#ff0050'],
        'Materials, metallurgy': ['Chemistry', '#ff0000'],
        'Other special machines': ['Mechanical engineering', '#bb00ff'],
        'Thermal processes and apparatus': ['Mechanical engineering', '#bb00ff'],
        'Surface technology, coating': ['Chemistry', '#ff0000'],
        'Transport': ['Mechanical engineering', '#bb00ff'],
        'Handling': ['Mechanical engineering', '#bb00ff'],
        'Mechanical elements': ['Mechanical engineering', '#bb00ff']
        }

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
            len(segments_list) + 1,
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
