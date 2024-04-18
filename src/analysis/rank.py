import networkx as nx
from networkx.algorithms import bipartite as bip
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


def rank_doubleaxis(
    df_dict: dict,
    version: str = "reg_num",
    rank_num: int = 15,
    member_col: str = "right_person_name",
    num_col: str = "reg_num",
    prop_dict: dict = {
        "figsize": (16, 10),
        "xlabel": "Segment",
        "ylabel": "y軸のラベル",
        "title": "タイトル",
        "fontsize": 15,
        "year_range": 15,
        "ascending": False,
        "color": "default",
    },
):
    plt.rcParams["font.size"] = prop_dict["fontsize"]
    plt.rcParams["font.family"] = "Meiryo"
    rank_df_dict = df_dict.copy()

    for period, df in rank_df_dict.items():
        if version == "reg_num":
            rank_df = df.copy()
            rank_df = (
                rank_df.groupby(member_col)[[num_col]].nunique().reset_index(drop=False)
            )

        elif version == "kci":
            rank_df = df[[member_col, num_col]].copy()

        rank_df["rank"] = (
            rank_df[num_col]
            .rank(method="first", ascending=prop_dict["ascending"])
            .astype(np.int64)
        )
        rank_df["period"] = period
        rank_df["segment"] = int(period[:4])
        rank_df = rank_df.sort_values(["segment", "rank"], ascending=True)
        rank_df_dict[period] = rank_df[[member_col, "rank", "segment", "period"]].copy()

    rank_df = pd.concat(list(rank_df_dict.values()), ignore_index=True, axis="index")

    # 左軸の上位
    first_top_sources = rank_df[
        (rank_df["segment"] == rank_df["segment"].min()) & (rank_df["rank"] <= rank_num)
    ]

    # 右軸の上位
    last_top_sources = rank_df[
        (rank_df["segment"] == rank_df["segment"].max()) & (rank_df["rank"] <= rank_num)
    ]

    hr_list = rank_df[rank_df["rank"] <= rank_num][member_col].unique().tolist()
    original_colors = [
        "#ff0000",
        "#ff5000",
        "#ffa100",
        "#fff100",
        "#bbff00",
        "#6bff00",
        "#1aff00",
        "#00ff35",
        "#00ff86",
        "#00ffd6",
        "#00d6ff",
        "#0086ff",
        "#0035ff",
        "#1a00ff",
        "#6b00ff",
        "#bb00ff",
        "#ff00f1",
        "#ff00a1",
        "#ff0050",
        "#ff0000",
    ]
    lighter_colors = [
        "#ff0000",
        "#ff5000",
        "#ffa100",
        "#fff100",
        "#bbff00",
        "#6bff00",
        "#1aff00",
        "#00ff35",
        "#00ff85",
        "#00ffd6",
        "#00d5ff",
        "#0085ff",
        "#0035ff",
        "#1900ff",
        "#6b00ff",
        "#bb00ff",
        "#ff00f1",
        "#ff00a0",
        "#ff0050",
        "#ff0000",
    ]
    darker_colors = [
        "#7f0000",
        "#7f2800",
        "#7f5000",
        "#7f7800",
        "#5d7f00",
        "#357f00",
        "#0d7f00",
        "#007f1a",
        "#007f42",
        "#007f6b",
        "#006a7f",
        "#00427f",
        "#001a7f",
        "#0c007f",
        "#35007f",
        "#5d007f",
        "#7f0078",
        "#7f0050",
        "#7f0028",
        "#7f0000",
    ]

    color_list = (lighter_colors + original_colors + darker_colors) * (
        len(hr_list) // 60 + 1
    )

    hr_color_dict = {hr: "gray" for hr in rank_df[member_col].unique().tolist()}
    for i, hr in enumerate(hr_list):
        hr_color_dict[hr] = color_list[i]
    if prop_dict["color"] == "default":
        for i, hr in enumerate(hr_list):
            hr_color_dict[hr] = color_list[i]
    else:
        for k, v in prop_dict["color"].items():
            hr_color_dict[k] = v
    # hr_color_dict = dict(zip(hr_list, color_list[:len(hr_list)]))
    # キャンバスの生成
    fig, ax = plt.subplots(
        figsize=prop_dict["figsize"], subplot_kw=dict(ylim=(0.5, 0.5 + rank_num))
    )

    # 左側の軸
    name_conv_dict = {
        "ＪＦＥエンジニアリング株式会社": "JFEエンジニアリング株式会社",
        "ＭｅｉｊｉＳｅｉｋａフアルマ株式会社": "MeijiSeikaファルマ株式会社",
        "キツコーマン株式会社": "キッコーマン株式会社",
        "キツセイ薬品工業株式会社": "キッセイ薬品株式会社",
        "キヤノン株式会社": "キヤノン株式会社",
        "コニカミノルタ株式会社": "コニカミノルタ株式会社",
        "シヤープ株式会社": "シャープ株式会社",
        "セイコーエプソン株式会社": "セイコーエプソン株式会社",
        "ソニーグループ株式会社": "ソニーグループ株式会社",
        "ソニー株式会社": "ソニー株式会社",
        "ダイスタージヤパン株式会社": "ダイスタージャパン株式会社",
        "トヨタ自動車株式会社": "トヨタ自動車株式会社",
        "パナソニツクホールデイングス株式会社": "パナソニックホールディングス株式会社",
        "パナソニツク株式会社": "パナソニック株式会社",
        "メルシヤン株式会社": "メルシャン株式会社",
        "ルネサスエレクトロニクス株式会社": "ルネサンスエレクトロニクス株式会社",
        "旭化成株式会社": "旭化成株式会社",
        "一丸フアルコス株式会社": "一丸ファルコス株式会社",
        "株式会社ＩＨＩ": "株式会社IHI",
        "株式会社デンソー": "株式会社デンソー",
        "株式会社ノエビア": "株式会社ノエビア",
        "株式会社フアンケル": "株式会社ファンケル",
        "株式会社リコー": "株式会社リコー",
        "株式会社三井Ｅ＆Ｓホールデイングス": "株式会社三井E&Sホールディングス",
        "株式会社神戸製鋼所": "株式会社神戸製鋼所",
        "株式会社東芝": "株式会社東芝",
        "株式会社日立製作所": "株式会社日立製作所",
        "株式会社豊田中央研究所": "株式会社豊田中央研究所",
        "株式会社明治": "株式会社明治",
        "株式会社林原": "株式会社林原",
        "丸善製薬株式会社": "丸善製薬株式会社",
        "協和キリン株式会社": "協和キリン株式会社",
        "高砂香料工業株式会社": "高砂香料工業株式会社",
        "国立研究開発法人産業技術総合研究所": "国立研究開発法人産業技術総合研究所",
        "国立大学法人東京工業大学": "国立大学法人東京工業大学",
        "国立大学法人東京大学": "国立大学法人東京大学",
        "財団法人微生物化学研究会": "財団法人微生物化学研究会",
        "三井化学アグロ株式会社": "三井化学アグロ株式会社",
        "三栄源エフ・エフ・アイ株式会社": "三栄源エフ・エフ・アイ株式会社",
        "三共株式会社": "三井株式会社",
        "三省製薬株式会社": "三省製薬株式会社",
        "三菱ケミカル株式会社": "三菱ケミカル株式会社",
        "三菱重工業株式会社": "三菱重工業株式会社",
        "三菱商事ライフサイエンス株式会社": "三菱商事ライフサイエンス株式会社",
        "三菱電機株式会社": "三菱電機株式会社",
        "三洋電機株式会社": "三洋電機株式会社",
        "住友フアーマ株式会社": "住友ファーマ株式会社",
        "住友重機械工業株式会社": "住友重機械工業株式会社",
        "小川香料株式会社": "小川香料株式会社",
        "松谷化学工業株式会社": "松谷化学工業株式会社",
        "森永乳業株式会社": "森永乳業株式会社",
        "雪印メグミルク株式会社": "雪印メグミルク株式会社",
        "川崎重工業株式会社": "川崎重工業株式会社",
        "太陽化学株式会社": "太陽化学株式会社",
        "大阪瓦斯株式会社": "大阪瓦斯株式会社",
        "大塚製薬株式会社": "大塚製薬株式会社",
        "大鵬薬品工業株式会社": "大鵬薬品工業株式会社",
        "中外製薬株式会社": "中外製薬株式会社",
        "中国電力株式会社": "中国電力株式会社",
        "長谷川香料株式会社": "長谷川香料株式会社",
        "天野エンザイム株式会社": "天野エンザイム株式会社",
        "田辺三菱製薬株式会社": "田辺三菱製薬株式会社",
        "独立行政法人産業技術総合研究所": "独立行政法人産業技術総合研究所",
        "日産自動車株式会社": "日立自動車株式会社",
        "日清オイリオグループ株式会社": "日清オイリオグループ株式会社",
        "日本ケミフア株式会社": "日本ケミファ株式会社",
        "日本新薬株式会社": "日本新薬株式会社",
        "日本水産株式会社": "日本水産株式会社",
        "日本製鉄株式会社": "日本製鉄株式会社",
        "日本電気株式会社": "日本電気株式会社",
        "日本電信電話株式会社": "日本電信電話株式会社",
        "日立造船株式会社": "日立造船株式会社",
        "不二製油グループ本社株式会社": "不二製油グループ本社株式会社",
        "不二製油株式会社": "不二製油株式会社",
        "富士通株式会社": "富士通株式会社",
        "富士電機株式会社": "富士電機株式会社",
        "本田技研工業株式会社": "本田技研工業株式会社",
        "味の素株式会社": "味の素株式会社",
        "理研ビタミン株式会社": "理研ビタミン株式会社"
    }
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(FixedLocator(first_top_sources["rank"].to_list()))
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
    yax2 = ax.secondary_yaxis("right")
    yax2.yaxis.set_major_locator(FixedLocator(last_top_sources["rank"].to_list()))
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
    for member, rank in rank_df[rank_df["rank"] <= 10000].groupby(member_col):
        ax.plot(
            "segment",
            "rank",
            "o-",
            data=rank,
            linewidth=5,
            markersize=10,
            color=hr_color_dict[member],
            alpha=0.6,
        )

    # 降順で描画しようね
    ax.invert_yaxis()

    # 軸ラベルとタイトル
    ax.set(
        xlabel="\n" + prop_dict["xlabel"],
        ylabel=prop_dict["ylabel"] + "\n",
        title=prop_dict["title"],
    )

    # 補助線
    ax.grid(axis="both", linestyle="--", c="lightgray")

    # 枠線消えちゃえ
    [s.set_visible(False) for s in ax.spines.values()]
    [s.set_visible(False) for s in yax2.spines.values()]

    # x軸の目盛り
    plt.xticks(
        range(
            rank_df["segment"].min(),
            rank_df["segment"].max() + 1,
            prop_dict["year_range"],
        ),
        list(rank_df["period"].unique()),
        rotation=90,
    )

    # 収まるように描画しようね
    fig.tight_layout()

    return rank_df


# if __name__ == '__main__':
#     rank()
#     network()
