#%%
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Yu Mincho'
plt.rcParams['font.size'] = 15

sys.path.append(str(Path(__file__).resolve().parents[2]) / 'src')
from calculation.biadjm import compute_pref_schmoch_lq
from calculation.fc_algorithm import Fitn_Comp

# Computation of RCA
def rca(biadjm):
    RCA = np.zeros_like(biadjm, dtype=float)
    rows_degrees = biadjm.sum(axis=1)
    cols_degrees = biadjm.sum(axis=0)
    tot_degrees = biadjm.sum()
    for i in range(np.shape(biadjm)[0]):
        if rows_degrees[i] != 0:
            for j in range(np.shape(biadjm)[1]):
                if cols_degrees[j] != 0:
                    RCA[i,j] = (biadjm[i, j] / rows_degrees[i]) / (cols_degrees[j] / tot_degrees)
    return RCA


#%%
raw_df = pd.read_csv(
    'data/interim/internal/filtered_before_agg/japan.csv',
    encoding='utf-8',
    sep=',',
    dtype={
        'ipc': object,
        'reg_num': object,
        'app_year': np.int64,
        'app_nendo': np.int64,
        'reg_year': np.int64,
        'reg_nendo': np.int64,
        'right_person_addr': str,
        'schmoch35': np.int64,
    }
    )\
    .assign(
        ipc3 = lambda x: x['ipc'].str[:3],
        ipc4 = lambda x: x['ipc'].str[:4],
    )\
    .drop(
        columns=['ipc','right_person_name']
    )\
    .drop_duplicates(
        keep='first'
    )
display(raw_df)


# %%
df = raw_df.query('1981 <= app_nendo <= 2015', engine='python')\
           .drop(columns=['app_year', 'reg_year', 'reg_nendo'])\
           .drop_duplicates(keep='first')\
           .reset_index(drop=True)

#%%
df['reg_num'].nunique()
# %%
long_df = (
    df.drop(
        columns=['ipc3', 'ipc4']
        )\
        .drop_duplicates(keep='first')\
        .assign(
            app_nendo_period = '2015-2015'
        )\
        .drop(columns=['app_nendo'])\
        .assign(
            addr_count=lambda x: x.groupby('reg_num')['right_person_addr'].transform('nunique'),
            class_count=lambda x: x.groupby('reg_num')['schmoch35'].transform('nunique')
        )\
        .assign(
            weight=lambda x: 1 / (x['addr_count'] * x['class_count'])
        )\
        .groupby(['app_nendo_period', 'right_person_addr', 'schmoch35'], as_index=False)\
        .agg(
            patent_count=('weight', 'sum')
        )\
        .drop(columns=['app_nendo_period'])\
        .rename(columns={'right_person_addr':'prefecture'})
)
display(long_df)


#%%
schmoch35_df = pd.read_csv(
    'data/processed/external/schmoch/35.csv',
    encoding='utf-8',
    sep=',',
    ).filter(items=['Field_number', 'Field_en'])\
    .drop_duplicates(
        keep='first'
    )
 
display(schmoch35_df)



long_mcp = compute_pref_schmoch_lq(long_df)\
    .assign(
        app_nendo_period = '1975-2015'
    )
long_mcp


#%%
# 1) 行=地域, 列=技術 の行列にピボット
biadjm_df = long_df.pivot_table(
    index="prefecture",
    columns="schmoch35",
    values="patent_count",
    aggfunc="sum",
    fill_value=0
)

# 2) 0のみの行/列を削除（READMEが「重要」と明記）
biadjm_df = biadjm_df.loc[biadjm_df.sum(axis=1) > 0, :]
biadjm_df = biadjm_df.loc[:, biadjm_df.sum(axis=0) > 0]

# 3) NumPy配列へ
biadjm_counts = biadjm_df.values  # 重み付き（特許数）

# 4) presence行列（RCA>=1 を 1、それ以外 0）を作成（Fitn_Compに推奨）
R = rca(biadjm_counts)
# biadjm_presence = (R >= 1.0).astype(int)
biadjm_presence = long_mcp.pivot_table(
    index="prefecture",
    columns="schmoch35",
    values="mpc",
    aggfunc="sum",
    fill_value=0
)

#%%
# （オプション）presence が全0になった行/列を再度落とす
# presence_df = pd.DataFrame(biadjm_presence, index=biadjm_df.index, columns=biadjm_df.columns)
# presence_df = presence_df.loc[presence_df.sum(axis=1) > 0, :]
# presence_df = presence_df.loc[:, presence_df.sum(axis=0) > 0]
# biadjm_presence = presence_df.values

# 5) Fitness & Complexity（presenceを入力）
fitness, complexity = Fitn_Comp(biadjm_presence.values)
fitness
prefectures = biadjm_presence.index
result_df = pd.DataFrame(
    {
        'prefecture': prefectures,
        'fitness': fitness
    }
).sort_values(by='fitness', ascending=False, ignore_index=True)



# %%
result_df.assign(
    rank = lambda x: x['fitness'].rank(method='min', ascending=False).astype(np.int64)
).sort_values(by='rank', ascending=True, ignore_index=True)

#%%
long_mcp.merge(
    schmoch35_df,
    left_on='schmoch35',
    right_on='Field_number',
    how='left'
    )\
    .pivot_table(
    index="prefecture",
    columns="Field_en",
    values="mpc",
    aggfunc="sum",
    fill_value=0
).to_clipboard()
#%%
long_mcp.merge(
    result_df,
    on='prefecture',
    how='left'
).assign(
    diversity = lambda x: x.groupby('prefecture')['mpc'].transform('sum')
)\
.filter(items=['prefecture', 'fitness', 'diversity'])\
.drop_duplicates(keep='first', ignore_index=True).to_clipboard()
#%%
from japanmap import pref_names, picture
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# ---- 都道府県名→コード ----
name_to_code = {v: k for k, v in enumerate(pref_names)}
result_df["code"] = result_df["prefecture"].map(name_to_code)

# ---- fitness の正規化（着色用）----
# fmin, fmax = result_df["fitness"].min(), result_df["fitness"].max()
fmin, fmax = 0, 1
result_df["fitness_norm"] = (result_df["fitness"] - fmin) / (fmax - fmin)

# ---- 白→黒 のカラーマップ ----
cmap_bw = mcolors.LinearSegmentedColormap.from_list("white_to_black", ["white", "black"])

# ---- 正規化値→RGB(0-255) ----
result_df["color"] = result_df["fitness_norm"].apply(
    lambda v: tuple(int(255*x) for x in cmap_bw(v)[:3])
)

# ---- japanmap 用の色マップ（pref_code -> (R,G,B)）----
cmap = {row.code: row.color for row in result_df.itertuples()}

# ---- 可視化 ----
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(picture(cmap))
ax.axis("off")

# ---- カラーバー（元の fitness スケールで表示）----
norm = mcolors.Normalize(vmin=fmin, vmax=fmax)
sm = cm.ScalarMappable(cmap=cmap_bw, norm=norm)
sm.set_array([])  # ダミー
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Fitness", rotation=270, labelpad=15)

# 目盛りを少し見やすく（任意）
cbar.set_ticks(np.linspace(fmin, fmax, 5))

plt.tight_layout()
plt.show()



#%%
import plotly.graph_objects as go
def add_district_traces(fig, df_income, custom_colors, year_order):
    """
    Adds traces for each district to the bump chart, including lines and annotations.
    """
    # Add a line for each district
    for i, district in enumerate(df_income["District Name"].unique()):
        district_data = df_income[df_income["District Name"] == district]
        
        # Get the last point for the annotation
        last_point = district_data.iloc[-1]

        # Get the index of the last year in the year_order
        last_year_index = year_order.index(last_point["Year"])

        # Get the color from the custom color list (using the index 'i')
        line_color = custom_colors[i % len(custom_colors)]  

        # Add the line trace
        fig.add_trace(
            go.Scatter(
                x=district_data["Year"],
                y=district_data["Ranking"],
                mode="lines+markers",
                name=district,
                line=dict(color=line_color),  
                marker=dict(size=20), 
                hovertemplate="<br><b>Year: </b>%{x}<br><b>Ranking: </b>%{y}<br><b>Income: </b>%{customdata[0]:,.2f}<extra></extra>",  # Include 'Income' in hover
                customdata=district_data[["Income"]],  # Add 'Income' to custom data for hover
            )
        )

        # Add annotation with a buffer to the left of the last year point
        buffer = 0.05  # Adjust this value to increase or decrease the buffer
        fig.add_annotation(
            x=last_year_index + buffer,  # Add buffer to the x position
            y=last_point["Ranking"],
            text=district,
            showarrow=False,  # Remove the arrow
            font=dict(size=15, color=line_color),  # Match the color of the label with the line
            xanchor="left",  # Align text to the left of the specified x position
        )

def add_subtitle(fig, subtitle, subtitle_font_size=14, subtitle_color="gray", y_offset=0.92, x_offset=0.5):
    """
    Adds a subtitle to a Plotly figure.
    """
    fig.add_annotation(
        text=subtitle,
        x=x_offset,  # Horizontal position
        y=y_offset,  # Vertical position
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=subtitle_font_size, color=subtitle_color),
        align="center",
    )
    return fig

def add_footer(fig, footer, footer_font_size=12, footer_color="gray", y_offset=-0.1, x_offset=0.5):
    """
    Adds a footer to a Plotly figure.
    """
    fig.add_annotation(
        text=footer,
        x=x_offset,  # Horizontal position
        y=y_offset,  # Vertical position 
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=footer_font_size, color=footer_color),
        align="center",
    )
    return fig

def customize_layout(fig):
    """
    Customizes the layout of the figure, including titles, axis settings, and grid visibility.
    """
    # Invert the y-axis because higher rankings should appear at the top
    fig.update_yaxes(
        autorange="reversed",
        tickmode="linear",
        dtick=1,
        showticklabels=False  # Remove y-axis labels
    )

    # Customize layout settings (titles, grid, etc.)
    fig.update_layout(
        title={
        "text": "Ranking of Valencia Districts by Income: 2015 vs. 2022",
        "y": 0.925,  # Adjust vertical position (default is 1.0)
    },
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(family='Poppins'),
        height=700, 
        width=700,   
        showlegend=False,  # Remove the legend
        xaxis_showgrid=False,  # Remove grid lines from the x-axis
        yaxis_showgrid=False,  # Remove grid lines from the y-axis
    )
def add_ranking_annotations(fig, df_income, year_order):
    """
    Adds annotations for the ranking at each marker on the bump chart.
    """
    for district in df_income["District Name"].unique():
        district_data = df_income[df_income["District Name"] == district]
        for _, row in district_data.iterrows():
            # Get the position of the year in the year order
            year_index = year_order.index(row["Year"])
            y_offset = 0.45
            fig.add_annotation(
                x=year_index,
                y=row["Ranking"] + y_offset,
                text=str(int(row["Ranking"])),  # Display the ranking as an integer
                showarrow=False,
                font=dict(size=12, color="white"),  # Customize font size and color
                xanchor="center",  # Align text to the marker
                yanchor="bottom",  # Position the text just below the marker
            )
def get_custom_colors(background="light"):
    """
    Returns a list of custom colors in hex format for the bump chart lines.
    Based on the background type (light or dark).
    Parameters:
        background (str): Type of background, either "light" or "dark".
    Returns:
        list: A list of custom colors for the specified background type.
    """
    # Colors for light background
    colors_for_light_bg = [
        "#4B7CCC",  # Medium Blue
        "#F2668B",  # Pink
        "#03A688",  # Medium Teal
        "#FFAE3E",  # Amber
        "#B782B8",  # Purple
        "#A67F63",  # Chestnut
        "#0E8B92",  # Deep Teal
        "#D4AC2C",  # Bronze Yellow
        "#7E9F5C",  # Forest Green
        "#F7BCA3",  # Burnt Orange
        "#E63946",  # Red
        "#7DA9A7",  # Slate Blue
        "#457B9D",  # Steel Blue
        "#E094AC",  # Rose
        "#1D3557",  # Dark Blue
        "#2A9D8F",  # Teal Green
        "#B38A44",  # Antique Gold
        "#C68045",  # Copper
        "#264653",  # Charcoal Blue
    ]

    # Colors for dark background
    colors_for_dark_bg = [
        "#A8C9F4",  # Pastel Blue
        "#FFB3C6",  # Pastel Pink
        "#A0E5D6",  # Soft Teal
        "#FFEC88",  # Pastel Yellow
        "#E2A8D3",  # Pastel Lavender
        "#D0B89D",  # Soft Tan
        "#80D0D4",  # Light Aqua
        "#F1E59B",  # Pale Lemon
        "#B4D79E",  # Light Mint Green
        "#F2B48C",  # Soft Coral
        "#FFB4B4",  # Light Red
        "#A2D1D1",  # Light Cyan
        "#7FBCD1",  # Light Steel Blue
        "#F1A8D6",  # Light Rose
        "#5C7F9E",  # Light Denim Blue
        "#70C7B7",  # Soft Green-Teal
        "#C8A67F",  # Pastel Gold
        "#D7A584",  # Light Copper
        "#A4B6D4",  # Soft Charcoal Blue
    ]

    if background == "dark":
        return colors_for_dark_bg
    else:
        return colors_for_light_bg
def create_bump_chart(df_income):
    """
    Creates and displays the bump chart using the provided dataframe.
    """
    # Get the list of custom colors
    custom_colors = get_custom_colors()  

    # Create the bump chart figure
    fig = go.Figure()

    # Add district traces to the figure
    year_order = sorted(df_income["Year"].unique()) 
    add_district_traces(fig, df_income, custom_colors, year_order)

    # Add a subtitle to the figure
    subtitle = "Visualizing changes in income across Valencia's districts over seven years"
    add_subtitle(fig, subtitle, subtitle_font_size=15, subtitle_color="grey", y_offset=1.050, x_offset=-0.0875)

    # Add a footer to the figure
    footer = "Source: INE (Instituto Nacional de Estadística). Data retrieved on November 2024."
    add_footer(fig, footer, footer_font_size=12, footer_color="grey", y_offset=-0.1, x_offset=0.35)

    # Customize the layout of the figure
    customize_layout(fig)

    # Show the figure
    fig.show()


# Assuming df_income is your DataFrame
create_bump_chart(df_income)


# %%
pref_df = pd.merge(
    long_mcp,
    result_df.assign(
        rank = lambda x: x['fitness'].rank(method='min', ascending=False).astype(np.int64)
    ).sort_values(by='rank', ascending=True, ignore_index=True),
    on=['prefecture'],
    how='left'
    )
# %%

def aggregate_prefecture(df: pd.DataFrame) -> pd.DataFrame:
    """
    prefectureごとに集計を行い、新しいDataFrameを作成する。

    Args:
        df (pd.DataFrame): 入力DataFrame
            必須列: ["prefecture","schmoch35","patent_count","rta","class_q",
                     "mpc","app_nendo_period","fitness","rank"]

    Returns:
        pd.DataFrame: 集計後のDataFrame
    """
    agg_df = (
        df.groupby("prefecture", as_index=False)
        .agg({
            # schmoch35は削除 → 集計しない
            "patent_count": ["sum", list],  # 合計列とlist列を作成
            "rta": list,
            "class_q": list,
            "mpc": "sum",  # degree_centrality
            "app_nendo_period": lambda x: list(set(x))[0],
            "fitness": lambda x: list(set(x))[0],
            "rank": lambda x: list(set(x))[0]
        })
    )

    # MultiIndex列をフラット化
    agg_df.columns = [
        "prefecture",
        "patent_count_sum",
        "patent_count_list",
        "rta_list",
        "class_q_list",
        "degree_centrality",
        "app_nendo_period_unique",
        "fitness_unique",
        "rank_unique",
    ]

    return agg_df

# %%
grp_df = pd.read_csv(
    '../../data/processed/external/grp/2015.csv',
    encoding='utf-8',
    sep=',',
)
grp_df
# %%
agg_pref_df = aggregate_prefecture(pref_df)\
              .merge(
    grp_df,
    left_on='prefecture',
    right_on='prefecture',
    how='left'
)
agg_pref_df
# %%

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from typing import Dict, Tuple, Optional

# --- 末尾付きの辞書（略称, 地域） ---
pref_dict: Dict[str, Tuple[str, str]] = {
    "北海道": ("HK", "Hokkaido"),
    "青森県": ("AO", "Tohoku"),
    "岩手県": ("IW", "Tohoku"),
    "宮城県": ("MG", "Tohoku"),
    "秋田県": ("AK", "Tohoku"),
    "山形県": ("YT", "Tohoku"),
    "福島県": ("FS", "Tohoku"),
    "茨城県": ("IB", "Kanto"),
    "栃木県": ("TC", "Kanto"),
    "群馬県": ("GM", "Kanto"),
    "埼玉県": ("ST", "Kanto"),
    "千葉県": ("CH", "Kanto"),
    "東京都": ("TK", "Kanto"),
    "神奈川県": ("KN", "Kanto"),
    "新潟県": ("NI", "Chubu"),
    "富山県": ("TY", "Chubu"),
    "石川県": ("IS", "Chubu"),
    "福井県": ("FI", "Chubu"),
    "山梨県": ("YN", "Chubu"),
    "長野県": ("NN", "Chubu"),
    "岐阜県": ("GF", "Chubu"),
    "静岡県": ("SZ", "Chubu"),
    "愛知県": ("AI", "Chubu"),
    "三重県": ("ME", "Kansai"),
    "滋賀県": ("SH", "Kansai"),
    "京都府": ("KY", "Kansai"),
    "大阪府": ("OS", "Kansai"),
    "兵庫県": ("HG", "Kansai"),
    "奈良県": ("NR", "Kansai"),
    "和歌山県": ("WK", "Kansai"),
    "鳥取県": ("TT", "Chugoku"),
    "島根県": ("SM", "Chugoku"),
    "岡山県": ("OY", "Chugoku"),
    "広島県": ("HS", "Chugoku"),
    "山口県": ("YC", "Chugoku"),
    "徳島県": ("TS", "Shikoku"),
    "香川県": ("KG", "Shikoku"),
    "愛媛県": ("EH", "Shikoku"),
    "高知県": ("KC", "Shikoku"),
    "福岡県": ("FO", "Kyushu"),
    "佐賀県": ("SG", "Kyushu"),
    "長崎県": ("NS", "Kyushu"),
    "熊本県": ("KM", "Kyushu"),
    "大分県": ("OT", "Kyushu"),
    "宮崎県": ("MZ", "Kyushu"),
    "鹿児島県": ("KS", "Kyushu"),
    "沖縄県": ("ON", "Kyushu"),
}

# 地域→色
region_colors = {
    "Hokkaido": "tab:gray",
    "Tohoku": "navy",
    "Kanto": "tab:red",
    "Chubu": "tab:blue",
    "Kansai": "tab:orange",
    "Chugoku": "tab:green",
    "Shikoku": "purple",
    "Kyushu": "tab:brown",
}

# 軸ラベル変換
label_map = {
    "kci": "KCI",
    "kh_1": r"Ubiquity ($k_{p,1}$)",
    "diversity": r"Diversity ($k_{p,0}$)",
    "patent_count": "Patent counts",
}

def _abbr_and_region(jp_name: str) -> Tuple[str, Optional[str]]:
    """都道府県名から (略称, 地域) を返す"""
    if jp_name in pref_dict:
        return pref_dict[jp_name][0], pref_dict[jp_name][1]
    return jp_name[:2], None  # fallback

def plot_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    region_coloring: bool = True,
    show_legend: bool = True,
    s: int = 30,
) -> None:
    """主要指標の散布図（略称ラベル + 地域色分け + 参照線 + ログ軸）を描画する"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # region/abbr 列を作成
    _tmp = df.copy()
    _tmp["__abbr__"], _tmp["__region__"] = zip(*_tmp["prefecture"].map(_abbr_and_region))

    # 散布
    texts = []
    handles = []
    if region_coloring:
        for region, sub in _tmp.groupby("__region__"):
            color = region_colors.get(region, "tab:red")
            sc = ax.scatter(sub[xcol], sub[ycol], s=s, color=color, label=region or "Unknown", alpha=0.8)
            handles.append(sc)
            for _, row in sub.iterrows():
                texts.append(ax.text(row[xcol], row[ycol], row["__abbr__"], fontsize=8))
    else:
        ax.scatter(_tmp[xcol], _tmp[ycol], s=s, color="tab:red")
        for _, row in _tmp.iterrows():
            texts.append(ax.text(row[xcol], row[ycol], row["__abbr__"], fontsize=8))

    # log scale
    if "patent_count" in xcol:
        ax.set_xscale("log")
    if "patent_count" in ycol:
        ax.set_yscale("log")

    # 参照線
    def add_reference_line(col: str, axis: str) -> None:
        val = 0.0 if col == "kci" else float(df[col].mean())
        if axis == "x":
            ax.axvline(val, color="gray", linestyle="--", lw=1)
        else:
            ax.axhline(val, color="gray", linestyle="--", lw=1)
        if col == "diversity":
            if axis == "x":
                ax.set_xlim(-2, 37)
                ax.set_xticks(range(0, 35+1, 5))
                ax.set_xticklabels(range(0, 35+1, 5))
            else:
                ax.set_ylim(-2, 37)
                ax.set_yticks(range(0, 35+1, 5))
                ax.set_yticklabels(range(0, 35+1, 5))

    add_reference_line(xcol, "x")
    add_reference_line(ycol, "y")

    # 軸ラベル・タイトル
    xlabel = label_map.get(xcol, xcol)
    ylabel = label_map.get(ycol, ycol)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel} vs. {ylabel} (corr: {df[xcol].corr(df[ycol]):.2f})")
    
    # ラベル調整
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        expand_points=(1.05, 1.05),
        expand_text=(1.05, 1.05),
        force_text=0.8,
        lim=200,
    )

    # 凡例
    if region_coloring and show_legend and handles:
        ax.legend(frameon=False, title="Region", loc="best", fontsize=9)

    # plt.tight_layout()
    # ax.set_aspect('equal', adjustable='box')
    
    plt.show()

#%%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'degree_centrality',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'fitness_unique',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'degree_centrality',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'degree_centrality',
    'fitness_unique',
    region_coloring=False
)

# %%
plot_scatter(
    agg_pref_df,
    'patent_count_sum',
    'GRP_per_capita_1000yen',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'degree_centrality',
    'GRP_per_capita_1000yen',
    region_coloring=False
)
# %%
plot_scatter(
    agg_pref_df,
    'GRP_per_capita_1000yen',
    'fitness_unique',
    region_coloring=False
)


# %%
