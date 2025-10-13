#%%
import sys
sys.path.append("../../src")
import os

import numpy as np
import pandas as pd
from typing import Literal

import matplotlib.pyplot as plt
from IPython.display import display

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Yu Mincho'
plt.rcParams['font.size'] = 15

from scipy.stats import spearmanr



#%%
raw_df = pd.read_csv(
    '../../data/interim/internal/jp_filtered/japan_corporations.csv',
    encoding='utf-8',
    sep=',',
    usecols=['reg_num', 'app_nendo', 'prefecture', 'schmoch35'],
    dtype={
        'reg_num': object,
        'app_nendo': np.int64,
        'prefecture': str,
        'schmoch35': np.int64,
    }
    )\
    .drop_duplicates(
        keep='first'
    )
display(raw_df)


# %%
df = raw_df.query('1981 <= app_nendo <= 2015', engine='python')

#%%
long_df = df.assign(
    addr_count=lambda x: x.groupby('reg_num')['prefecture'].transform('nunique'),
    class_count=lambda x: x.groupby('reg_num')['schmoch35'].transform('nunique')
).assign(
    weight=lambda x: 1 / (x['addr_count'] * x['class_count'])
).groupby(['prefecture', 'schmoch35'], as_index=False)\
.agg(
    patent_count=('weight', 'sum')
)
long_df
#%%
def compute_pref_schmoch_lq(
    df: pd.DataFrame,
    aggregate: Literal[True, False] = True,
    *,
    prefecture_col: str = "prefecture",
    class_col: str = "schmoch35",
    count_col: str = "patent_count",
    q: float = 0.5,
) -> pd.DataFrame:
    """Compute LQ (a.k.a. RTA/RCA-like) and mpc with a per-class quantile cutoff.

    For each technology class k, set a_k to the q-th percentile (default: 25th)
    of the distribution of patent counts across locations. Then define:
      mpc = 1 if (rta >= 1) OR (count >= a_k), else 0.

    Args:
        df: A DataFrame containing at least (prefecture_col, class_col, count_col).
        aggregate: If True, pre-aggregate counts by (prefecture, class).
        prefecture_col: Column name of location/prefecture.
        class_col: Column name of technology class.
        count_col: Column name of patent count for (prefecture, class).
        q: Quantile for per-class cutoff a_k (default 0.25).

    Returns:
        DataFrame with columns:
          [prefecture_col, class_col, count_col, rta, class_q, mpc]
        where class_q is the per-class quantile cutoff a_k.
    """
    cols = [prefecture_col, class_col, count_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    # 必要列だけ抽出（メモリ節約）
    base = df[cols]

    # 事前集計（重複( p,c )がある場合を安全に処理）
    if aggregate:
        base = (
            base.groupby([prefecture_col, class_col], observed=True, sort=False, as_index=False)[count_col]
                .sum()
        )

    # 総計（スカラー）
    total = float(base[count_col].sum())
    if total == 0:
        # 全ゼロなら早期リターン（rta=NaN, mpc=0）
        out = base.copy()
        out["rta"] = np.nan
        out["class_q"] = 0.0
        out["mpc"] = 0
        return out

    # 各クラス内での地域別件数分布の q 分位点（= a_k）を算出し列として付与
    # ※ groupby.transform でクラスごとの同一長ベクトルを返す
    result = (
        base
        .assign(
            _c_total=lambda x: x.groupby(class_col, observed=True)[count_col].transform("sum"),
            _p_total=lambda x: x.groupby(prefecture_col, observed=True)[count_col].transform("sum"),
        )
        .assign(
            rta=lambda x: (x[count_col] / x["_c_total"]) / (x["_p_total"] / total)
        )
        .drop(columns=["_c_total", "_p_total"])
        .assign(
            class_q=lambda x: x.groupby(class_col)[count_col].transform(
                lambda s: (s.quantile(0.75)-s.quantile(0.25))*1.5
            )
        )
        .assign(
            mpc=lambda x: np.where(
                (x["rta"] >= 1.0) | (x[count_col] >= x["class_q"]),
                1, 0
            ).astype(np.int64)
        )
    )

    return result
long_mcp = compute_pref_schmoch_lq(long_df, class_col='schmoch35')
long_mcp


#%%
window_size = 5
tech = 'schmoch35'
sep_mcp = pd.concat(
                   [
                    compute_pref_schmoch_lq(
                        df.drop_duplicates(keep='first')\
                                .query('@window-@window_size+1 <= app_nendo <= @window', engine='python')\
                                .assign(
                                    addr_count=lambda x: x.groupby('reg_num')['prefecture'].transform('nunique'),
                                    class_count=lambda x: x.groupby('reg_num')[tech].transform('nunique')
                                )\
                                .assign(
                                    weight=lambda x: 1 / (x['addr_count'] * x['class_count'])
                                )\
                                .groupby(['prefecture', tech], as_index=False)\
                                .agg(
                                    patent_count=('weight', 'sum')
                                )
                                , class_col=tech
                                )\
                                .assign(
                                    app_nendo_period = lambda x: f'{window-window_size+1}-{window}'
                                )
                    for window in range(1985, 2015+1)
                    ], 
                   axis='index',
                   ignore_index=True)
sep_mcp



#%%
def define_mcp(mcp: pd.DataFrame) -> pd.DataFrame:
    
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


    # Computation of Fitness and Complexity
    def Fitn_Comp(biadjm):
        FFQQ_ERR = 10 ** -4
        spe_value = 10**-3
        bam = np.array(biadjm)
        c_len, p_len = bam.shape
        ff1 = np.ones(c_len)
        qq1 = np.ones(p_len)
        ff0 = np.sum(bam, axis=1)
        ff0 = ff0 / np.mean(ff0)
        qq0 = 1. / np.sum(bam, axis=0)
        qq0 = qq0 / np.mean(qq0)

        ff0 = ff1
        qq0 = qq1
        ff1 = np.dot(bam, qq0)
        qq1 = 1./(np.dot(bam.T, 1. / ff0))
        ff1 /= np.mean(ff1)
        qq1 /= np.mean(qq1)
        coef = spearmanr(ff0, ff1)[0]

        coef = 0.
        i=0
        while np.sum(abs(ff1 - ff0)) > FFQQ_ERR and np.sum(abs(qq1 - qq0)) > FFQQ_ERR and 1-abs(coef)>spe_value:
            i+=1
            print(i)
            ff0 = ff1
            qq0 = qq1
            ff1 = np.dot(bam, qq0)
            qq1 = 1./(np.dot(bam.T, 1. / ff0))
            ff1 /= np.mean(ff1)
            qq1 /= np.mean(qq1)
            coef = spearmanr(ff0, ff1)[0]
        return (ff0, qq0)


    # Computation of Coherent Diversification
    def coherence(biadjm):
        bam = np.array(biadjm)
        u = biadjm.sum(axis=0)       # u_p
        d = biadjm.sum(axis=1)       # d_c

        B = np.zeros((biadjm.shape[1], biadjm.shape[1]))
        for p in range(biadjm.shape[1]):
            for p2 in range(biadjm.shape[1]):
                B[p, p2] = (biadjm[:, p] * biadjm[:, p2] / d).sum() / max(u[p], u[p2])
        div = np.sum(bam,axis=1)
        gamma = np.nan_to_num(np.dot(B,bam.T).T)
        GAMMA = bam * gamma
        return np.nan_to_num(np.sum(GAMMA,axis=1)/div)

    biadjm_presence = mcp.pivot_table(
        index="prefecture",
        columns=tech,
        values="mpc",
        aggfunc="sum",
        fill_value=0
    )
    fitness, complexity = Fitn_Comp(biadjm_presence.values)
    coherence_score = coherence(biadjm_presence.values)
    prefectures = biadjm_presence.index
    result_df = pd.DataFrame(
        {
            'prefecture': prefectures,
            'fitness': fitness,
            'coherence': coherence_score
        }
    ).sort_values(by='fitness', ascending=False, ignore_index=True)
    tech_result_df = pd.DataFrame(
        {
            'tech': tech,
            'complexity': complexity
        }
    ).sort_values(by='complexity', ascending=False, ignore_index=True)
    return result_df, tech_result_df
   

#%%

#%%
long_grp_df = pd.read_csv(
    '../../data/processed/external/grp/grp_capita.csv',
    encoding='utf-8',
    sep=',',
    ).query('year in [1995, 2005]', engine='python')\
    .sort_values(by=['prefecture', 'year'], ascending=True, ignore_index=True)\
    .assign(
    #     # ln_gdp_capita = lambda x: np.log(x['GRP_per_capita_yen']), 
        gdp_capita_pct = lambda x: x.groupby('prefecture')['GRP_per_capita_yen'].pct_change()*100
    )\
    .dropna(ignore_index=True)
result_df = pd.merge(define_mcp(long_mcp)[0], long_grp_df, on='prefecture', how='left')\
    .assign(
        ln_coherence = lambda x: np.log(x['coherence']),
    )
result_df
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# カスタムカラーマップ（青→グレー→黄）
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom", ["#08306b", "gray", "tab:orange"]
)

sns.scatterplot(
    data=result_df,
    x='fitness', y='coherence',
    hue='gdp_capita_pct',
    palette='viridis',
    legend=False  # レジェンドを非表示
)

plt.xlabel('Fitness (F)')
plt.ylabel('Coherence (Γ)')

# カラーバーを追加
norm = plt.Normalize(
    result_df['gdp_capita_pct'].min(),
    result_df['gdp_capita_pct'].max()
)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('% Δ GRPpc')
plt.xscale('log')
plt.yscale('log')
#%%
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# カスタムカラーマップ（青→グレー→黄）
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom", ["#08306b", "gray", "tab:orange"]
)

sns.scatterplot(
    data=result_df.assign(
        ln_coherence = lambda x: np.log(x['coherence']),
    ),
    x='ln_coherence', y='gdp_capita_pct',
    hue='fitness',
    palette='viridis',
    legend=False  # レジェンドを非表示
)

plt.xlabel('log(Coherence (Γ))')
plt.ylabel('% Δ GRPpc')

# カラーバーを追加
norm = plt.Normalize(
    result_df['fitness'].min(),
    result_df['fitness'].max()
)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Fitness (F)')
# plt.xscale('log')
# plt.yscale('log')
#%%
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# カスタムカラーマップ（青→グレー→黄）
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom", ["#08306b", "gray", "tab:orange"]
)

sns.scatterplot(
    data=result_df,
    x='fitness', y='coherence',
    hue='gdp_capita_pct',
    palette='viridis',
    legend=False  # レジェンドを非表示
)

plt.xlabel('Fitness (F)')
plt.ylabel('log(Coherence (Γ))')

# カラーバーを追加
norm = plt.Normalize(
    result_df['gdp_capita_pct'].min(),
    result_df['gdp_capita_pct'].max()
)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('% Δ GRPpc')
plt.xscale('log')
plt.title(f'{result_df["fitness"].corr(result_df["ln_coherence"]):.3f}')
# plt.yscale('log')

#%%
import plotly.express as px

# 青→グレー→黄（連続）カラースケール
custom_colorscale = [
    (0.0, "#313695"),  # dark blue
    (0.5, "#542788"),  # purple
    (1.0, "#d73027"),  # red
]
fig = px.scatter(
    result_df,
    x="fitness",
    y="coherence",
    color="gdp_capita_pct",
    color_continuous_scale='viridis',
    hover_name="prefecture",  # ←列名に合わせて変更
    hover_data={
        "fitness":":.3g",
        "coherence":":.3g",
        "gdp_capita_pct":":.1f",
    },
    labels={
        "fitness": "Fitness (F)",
        "coherence": "Coherence (Γ)",
        "gdp_capita_pct": "% Δ GRPpc",
    },
)

# 軸をログにして、背景白・補助線なし・主軸線黒
fig.update_xaxes(
    type="log",
    showgrid=False,       # 補助線オフ
    zeroline=False,       # 0ライン非表示（ログ軸なので念のため）
    showline=True,        # 軸線を表示
    linecolor="black",    # 軸線の色
    linewidth=1
)
fig.update_yaxes(
    type="log",
    showgrid=False,
    zeroline=False,
    showline=True,
    linecolor="black",
    linewidth=1
)

fig.update_layout(
    plot_bgcolor="white",   # 背景白
    paper_bgcolor="white",  # 外側も白
    coloraxis_colorbar=dict(title="% Δ GRPpc"),
    width=600, height=500,
)

fig.show()

#%%
# 5) Fitness & Complexity（presenceを入力）
result_df = pd.concat([define_mcp(sep_mcp.query('app_nendo_period == @period', engine='python'))\
                        [0]\
                        .assign(
                                app_nendo_period = period, 
                                # rank = lambda x: x['fitness'].rank(method='min', ascending=False).astype(np.int64)
                                )
                        for period in sep_mcp['app_nendo_period'].unique()], ignore_index=True)
result_df

#%%
result_df['app_nendo_period'].unique()

# %%
pref_df = pd.merge(
    sep_mcp.groupby(['app_nendo_period', 'prefecture']).agg(
        {'mpc': 'sum', 'patent_count': 'sum'}
    ),
    result_df,
    on=['app_nendo_period', 'prefecture'],
    how='left'
    )
pref_df
# %%

# %%
grp_df = pd.read_csv(
    '../../data/processed/external/grp/grp_capita.csv',
    encoding='utf-8',
    sep=',',
    )\
    .sort_values(by=['prefecture', 'year'], ascending=True, ignore_index=True)\
    .assign(
        capita_density = lambda x: x['capita'] / x['area'],
        ln_capita_density = lambda x: np.log(x['capita_density']),
        ln_GRP = lambda x: np.log(x['GRP']),
        ln_GRP_t5 = lambda x: x.groupby('prefecture')['ln_GRP'].shift(-window_size),
        g5_bar = lambda x: (x['ln_GRP_t5'] - x['ln_GRP'])/window_size,
        ln_GRP_pc_yen = lambda x: np.log(x['GRP_per_capita_yen']), 
        ln_GRP_pc_yen_t5 = lambda x: x.groupby('prefecture')['ln_GRP_pc_yen'].shift(-window_size),
        g5_bar_pc_yen = lambda x: (x['ln_GRP_pc_yen_t5'] - x['ln_GRP_pc_yen'])/window_size,
        ln_capita = lambda x: np.log(x['capita']),
        ln_g5_bar = lambda x: np.log1p(x['g5_bar']),
        ln_g5_bar_pc_yen = lambda x: np.log1p(x['g5_bar_pc_yen']),
    )\
    .rename(columns={'year': 'tau'})\
    .drop_duplicates(keep='first', ignore_index=True)\
    .query('(1981 <= tau <= 2015) & (tau-1980-@window_size)%@window_size==0', engine='python')
grp_df
#%%
fitness_df = pref_df.copy()\
                    .assign(
                        tau = lambda x: x['app_nendo_period'].str[-4:].astype(np.int64),
                        ln_patent_count = lambda x: np.log1p(x['patent_count']), 
                        ln_fitness = lambda x: np.log(x['fitness']),
                        ln_coherence = lambda x: np.log(x['coherence']),
                        ln_mcp = lambda x: np.log(x['mpc']),
                        z_fitness = lambda x: (x['fitness'] - x['fitness'].mean()) / x['fitness'].std(),
                        fitness_lag1 = lambda x: x.groupby('prefecture')['fitness'].shift(1),
                        fitness_lag2 = lambda x: x.groupby('prefecture')['fitness'].shift(2),
                        fitness_lag3 = lambda x: x.groupby('prefecture')['fitness'].shift(3),
                        fitness_lag4 = lambda x: x.groupby('prefecture')['fitness'].shift(4),
                        fitness_lag5 = lambda x: x.groupby('prefecture')['fitness'].shift(5),
                        fitness_lead1 = lambda x: x.groupby('prefecture')['fitness'].shift(-6),
                        fitness_lead2 = lambda x: x.groupby('prefecture')['fitness'].shift(-7),
                        fitness_lead3 = lambda x: x.groupby('prefecture')['fitness'].shift(-8),
                        fitness_lead4 = lambda x: x.groupby('prefecture')['fitness'].shift(-9),
                        fitness_lead5 = lambda x: x.groupby('prefecture')['fitness'].shift(-10),
                    )
panel_df = pd.merge(
    grp_df,
    fitness_df,
    on=['prefecture', 'tau'],
    how='inner'
    )\
    .set_index(['prefecture', 'tau'])
panel_df
#%%
panel_df.columns

#%%
from linearmodels.panel import PanelOLS

#%%
# for fit in ['_lag1', '_lag2', '_lag3', '_lag4', '_lag5']:
for fit in ['_lead1', '_lead2', '_lead3', '_lead4', '_lead5']:
    print('*'*115)
    print('fitness'+fit)
    model = PanelOLS.from_formula(
           f"g5_bar_pc_yen ~ 1 + ln_GRP + fitness{fit} + ln_capita_density + ln_patent_count + EntityEffects + TimeEffects",
        # "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + ln_capita",
        data=panel_df
    )
    # Driscoll-Kraay SE (空間+時間の依存を許容)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    # res = model.fit(cov_type="clustered", cluster_entity=True)
    print(res.summary)
    
#%%

for fixed_effect in ['', '+EntityEffects', '+TimeEffects', '+EntityEffects + TimeEffects']:
    print('*'*115)
    print(fixed_effect)
    model = PanelOLS.from_formula(
           f"g5_bar_pc_yen ~ 1 + ln_GRP + ln_coherence + ln_capita_density + ln_patent_count{fixed_effect}",
        # "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + ln_capita",
        data=panel_df
    )
    # Driscoll-Kraay SE (空間+時間の依存を許容)
    res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    # res = model.fit(cov_type="clustered", cluster_entity=True)
    print(res.summary)
    
    
    #%%
    
    
#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS

# PanelData の読み込みをバージョン差に対応
try:
    from linearmodels.panel.data import PanelData  # こちらで存在する版が多い
except Exception:
    PanelData = None

# ====== 1) 推定（あなたの元コードのまま） ======
formula = "g5_bar_pc_yen ~ 1 + ln_GRP + fitness + ln_capita_density + ln_patent_count + EntityEffects + TimeEffects"
model = PanelOLS.from_formula(formula, data=panel_df)
res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
print(res.summary)

# ====== 2) PanelOLSが実際に使ったサンプルを再現 ======
y_name = "g5_bar_pc_yen"
x_vars  = ["ln_GRP", "fitness", "ln_capita_density", "ln_patent_count"]

if PanelData is not None:
    # 同期・整形を PanelData に任せるルート
    y_pd = PanelData(panel_df[y_name]).dataframe
    X_pd = PanelData(panel_df[x_vars]).dataframe
    est_df = pd.concat([y_pd, X_pd], axis=1).dropna()
else:
    # フォールバック（pandas のみ）
    # MultiIndex（entity, time）前提。列を素直に連結→同時欠損を落とす
    est_df = pd.concat([panel_df[[y_name] + x_vars]], axis=1).dropna()

# ====== 3) 二重デミーン（within: 個体×時点） ======
X = est_df[x_vars].copy()

# MultiIndex: level=0=entity, level=1=time を仮定（名前は任意）
Gi = X.groupby(level=0).transform("mean")  # 個体平均 x_i.
Gt = X.groupby(level=1).transform("mean")  # 時点平均 x_.t
G  = X.mean()                               # 全体平均 x_..

X_within = X - Gi - Gt + G

# ====== 4) VIF 計算（within 変換後） ======
X_for_vif = sm.add_constant(X_within, has_constant="add").to_numpy()
vif_vals = []
for j, name in enumerate(["const"] + x_vars):
    try:
        v = variance_inflation_factor(X_for_vif, j)
    except np.linalg.LinAlgError:
        v = np.inf
    vif_vals.append((name, float(v)))

vif_df = pd.DataFrame(vif_vals, columns=["variable", "VIF"]).sort_values("VIF", ascending=False)
print("\n[Two-way FE (within) based VIF on the exact estimation sample]")
print(vif_df)

# ====== 5) ざっと可視化（横棒） ======
import matplotlib.pyplot as plt

def plot_vif_horizontal(vif_df, threshold: float = 10.0):
    d = vif_df.sort_values("VIF", ascending=True)
    plt.figure(figsize=(7, 4))
    bars = plt.barh(d["variable"], d["VIF"], color='navy')
    plt.axvline(threshold, linestyle="--", label=f"閾値:{threshold}", color='red')
    for bar, v in zip(bars, d["VIF"]):
        plt.text(v + 0.2, bar.get_y() + bar.get_height()/2, f"{v:.2f}", va="center", fontsize=15)
    plt.xlabel("VIF"); #plt.title("VIF (Two-way FE within)")
    plt.yticks(range(0, 5), ['$ln({GRP})$','$Fitness$','$ln({capita\_density})$','$ln({patent\_count})$', 
                             f'$alpha$'][::-1], fontsize=15)
    plt.tight_layout(); plt.show()

plot_vif_horizontal(vif_df)


#%%
resid = res.resids  # MultiIndex (entity, time) 付きの Series
resid_df = resid.unstack(level=0)  # 行=時間, 列=都道府県 に変換
import statsmodels.api as sm

# 例: 1期ラグの自己相関
acf_1 = resid_df.apply(lambda x: x.autocorr(lag=3))

print(acf_1.sort_values(ascending=False))




# %%
import seaborn as sns
plt.rcParams['font.size'] = 6
fig, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(panel_df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
plt.show()
# %%
