#! (root)/notebooks/10_network_long/p1_CorporationsFiner.py python3
# -*- coding: utf-8 -*-

# %%
import networkx as nx
import numpy as np
import pandas as pd
%run ../../src/initialize/initial_conditions.py
%run 0_LoadLibraries.py


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


color_list = ['red']+[
    'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
    'grey', 'violet', 'indigo', 'turquoise', 'gold', 'lime', 'coral',
    'navy', 'skyblue', 'tomato', 'olive', 'cyan', 'darkred', 'darkgreen',
    'darkblue', 'darkorange', 'darkviolet', 'deeppink', 'firebrick', 'darkcyan',
    'darkturquoise', 'darkslategray', 'darkgoldenrod', 'mediumblue', 'mediumseagreen',
    'mediumpurple', 'mediumvioletred', 'midnightblue', 'saddlebrown', 'seagreen',
    'sienna', 'steelblue'
][10:]


def ccdf(target_array: np.array):
    freq_array = np.array(np.bincount(target_array))
    p_list = []
    cumsum = 0.0
    s = float(freq_array.sum())
    for freq in freq_array:
        if freq != 0:
            cumsum += freq / s
            p_list.apend(cumsum)
        else:
            p_list.append(1.0)
    ccdf_array = 1 - np.arrau(p_list)
    if ccdf_array[0] == 0:
        ccdf_array[0] = 1.0
    return ccdf_array


def descript_network(graph_df: pd.DataFrame,
                     hr_col: str = 'right_person_name',
                     class_col: str = 'ipc_class',
                     segment_col: str = '1981-2010'
                     ):
    # ネットワークの特性を調べる
    BG = nx.Graph()
    BG.add_nodes_from(graph_df[hr_col].unique(), bipartite=0)
    BG.add_nodes_from(graph_df[class_col].unique(), bipartite=1)
    BG.add_edges_from(list(zip(graph_df[hr_col], graph_df[class_col])))

    if nx.is_connected(BG):
        print('連結性ある')
    else:
        print('連結性ない')

    # 特許権者ノード
    hr_nodes = {n for n, d in BG.nodes(data=True) if d['bipartite'] == 0}
    hr_degree_array = ccdf(dict(bip.degrees(BG, hr_nodes)[1]).values())

    # 特許分類ノード
    ipc_nodes = set(BG) - hr_nodes
    ipc_degree_array = ccdf(dict(bip.degrees(BG, ipc_nodes)[1]).values())

    # G = nx.from_pandas_edgelist(graph_df, 'source', 'target', edge_attr=True)
    # print(nx.info(G))
    # print('平均最短距離:', nx.average_shortest_path_length(G))
    # print('平均クラスタリング係数:', nx.average_clustering(G))
    # print('直径:', nx.diameter(G))
    # print('半径:', nx.radius(G))
    # print('次数中心性:', nx.degree_centrality(G))
    # print('媒介中心性:', nx.betweenness_centrality(G))
    # print('固有ベクトル中心性:', nx.eigenvector_centrality(G))
    # print('PageRank:', nx.pagerank(G))
    # print('次数分布:', nx.degree_histogram(G))
    res_dict = {
        'network': BG}

    return BG


global data_dir, ex_dir, output_dir
data_dir = '../../data/processed/internal/graph/'
ex_dir = '../../data/processed/external/'
output_dir = '../../output/figures/'

# %%
edge_df = pd.read_csv(data_dir + f'{input_condition}_edge.csv',
                      encoding='utf-8',
                      sep=','
                      )
node_df = pd.read_csv(data_dir + f'{input_condition}_node.csv',
                      encoding='utf-8',
                      sep=','
                      )
# edge_df
# graph_dict = {}
# for s in graph_df['segment'].unique():

#        graph_df = graph_df[graph_df['segment'] == '1981-2010']
# graph_df.groupby('right_person_name')[['mcp']].sum().sort_values('mcp', ascending=False).head(10)
graph_df = pd.merge(edge_df, node_df, left_on='Source', right_on='node_id', how='left')[['Target', 'label']]\
    .rename(columns={'label': region_corporation})
graph_df = pd.merge(graph_df, node_df, left_on='Target', right_on='node_id', how='left')[['label', region_corporation]]\
    .rename(columns={'label': classification})
graph_df

# %%
schmoch_df = pd.read_csv(ex_dir + 'schmoch/35.csv', encoding='utf-8', sep=',', usecols=['Field_en', 'schmoch5'])\
    .rename(columns={'Field_en': 'schmoch35', 'schmoch5': 'schmoch5'})\
    .drop_duplicates(ignore_index=True)
schmoch_df

# %%
BG = nx.Graph()
BG.add_nodes_from(graph_df[region_corporation].unique(), bipartite=0)
BG.add_nodes_from(graph_df[classification].unique(), bipartite=1)
BG.add_edges_from(
    list(zip(graph_df[region_corporation], graph_df[classification])))

if nx.is_connected(BG):
    print('連結性ある')
else:
    print('連結性ない')

# 特許権者ノード
hr_nodes = {n for n, d in BG.nodes(data=True) if d['bipartite'] == 0}
# hr_degree_array = ccdf(dict(bip.degrees(BG, hr_nodes)[1]).values())
hr_nodes


# 特許分類ノード
ipc_nodes = set(BG) - hr_nodes
ipc_nodes
# ipc_degree_array = ccdf(dict(bip.degrees(BG, ipc_nodes)[1]).values())


# %%
# 特許権者ノードの密度
print(round(bip.density(BG, hr_nodes), 3))

# 特許分類ノードの密度
print(round(bip.density(BG, ipc_nodes), 3))


# %%
# 特許権者同士のつながり
hr_G = bip.projected_graph(BG, hr_nodes)
hr_degree_dict = dict(hr_G.degree())

print(max(hr_degree_dict, key=hr_degree_dict.get))
# hr_degree_dict

# 特許分類同士のつながり
ipc_G = bip.weighted_projected_graph(BG, ipc_nodes)
nx.to_pandas_edgelist(ipc_G)
# ipc_degree_dict = dict(ipc_G.degree())
# for n, w in dict(bip.degrees(BG, ipc_nodes)[1]).items():
#     ipc_G.nodes[n]['weight'] = w
# print(max(ipc_degree_dict, key=ipc_degree_dict.get))
# ipc_degree_dict


# %%

# Step 1: Load and prepare data
data = graph_df.copy()
# Step 2: Calculate k_c,0 and k_p,0
k_c_0 = data.groupby('right_person_name')['schmoch35'].count().to_dict()
k_p_0 = data.groupby('schmoch35')['right_person_name'].count().to_dict()

# Step 3: Construct the bipartite graph
B = nx.Graph()
B.add_nodes_from(data['right_person_name'], bipartite=0)
B.add_nodes_from(data['schmoch35'], bipartite=1)
edges = [(row['right_person_name'], row['schmoch35'])
         for idx, row in data.iterrows()]
B.add_edges_from(edges)

# Step 4: Initialize the product-product matrix M_pp with zeros
countries = list(data['right_person_name'].unique())
products = list(data['schmoch35'].unique())
M_pp = np.zeros((len(products), len(products)))

# Create dictionaries for quick lookup
country_index = {country: i for i, country in enumerate(countries)}
product_index = {product: i for i, product in enumerate(products)}

# Create adjacency matrix M_cp
M_cp = np.zeros((len(countries), len(products)))
for idx, row in data.iterrows():
    M_cp[country_index[row['right_person_name']],
         product_index[row['schmoch35']]] = 1

# Step 5: Calculate M_pp'
for p in products:
    for p_prime in products:
        if p != p_prime:
            for c in countries:
                if M_cp[country_index[c], product_index[p]] and M_cp[country_index[c], product_index[p_prime]]:
                    M_pp[product_index[p], product_index[p_prime]] += (
                        M_cp[country_index[c], product_index[p]] * M_cp[country_index[c], product_index[p_prime]]) / (k_p_0[p] * k_c_0[c])

# Step 6: Normalize the matrix
np.fill_diagonal(M_pp, 0)
mean_M_pp = np.mean(M_pp)
std_M_pp = np.std(M_pp)
M_pp_normalized = (M_pp - mean_M_pp) / std_M_pp

print('Normalized Product-Product Matrix (M_pp):')
print(M_pp_normalized)

eigenvalues, eigenvectors = np.linalg.eig(M_pp_normalized)
eigenvectors = np.real(eigenvectors)
second_largest_index = eigenvalues.argsort()[-1]
second_largest_eigenvector = eigenvectors[:, second_largest_index]
print('2番目に大きい固有値に対応する固有ベクトル:', second_largest_eigenvector)
print(second_largest_index)
TCI_dict = {}
for k, v in product_index.items():
    TCI_dict[k] = second_largest_eigenvector[v]
    print(k, second_largest_eigenvector[v])


projected_edge_df = pd.DataFrame(M_pp*100).reset_index(drop=False).melt(id_vars='index')\
    .rename(columns={'index': 'source', 'variable': 'target', 'value': 'weight'})
projected_node_df = pd.DataFrame.from_dict(product_index, orient='index').reset_index(drop=False)\
    .rename(columns={0: 'node_id', 'index': 'label'})[['node_id', 'label']]
weight_df = pd.DataFrame.from_dict(dict(bip.degrees(BG, ipc_nodes)[1]), orient='index')\
                        .reset_index(drop=False).rename(columns={0: 'weight', 'index': 'label'})[['label', 'weight']]
projected_node_df = pd.merge(
    projected_node_df, weight_df, on='label', how='left')
projected_node_df = pd.merge(projected_node_df, schmoch_df, left_on='label', right_on='schmoch35', how='left')[['node_id', 'label', 'weight', 'schmoch5']]\
    .rename(columns={'schmoch5': 'category'})

projected_edge_df.to_csv(data_dir + f'{input_condition}_projected_edge.csv',
                         encoding='utf-8',
                         sep=',',
                         index=False)

projected_node_df.to_csv(data_dir + f'{input_condition}_projected_node.csv',
                         encoding='utf-8',
                         sep=',',
                         index=False)

# projected_node_list

projected_fixed_edge_df = pd.read_csv(data_dir + f'{input_condition}_projected_edge.csv',
                                      encoding='utf-8',
                                      sep=','
                                      )
projected_fixed_edge_df = projected_fixed_edge_df.query(
    'source != target').copy()
projected_fixed_edge_df['normarized_weight'] = (projected_edge_df['weight'] - projected_edge_df['weight'].min())/(
    projected_edge_df['weight'].max() - projected_edge_df['weight'].min())
# projected_fixed_edge_df['normarized_weight'].plot(kind='hist', bins=int(np.log2(len(projected_fixed_edge_df))+1), alpha=0.5)
print(*projected_fixed_edge_df[round(projected_fixed_edge_df['normarized_weight'], 4) ==
      round(projected_fixed_edge_df['normarized_weight'].quantile(0.9), 4)]['weight'].values)

projected_fixed_edge_df['normarized_weight'].median(numeric_only=True)
projected_node_df