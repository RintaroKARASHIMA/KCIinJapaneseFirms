import networkx as nx
from networkx.algorithms import bipartite as bip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator
import matplotlib.ticker as ptick
from pylab import *
from matplotlib import rc

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 15

# RCAの分布
def rca(rca_tbl: pd.pivot_table, 
        prop_dict: dict={'figsize': (12, 8), 
                         'title': 'RTA(顕示比較優位指数)の確率密度分布', 
                         'xlabel': 'ln(RTA)', 
                         'ylabel': 'lnP(ln(RTA))'
                         }):
    """_summary_
    
    Args:
        rca_tbl (pd.pivot_table): RCA行列
        prop_dict (dict, optional): グラフのプロパティ. 
        Defaults to {'figsize': (12, 8), 'title': 'RTA(顕示比較優位指数)の確率密度分布', 'xlabel': 'ln(RTA)', 'ylabel': 'lnP(ln(RTA))'}.
        
    Returns:
        None
    """
    rca_array = rca_tbl.copy().to_numpy()
    rca_log_array = np.round(
                        np.log(rca_array[rca_array != 0])
                            , 2)
    rca_log_list = rca_log_array.flatten().tolist()

    freq_list = []
    xtick_list = []
    for r in range(int(min(rca_log_list)), int(max(rca_log_list))+1):
        freq_list.append(sum([1 for i in rca_log_list if r-0.25<=i<r+0.25]))
        freq_list.append(sum([1 for i in rca_log_list if r+0.25<=i<r+0.75]))
        xtick_list.append(r)
    freq_array = np.array(freq_list)
    freq_array = np.log(freq_array / freq_array.sum())

    plt.figure(figsize=prop_dict['figsize'])
    plt.title(str(prop_dict['title']), fontsize=20)
    plt.axvline(x=abs(int(min(rca_log_list))*2), 
                color='tab:red', linestyle=':', linewidth=3)
    plt.axvline(x=abs(int(min(rca_log_list))*2)-10**(-3), 
                color='tab:red', linestyle=':', linewidth=3)

    plt.plot(range(len(freq_array)), freq_array, 
            'o', markersize=5, color='tab:blue')
    plt.grid(color='gray',linestyle=':')
    plt.xticks(range(0, len(xtick_list)*2, 2), xtick_list, fontsize=15)

    plt.xlabel(prop_dict['xlabel'], fontsize=15)
    plt.ylabel(prop_dict['ylabel'], fontsize=15)
    # plt.show()


# 二部グラフの生成と記述統計
def create_bipertite(adj_melted_df: pd.DataFrame):
    """_summary_
    
    Args:
        adj_melted_df (pd.DataFrame): RCA隣接行列をmeltしたもの
    
    Returns:
        dict: 二部グラフの情報
    """
    edge_df = adj_melted_df.copy()
    hr_list = list(edge_df['right_person_name'].unique())
    ipc_list = list(edge_df['ipc_class'].unique())
    edge_list = list(zip(edge_df[edge_df['value']==1]['right_person_name'], 
                        edge_df[edge_df['value']==1]['ipc_class']))
    BG = nx.Graph()
    BG.add_nodes_from(hr_list, bipartite=0)
    BG.add_nodes_from(ipc_list, bipartite=1)
    BG.add_edges_from(edge_list)

    hr_degree_dict = dict(bip.degrees(BG, hr_list)[1])

    print(BG, '\n', 
            '特許権者の数', len(hr_list), '\n', 
            'IPCクラスの数:', len(ipc_list), '\n', 
            '特許権者ノードの次数平均（次数削減前）:', np.mean(list(hr_degree_dict.values())), '\n')
    
    return {'BG': BG, 
            'hr_list': hr_list, 
            'ipc_list': ipc_list, 
            'hr_degree_dict': hr_degree_dict}

# 特許権者ノードの次数補累積分布
def hr_degree_distribution(graph, 
                            version: str='all', 
                            prop_dict: dict={'figsize':(10, 10), 
                                             'title':'特許権者ノード 次数の補累積分布', 
                                             'xlabel':'次数 Degree（=Ubiquity）', 
                                             'ylabel':'補累積密度 CCDF', 
                                             'xlim':(0.8, 10**2), 
                                             'ylim':(10**(-5), 2), 
                                             'color':'tab:green', 
                                             'figname':'hr_degree_distribution'}):
    """_summary_
    
    Args:
        graph (dict): 二部グラフの情報
        version (str, optional): 描画するグラフの種類. Defaults to 'all'.
        prop_dict (dict, optional): グラフのプロパティ. Defaults to {'figsize':(10, 10), 'title':'特許権者ノード 次数の補累積分布', 'xlabel':'次数 Degree（=Ubiquity）', 'ylabel':'補累積密度 CCDF'}.
    
    Returns:
        None
    """
    def ccdf(node_degree_dict):
        freq_array = np.array(np.bincount(list(node_degree_dict.values())))
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
    
    if version == 'all':
        hr_degree_dict = graph['hr_degree_dict']
        ccdf_array = ccdf(hr_degree_dict)
        """ Plot CCDF """
        fig, ax = plt.subplots(figsize=prop_dict['figsize'])
        ax.plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=8, 
                    color=prop_dict['color'], alpha=0.7)
        
    elif version == 'sep_year':
        color_list = [
            'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
            'grey', 'violet', 'indigo', 'turquoise', 'gold', 'lime', 'coral',
            'navy', 'skyblue', 'tomato', 'olive', 'cyan', 'darkred', 'darkgreen',
            'darkblue', 'darkorange', 'darkviolet', 'deeppink', 'firebrick', 'darkcyan',
            'darkturquoise', 'darkslategray', 'darkgoldenrod', 'mediumblue', 'mediumseagreen',
            'mediumpurple', 'mediumvioletred', 'midnightblue', 'saddlebrown', 'seagreen',
            'sienna', 'steelblue'
            ][10:]
        fig, axes = plt.subplots(len(list(graph.keys())) // 3, 3, 
                                 tight_layout=True, 
                                 sharex = 'all', 
                                 sharey = 'all', 
                                 figsize=(prop_dict['figsize'][0]//2*3, prop_dict['figsize'][1]//2+1))
        
        nrows = 0
        ncols = 0
        color_count = 0

        for period, BG_dict in graph.items():
            hr_degree_dict = BG_dict['hr_degree_dict']
            ccdf_array = ccdf(hr_degree_dict)
            if len(list(graph.keys())) // 3 == 1:
                axes[ncols].plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=6, 
                                        color=color_list[color_count], label=period, alpha=1)
                
                axes[ncols].set_xscale('log')
                axes[ncols].set_yscale('log')
                
                axes[ncols].grid(axis='both', which='major', alpha=0.5, linestyle='--', linewidth=0.8, color='gray')
                axes[ncols].legend(loc='upper right', fontsize=12)
                axes[ncols].xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
                axes[ncols].ticklabel_format(style='plain',axis='x')
            else:
                axes[nrows, ncols].plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=6, 
                                        color=color_list[color_count], label=period, alpha=1)
                
                axes[nrows, ncols].set_xscale('log')
                axes[nrows, ncols].set_yscale('log')
                
                axes[nrows, ncols].set_xlim(prop_dict['xlim'])
                axes[nrows, ncols].set_ylim(prop_dict['ylim'])
                
                axes[nrows, ncols].grid(axis='both', which='major', alpha=0.5, linestyle='--', linewidth=0.8, color='gray')
                axes[nrows, ncols].legend(loc='upper right', fontsize=12)
                axes[nrows, ncols].xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
                axes[nrows, ncols].ticklabel_format(style='plain',axis='x')
            color_count += 1
            ncols += 1
            if ncols == 3:
                ncols = 0
                nrows += 1
        fig.suptitle(prop_dict['title'], fontsize=20)
        fig.supxlabel(prop_dict['xlabel'], fontsize=18)
        fig.supylabel(prop_dict['ylabel'], fontsize=18)
        plt.savefig(prop_dict['figname']+'_1.png')
        plt.show()
        
        fig, ax = plt.subplots(figsize=prop_dict['figsize'])
        color_count = 0
        for period, BG_dict in graph.items():
            hr_degree_dict = BG_dict['hr_degree_dict']
            ccdf_array = ccdf(hr_degree_dict)
            
            """ Plot CCDF """
            ax.plot(range(1, len(ccdf_array)+1), ccdf_array, 'o', markersize=8, 
                        color=color_list[color_count], label=period, alpha=0.6)
            color_count += 1
        ax.legend(loc='upper right', fontsize=18)
    
    ax.set_title(prop_dict['title']+'\n', fontsize=20)
    ax.set_xlabel(prop_dict['xlabel'], fontsize=18)
    ax.set_ylabel(prop_dict['ylabel'], fontsize=18)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # x軸の指数表記を普通に戻す魔法
    ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    
    ax.set_xlim(prop_dict['xlim'])
    ax.set_ylim(prop_dict['ylim'])
    
    ax.grid(axis='both', 
            which='major', 
            alpha=1, 
            linestyle='--', 
            linewidth=0.8, 
            color='gray')
    plt.savefig(prop_dict['figname']+'_2.png')
    plt.show()
    
    return None


# if __name__ == '__main__':
#     rank()
#     network()
