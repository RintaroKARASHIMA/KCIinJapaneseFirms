import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_reg_num_cdf(agg_df: pd.DataFrame, 
                      year_start: int, 
                      year_end: int, 
                      ) -> None:
    if agg_df['period'].nunique() == 1:
        cumsum = np.cumsum(
                            np.bincount(agg_df.query(f'period == {year_start}-{year_end}')\
                                        ['reg_num'])
                            )
        cdf = cumsum / cumsum.sum()
        ccdf = 1 - cdf
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        ax.scatter(
            range(1, len(cdf)+1),
            cdf,
            color='red',
            s=15
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Reg Num')
        ax.set_ylabel('CDF')
        ax.set_title('Reg Num CCDF')
        plt.show()
    else:
        cumsum_list = [
            np.cumsum(
                np.bincount(agg_df.query(f'period != {period}')['reg_num'])
            )
            for period in agg_df['period'].unique() if period != f'{year_start}-{year_end}'
        ]
        row_count = len(cumsum_list)//3 if len(cumsum_list)%3 == 0 else len(cumsum_list)//3 + 1
        fig, ax = plt.subplots(nrows=row_count, ncols=3, figsize=(15, row_count*5), dpi=300)
        ax = ax.flatten()
        for i, cumsum in enumerate(cumsum_list):
            cdf = cumsum / cumsum.sum()
            ccdf = 1 - cdf
            ax[i].scatter(
                range(1, len(cdf)+1),
                cdf,
                color='red',
                s=15
            )
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax[i].set_xlabel('Reg Num')
            ax[i].set_ylabel('CDF')
            ax[i].set_title('Reg Num CCDF')
        plt.show()
    return None


def reg_num_filter(weighted_df: pd.DataFrame, 
                ar: str,
                year_style: str,
                start_year: int,
                end_year: int,
                year_range: int,
                applicant_weight: str,
                class_weight: str,
                extract_population: str, 
                top_p_or_num: str, 
                top_p_or_num_value: int) -> pd.DataFrame:
    col_dict = {
        'duplication_fraction': 'class_fraction',
        'fraction_duplication': 'applicant_fraction',
        'fraction_fraction': 'both_fraction',
        }
    if f'{applicant_weight}_{class_weight}' == 'duplication_duplication':
        raise ValueError(f'{applicant_weight}_{class_weight} is not supported')
        return None
    weighted_df = weighted_df.filter(
                                items=[f'{ar}_{year_style}', extract_population,
                                       'reg_num', col_dict[f'{applicant_weight}_{class_weight}']],
                               )\
                            .query(f'{start_year} <= {ar}_{year_style} <= {end_year}', 
                                engine='python')\
                            .drop_duplicates(keep='first', ignore_index=True)
    long_df = weighted_df.groupby([extract_population], as_index=False)\
                         .agg(
                              weight = (col_dict[f'{applicant_weight}_{class_weight}'], 'sum'),
                             )\
                         .assign(
                                 period = f'{start_year}-{end_year}',
                                 top_flag = lambda x: np.where(x['weight'] >= x['weight'].quantile((100-top_p_or_num_value) / 100), 1, 0) 
                                                               if top_p_or_num == 'p'
                                                               else np.where(x['weight'].isin(x['weight'].nlargest(top_p_or_num_value)), 1, 0),
                         )
                         
    sep_df = pd.concat(
        [weighted_df.query(f'{year} <= {ar}_{year_style} <= {year+year_range-1}')\
                    .groupby([extract_population], as_index=False)\
                    .agg(
                        weight = (col_dict[f'{applicant_weight}_{class_weight}'], 'sum'),
                        )\
                    .assign(
                        period = f'{year}-{year+year_range-1}',
                        top_flag = lambda x: np.where(x['weight'] >= x['weight'].quantile((100-top_p_or_num_value) / 100), 1, 0) 
                                                              if top_p_or_num == 'p'
                                                              else np.where(x['weight'].isin(x['weight'].nlargest(top_p_or_num_value)), 1, 0),
                    )
         for year in range(start_year, end_year+1, year_range)],
        axis='index', ignore_index=True
    )
    plot_reg_num_cdf(long_df, start_year, end_year, year_range)
    plot_reg_num_cdf(sep_df, start_year, end_year, year_range)
    return pd.concat([long_df, sep_df], axis='index', ignore_index=True)\
             .query('top_flag == 1')\
             .drop(columns=['top_flag'])