import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def aggregate(weighted_df: pd.DataFrame, 
                ar: str,
                year_style: str,
                start_year: int,
                end_year: int,
                year_range: int,
                applicant_weight: str,
                class_weight: str,
                region_corporation: str, 
                classification: str,
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
                                items=[f'{ar}_{year_style}', 
                                       region_corporation,
                                       classification,
                                       'reg_num', col_dict[f'{applicant_weight}_{class_weight}']],
                               )\
                            .query(f'{start_year} <= {ar}_{year_style} <= {end_year}', 
                                engine='python')\
                            .drop_duplicates(keep='first', ignore_index=True)
    agg_df = weighted_df.groupby([region_corporation, classification], as_index=False)\
                         .agg(
                              weight = (col_dict[f'{applicant_weight}_{class_weight}'], 'sum'),
                             )\
                         .assign(
                                 period = f'{start_year}-{end_year}',
                         )
    return agg_df
                         
    