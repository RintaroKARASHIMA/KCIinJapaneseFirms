import pandas as pd
# import numpy as np


def by_classification(
                            df: pd.DataFrame,
                            classification: str
                            ):
    class_weight_df = (
        df.copy()
        .groupby(['right_person_name', 'reg_num'])[[classification]]
        .nunique()
        .reset_index(drop=False)
        .rename(columns={classification: 'class_weight'})
    )

    # ipc_weight_df['weight'] = round(1 / ipc_weight_df['weight'], 2)

    class_weight_df = pd.merge(
        df, class_weight_df,
        on=['reg_num', 'right_person_name'],
        how='left'
    )
    return class_weight_df


def by_applicant(
                df: pd.DataFrame
                ):
    applicant_weight_df = (
            df.copy()
            .groupby(['reg_num'])[['right_person_name']]
            .nunique()
            .reset_index(drop=False)
            .rename(columns={'right_person_name': 'applicant_weight'})
        )

    # applicant_weight_df['weight'] = round(1 / applicant_weight_df['weight'], 2)

    applicant_weight_df = pd.merge(
        df, applicant_weight_df,
        on=['reg_num'],
        how='left'
    )

    return applicant_weight_df
