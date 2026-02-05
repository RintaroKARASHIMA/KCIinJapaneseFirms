import numpy as np
import pandas as pd
from typing import Literal, Union


def compute_pref_schmoch_lq(
    df: pd.DataFrame,
    aggregate: Literal[True, False] = True,
    *,
    producer_col: str = "prefecture",
    class_col: str = "schmoch35",
    count_col: str = "patent_count",
    linkage: Literal["CM", "RTA"] = "RTA",
    rta_threshold: int = 1,
    cm_threshold: Union[float, Literal["outlier"]] = "outlier",
) -> pd.DataFrame:
    """Compute LQ (RTA/RCA-like) and mpc based on linkage rule.

    This function computes:
      rta = (count_{p,k} / sum_p count_{p,k}) / (sum_k count_{p,k} / sum_{p,k} count_{p,k})

    Then defines mpc as:
      - linkage == "RTA":
          mpc = 1 if rta >= rta_threshold else 0
      - linkage == "CM":
          mpc = 1 if (rta >= rta_threshold) AND (count >= threshold_k) else 0
          where threshold_k is:
            * if cm_threshold is float: threshold_k = cm_threshold (constant for all classes)
            * if cm_threshold == "outlier": threshold_k = (Q3 - Q1) * 1.5 computed within each class

    Args:
        df: DataFrame containing at least (producer_col, class_col, count_col).
        aggregate: If True, pre-aggregate counts by (producer_col, class_col).
        producer_col: Column name of location/prefecture.
        class_col: Column name of technology class.
        count_col: Column name of patent count for (producer, class).
        linkage: "RTA" or "CM" linkage rule for defining mpc.
        rta_threshold: Threshold for rta. Rows with rta >= rta_threshold are considered active.
        cm_threshold: Threshold for count_col used only when linkage == "CM".
            If float, uses that constant value.
            If "outlier", uses per-class (Q3 - Q1) * 1.5.

    Returns:
        DataFrame with columns:
          [producer_col, class_col, count_col, rta, cm_cutoff, mpc]
        where cm_cutoff is the cutoff used for the CM rule:
          - NaN when linkage == "RTA"
          - constant (float) when linkage == "CM" and cm_threshold is float
          - per-class value when linkage == "CM" and cm_threshold == "outlier"
    """
    cols = [producer_col, class_col, count_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    if linkage not in {"CM", "RTA"}:
        raise ValueError("linkage must be either 'CM' or 'RTA'")

    if rta_threshold is None:
        raise ValueError("rta_threshold must not be None")

    if linkage == "CM":
        if isinstance(cm_threshold, str) and cm_threshold != "outlier":
            raise ValueError("cm_threshold must be a float or 'outlier'")

    # 必要列だけ抽出（メモリ節約）
    base = df[cols]

    # 事前集計（重複( producer,class )がある場合を安全に処理）
    if aggregate:
        base = (
            base.groupby([producer_col, class_col], observed=True, sort=False, as_index=False)[count_col]
            .sum()
        )

    # 総計（スカラー）
    total = float(base[count_col].sum())
    if total == 0.0:
        out = base.copy()
        out["rta"] = np.nan
        out["cm_cutoff"] = np.nan
        out["mpc"] = 0
        return out

    # rta 計算
    result = (
        base.assign(
            _c_total=lambda x: x.groupby(class_col, observed=True)[count_col].transform("sum"),
            _p_total=lambda x: x.groupby(producer_col, observed=True)[count_col].transform("sum"),
        )
        .assign(
            rta=lambda x: (x[count_col] / x["_c_total"]) / (x["_p_total"] / total)
        )
        .drop(columns=["_c_total", "_p_total"])
    )

    # CM 用のカットオフ（必要な場合のみ）
    if linkage == "CM":
        if isinstance(cm_threshold, str) and cm_threshold == "outlier":
            result = result.assign(
                cm_cutoff=lambda x: x.groupby(class_col, observed=True)[count_col].transform(
                    lambda s: (s.quantile(0.75) - s.quantile(0.25)) * 1.5
                )
            )
        else:
            cm_value = float(cm_threshold)
            result = result.assign(cm_cutoff=cm_value)
    else:
        result = result.assign(cm_cutoff=np.nan)

    # mpc 作成
    if linkage == "RTA":
        result = result.assign(
            mpc=lambda x: (x["rta"] >= float(rta_threshold)).astype(np.int64)
        )
    elif linkage == "CM":
        result = result.assign(
            mpc=lambda x: (
                (x["rta"] >= float(rta_threshold)) & (x[count_col] >= x["cm_cutoff"])
            ).astype(np.int64)
        )
    else:
        raise ValueError("linkage must be either 'RTA' or 'CM'")

    return result
