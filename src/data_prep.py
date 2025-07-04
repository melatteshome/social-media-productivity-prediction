from __future__ import annotations
from typing import Iterable, Mapping, Hashable, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from collections.abc import Mapping

def standardize_column_names(df: pd.DataFrame, strip: bool = True,
                       lower: bool = True, replace_spaces: bool = True) -> pd.DataFrame:
    
    cols = df.columns.astype(str)
    if strip:
        cols = cols.str.strip()
    if lower:
        cols = cols.str.lower()
    if replace_spaces:
        cols = cols.str.replace(r"\s+", "_", regex=True)
    return df.rename(columns=dict(zip(df.columns, cols)), copy=False)



def convert_columns_to_numeric(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    downcast: str | None = None,
    errors: str = "coerce",
) -> pd.DataFrame:

    # Make an explicit copy so we don't mutate the caller's DataFrame
    df_numeric = df.copy()

    for col in columns:
        if col not in df_numeric.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        # Convert and either coerce, ignore, or raise on errors as requested
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors=errors, downcast=downcast)

    return df_numeric


def missing_report(df: pd.DataFrame, sort_desc: bool = True) -> pd.DataFrame:

    total = df.shape[0]
    stats = (
        df.isna()
          .sum()
          .to_frame("n_missing")
          .assign(p_missing=lambda s: (s["n_missing"] / total).round(3))
    )
    return stats.sort_values("n_missing", ascending=not sort_desc)


def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: Literal["mean", "median", "mode"] = "mean",
    categorical_strategy: Literal["mode", "constant"] = "mode",
    fill_constant: str | int | float = "unknown",
    columns: Iterable[Hashable] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
   
    df_target = df if inplace else df.copy()

    cols = df_target.columns if columns is None else columns
    for col in cols:
        if df_target[col].isna().sum() == 0:
            continue  
        if is_numeric_dtype(df_target[col]):
            match numeric_strategy:
                case "mean":
                    val = df_target[col].mean()
                case "median":
                    val = df_target[col].median()
                case "mode":
                    val = df_target[col].mode(dropna=True).iat[0]
        else:
            match categorical_strategy:
                case "mode":
                    val = df_target[col].mode(dropna=True).iat[0]
                case "constant":
                    val = fill_constant

        df_target[col].fillna(val, inplace=True)

    return df_target



def deduplicate(
    df: pd.DataFrame,
    subset: Iterable[Hashable] | None = None,
    keep: Literal["first", "last", False] = "first",
    drop: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    mask_dupes = df.duplicated(subset=subset, keep=keep)
    dup_rows = df.loc[mask_dupes]
    kept_df = df.drop_duplicates(subset=subset, keep=keep) if drop else df.copy()
    return kept_df, dup_rows


def standardise_datetime(
    df: pd.DataFrame,
    dt_cols: Iterable[Hashable],
    current_unit: Literal["s", "ms", "us", "ns"] | None = None,
    set_tz: str | None = None,
    to_utc: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:

    df_target = df if inplace else df.copy()

    for col in dt_cols:
        series = df_target[col]

        # Convert numeric epoch → datetime
        if pd.api.types.is_integer_dtype(series) and current_unit:
            series = pd.to_datetime(series, unit=current_unit, utc=True)
        else:
            series = pd.to_datetime(series, utc=False, errors="coerce")

        if set_tz and series.dt.tz is None:
            series = series.dt.tz_localize(set_tz)

        if to_utc:
            series = series.dt.tz_convert("UTC")

        df_target[col] = series

    return df_target





def show_outliers(series: pd.Series, *, title: str = None, return_mask: bool = False , summarize= False):

    # Drop NaNs to avoid skewing the box plot or stats
    ser = series.dropna()
    if ser.empty:
        raise ValueError("Series is empty or only contains NaNs.")

    # ── 1. Visual: box plot ─────────────────────────────────────────────
    plt.figure(figsize=(4, 1.5))  # skinny box plot
    plt.boxplot(ser, vert=False, widths=0.6, patch_artist=True)
    plt.title(title or f"Box plot of '{series.name}'")
    plt.xlabel(series.name)
    plt.tight_layout()
    plt.show()

        # ── 2. Numeric: IQR method ──────────────────────────────────────────
    q1, q3 = ser.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (ser < lower) | (ser > upper)

    if summarize:
        count   = mask.sum()
        share   = mask.mean()
        print(f"> {count} outlier(s) "
              f"= {share:.2%} of {mask.size} rows")


    if return_mask:
        return mask.reindex(series.index, fill_value=False)  # align with original index
    else:
        return ser[mask]
    
 

import pandas as pd
import numpy as np

def fix_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "remove",
    iqr_mult: float = 1.5,
) -> pd.DataFrame:
    
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        column = pd.to_numeric(column)
        
        # raise TypeError(f"Column '{column}' must be numeric.")

    out_df = df.copy()

    # IQR fences
    q1, q3 = out_df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    low_fence  = q1 - iqr_mult * iqr
    high_fence = q3 + iqr_mult * iqr
    mask = (out_df[column] < low_fence) | (out_df[column] > high_fence)

    if method.lower() == "remove":
        out_df = out_df.loc[~mask]

    elif method.lower() == "median":
        median_val = out_df[column].median()
        out_df.loc[mask, column] = median_val

    elif method.lower() == "mode":
        mode_val = out_df[column].mode().iloc[0]
        out_df.loc[mask, column] = mode_val

    else:
        raise ValueError("method must be 'remove', 'median', or 'mode'")

    return out_df
