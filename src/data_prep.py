from __future__ import annotations
from typing import Iterable, Mapping, Hashable, Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

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



def prepare_dataframe(
    df: pd.DataFrame,
    dt_columns: Iterable[Hashable] | None = None,
    **impute_kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper that:
    1. Cleans column names
    2. Removes exact-row duplicates
    3. Imputes missing values
    4. Converts dt_columns to UTC

    Extra keyword args are forwarded to `impute_missing`.
    """
    df1 = standardize_column_names(df)
    df2, _ = deduplicate(df1)
    df3 = impute_missing(df2, **impute_kwargs)
    if dt_columns:
        df3 = standardise_datetime(df3, dt_columns, set_tz="Africa/Addis_Ababa", to_utc=True)
    return df3
