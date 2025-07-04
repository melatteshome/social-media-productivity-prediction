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
    """
    Visual check + detection of outliers in a numeric Pandas Series.
    
    Parameters
    ----------
    series : pd.Series
        One column from a DataFrame (numeric).
    title : str, optional
        Custom title for the plot.
    return_mask : bool, default False
        If True, return a Boolean mask instead of the outlier values.
    
    Returns
    -------
    pd.Series or pd.Index or pd.Series[bool]
        • When `return_mask=False` (default): the outlier values  
          (empty Series if none).  
        • When `return_mask=True`: a Boolean mask you can use to
          filter the original DataFrame (all False if none).
    """
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
    
 

def fix_outliers(
    data: pd.DataFrame | pd.Series,
    columns: Iterable[str] | None = None,
    method: str | Mapping[str, str] = "remove",
    iqr_mult: float = 1.5,
) -> pd.DataFrame | pd.Series:
    """
    Fix outliers in selected columns.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    columns : iterable of str, optional
        Columns to process.  If None (default) and `data` is a Series,
        the single Series is processed; if None and `data` is a DataFrame,
        all numeric columns are processed.
    method : str or dict
        "remove", "median", "mode", or a dict mapping column→method.
    iqr_mult : float, default 1.5
        Multiplier on IQR that defines the outlier fence.

    Returns
    -------
    Same type as `data`, with outliers handled.
    """
    # Unify to DataFrame internally
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # Decide which columns we should scan
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    else:
        # keep order & ensure they exist
        columns = [col for col in columns if col in df.columns]

    # Build per-column strategy map
    if isinstance(method, Mapping):
        method_map = {col: method.get(col, "remove").lower() for col in columns}
    else:
        method_map = {col: method.lower() for col in columns}

    # Collect rows to drop if any column uses "remove"
    drop_mask = pd.Series(False, index=df.index)

    for col in columns:
        col_method = method_map[col]

        # IQR bounds
        q1, q3 = df[col].quantile([0.25, 0.75])
        fence_low  = q1 - iqr_mult * (q3 - q1)
        fence_high = q3 + iqr_mult * (q3 - q1)
        mask = (df[col] < fence_low) | (df[col] > fence_high)

        # No “if not mask.any()” guard: we trust caller that outliers exist

        if col_method == "remove":
            drop_mask |= mask

        elif col_method == "median":
            df.loc[mask, col] = df[col].median()

        elif col_method == "mode":
            df.loc[mask, col] = df[col].mode().iloc[0]

        else:
            raise ValueError(
                f"Unknown method '{col_method}' for column '{col}'. "
                "Choose 'remove', 'median', or 'mode'."
            )

    if drop_mask.any():
        df = df.loc[~drop_mask]

    return df.squeeze()  # back to Series if that’s what came in



