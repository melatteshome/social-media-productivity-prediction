import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_univariate(
    data: pd.DataFrame | pd.Series,
    column: str | None = None,
    *,
    ax: plt.Axes | None = None,
    top_n: int | None = None,
    order: list[str] | None = None,
    palette: str | list[str] = "Set2",
    show_pct: bool = True,
    title: str | None = None,
    missing_label: str = "_missing_",
) -> plt.Axes:
    """
    Plot a bar chart of category frequencies (and optional percentages).

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Source data. If a DataFrame is passed, `column` must be supplied.
    column : str | None
        Column in `data` to analyse (ignored if `data` is already a Series).
    ax : matplotlib Axes, optional
        Existing axes to draw on. A new figure/axes are created if None.
    top_n : int, optional
        If set, only the `top_n` most frequent categories are shown (others are aggregated into "Other").
    order : list[str], optional
        Explicit ordering of categories on the x-axis. Overrides frequency order.
    palette : str or list
        Seaborn / matplotlib palette name **or** explicit color list.
    show_pct : bool, default True
        Annotate bars with percentage labels as well as counts.
    title : str, optional
        Custom plot title.
    missing_label : str, default "_missing_"
        Label assigned to missing values (NaN/None).

    Returns
    -------
    matplotlib Axes
        Axes with the plotted bar chart.

    Example
    -------
    >>> df = pd.DataFrame({"country": ["DE", "US", "US", None, "CN", "DE"]})
    >>> plot_categorical_univariate(df, "country", top_n=3)
    >>> plt.show()
    """
    # -------- prepare the Series --------
    ser = data if isinstance(data, pd.Series) else data[column]
    ser = ser.fillna(missing_label)

    counts = ser.value_counts(dropna=False)

    if top_n is not None and top_n < len(counts):
        top_categories = counts.nlargest(top_n)
        other_count = counts.sum() - top_categories.sum()
        counts = top_categories.append(pd.Series({"Other": other_count}))

    if order is not None:
        counts = counts.reindex(order, fill_value=0)

    # -------- plotting --------
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(counts) * 0.8), 4))

    sns.barplot(
        x=counts.index,
        y=counts.values,
        palette="Set2" if isinstance(palette, str) else palette,  # one colour per bar
        ax=ax,
    )

    # annotate values
    if show_pct:
        total = counts.sum()
        for p in ax.patches:
            height = p.get_height()
            pct = f"{height/total:0.1%}"
            ax.annotate(
                f"{int(height):,}\n{pct}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )


    ax.set_ylabel("Count")
    ax.set_xlabel(column if column else ser.name)
    ax.set_title(title or f"Univariate analysis of '{column or ser.name}'")

    # rotate ticks (NO ha here)
    ax.tick_params(axis="x", rotation=45)

    # align tick labels to the right
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    sns.despine(ax=ax)
    return ax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bivariate(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    kind: str | None = None,
    ax: plt.Axes | None = None,
    palette: str | list[str] = "Set2",
    bins: int = 30,
    aggfunc: str | callable = "mean",
    heatmap_stat: str = "count",           # "count", "mean", or "sum"
    annot: bool = True,
    title: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Quick bivariate plot between two columns, picking a sensible default.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe.
    x, y : str
        Columns to compare.
    kind : str, optional
        Force a plot type:
          * numeric–numeric ...... 'scatter', 'reg', 'hex', 'hist2d'
          * numeric–categorical .. 'box', 'violin', 'bar'
          * categorical–categorical 'heatmap'
        If None (default) the function decides based on dtypes.
    ax : matplotlib Axes, optional
        Supply to draw on an existing subplot.
    palette : str | list, default "Set2"
        Colour palette for box/violin/bar plots.
    bins : int, default 30
        Number of bins for 'hist2d'.
    aggfunc : str | callable, default "mean"
        Aggregator for 'bar' plots (e.g., "sum" or np.median).
    heatmap_stat : {"count","mean","sum"}, default "count"
        Statistic to display when both variables are categorical.
    annot : bool, default True
        Show values inside heat-map cells.
    title : str, optional
        Custom figure title.
    **kwargs
        Passed straight to the underlying seaborn/matplotlib call.

    Returns
    -------
    matplotlib Axes
    """
    # --- helpers -------------------------------------------------------
    def _is_num(series):
        return pd.api.types.is_numeric_dtype(series)

    x_is_num, y_is_num = _is_num(data[x]), _is_num(data[y])

    # Choose default plot type
    if kind is None:
        if x_is_num and y_is_num:
            kind = "scatter"
        elif x_is_num != y_is_num:        # one numeric, one categorical
            kind = "box"
        else:                             # both categorical
            kind = "heatmap"

    # Prepare axes
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    # --- numeric–numeric ----------------------------------------------
    if kind in {"scatter", "reg", "hex", "hist2d"}:
        if kind == "scatter":
            sns.scatterplot(data=data, x=x, y=y, ax=ax, **kwargs)
        elif kind == "reg":
            sns.regplot(data=data, x=x, y=y, ax=ax, scatter=True,
                        line_kws={"linewidth": 2}, **kwargs)
        elif kind == "hex":
            ax.hexbin(data[x], data[y], gridsize=30, cmap="Blues", **kwargs)
        else:  # 'hist2d'
            ax.hist2d(data[x], data[y], bins=bins, cmap="Blues", **kwargs)

    # --- numeric–categorical ------------------------------------------
    elif kind in {"box", "violin", "bar"}:
        plotter = dict(box=sns.boxplot,
                       violin=sns.violinplot,
                       bar=sns.barplot)[kind]
        plotter(data=data, x=x, y=y, palette=palette,
                estimator=None if kind != "bar" else getattr(np, aggfunc),
                ax=ax, **kwargs)

    # --- categorical–categorical --------------------------------------
    elif kind == "heatmap":
        if heatmap_stat == "count":
            pivot = pd.crosstab(data[y], data[x])
        else:
            values = data[y] if heatmap_stat == "sum" else data[y].astype(float)
            pivot = pd.pivot_table(data, index=y, columns=x,
                                   values=y, aggfunc=heatmap_stat)
        sns.heatmap(pivot, annot=annot, fmt=".0f", cmap="Blues", ax=ax,
                    cbar_kws={"label": heatmap_stat.capitalize()})

    else:
        raise ValueError(f"Unknown kind='{kind}'")

    # --- cosmetics -----------------------------------------------------
    ax.set_title(title or f"{kind.capitalize()} plot: {x} vs {y}")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    sns.despine(ax=ax)
    plt.tight_layout()
    return ax


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bivariate(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    kind: str | None = None,
    ax: plt.Axes | None = None,
    palette: str | list[str] = "Set2",
    bins: int = 30,
    aggfunc: str | callable = "mean",
    heatmap_stat: str = "count",           # "count", "mean", or "sum"
    annot: bool = True,
    title: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Quick bivariate plot between two columns, picking a sensible default.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe.
    x, y : str
        Columns to compare.
    kind : str, optional
        Force a plot type:
          * numeric–numeric ...... 'scatter', 'reg', 'hex', 'hist2d'
          * numeric–categorical .. 'box', 'violin', 'bar'
          * categorical–categorical 'heatmap'
        If None (default) the function decides based on dtypes.
    ax : matplotlib Axes, optional
        Supply to draw on an existing subplot.
    palette : str | list, default "Set2"
        Colour palette for box/violin/bar plots.
    bins : int, default 30
        Number of bins for 'hist2d'.
    aggfunc : str | callable, default "mean"
        Aggregator for 'bar' plots (e.g., "sum" or np.median).
    heatmap_stat : {"count","mean","sum"}, default "count"
        Statistic to display when both variables are categorical.
    annot : bool, default True
        Show values inside heat-map cells.
    title : str, optional
        Custom figure title.
    **kwargs
        Passed straight to the underlying seaborn/matplotlib call.

    Returns
    -------
    matplotlib Axes
    """
    # --- helpers -------------------------------------------------------
    def _is_num(series):
        return pd.api.types.is_numeric_dtype(series)

    x_is_num, y_is_num = _is_num(data[x]), _is_num(data[y])

    # Choose default plot type
    if kind is None:
        if x_is_num and y_is_num:
            kind = "scatter"
        elif x_is_num != y_is_num:        # one numeric, one categorical
            kind = "box"
        else:                             # both categorical
            kind = "heatmap"

    # Prepare axes
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    # --- numeric–numeric ----------------------------------------------
    if kind in {"scatter", "reg", "hex", "hist2d"}:
        if kind == "scatter":
            sns.scatterplot(data=data, x=x, y=y, ax=ax, **kwargs)
        elif kind == "reg":
            sns.regplot(data=data, x=x, y=y, ax=ax, scatter=True,
                        line_kws={"linewidth": 2}, **kwargs)
        elif kind == "hex":
            ax.hexbin(data[x], data[y], gridsize=30, cmap="Blues", **kwargs)
        else:  # 'hist2d'
            ax.hist2d(data[x], data[y], bins=bins, cmap="Blues", **kwargs)

    # --- numeric–categorical ------------------------------------------
    elif kind in {"box", "violin", "bar"}:
        plotter = dict(box=sns.boxplot,
                       violin=sns.violinplot,
                       bar=sns.barplot)[kind]
        plotter(data=data, x=x, y=y, palette=palette,
                estimator=None if kind != "bar" else getattr(np, aggfunc),
                ax=ax, **kwargs)

    # --- categorical–categorical --------------------------------------
    elif kind == "heatmap":
        if heatmap_stat == "count":
            pivot = pd.crosstab(data[y], data[x])
        else:
            values = data[y] if heatmap_stat == "sum" else data[y].astype(float)
            pivot = pd.pivot_table(data, index=y, columns=x,
                                   values=y, aggfunc=heatmap_stat)
        sns.heatmap(pivot, annot=annot, fmt=".0f", cmap="Blues", ax=ax,
                    cbar_kws={"label": heatmap_stat.capitalize()})

    else:
        raise ValueError(f"Unknown kind='{kind}'")

    # --- cosmetics -----------------------------------------------------
    ax.set_title(title or f"{kind.capitalize()} plot: {x} vs {y}")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    sns.despine(ax=ax)
    plt.tight_layout()
    return ax
