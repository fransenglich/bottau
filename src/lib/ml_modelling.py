import quantreo.features_engineering as fe
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

from lib import common
from lib import heatmap

"""
This file contains functions specific to ML models.
"""


def investigate(df: pd.DataFrame,
                featurenames: tuple[str],
                strategyname: str) -> pd.DataFrame:
    """Writes data and graphs that are strategy-agnostic.
    
    This is specific to features used in regressions."""

    # This is our design matrix.
    designmatrix: pd.DataFrame = pd.DataFrame()

    for name in featurenames:
        designmatrix[name] = df[name]

    # We do 3 things: add standard features, produce correlation matrix and VIF.
    # ---- corr matrix ----

    designmatrix['pct_close_futur'] = df['pct_close_futur']

    df['var'] = df['pct_close_futur'].rolling(window=common.WINDOW_SIZE).var()
    designmatrix['var'] = df['var']

    df['vol_parkinson_30'] = fe.volatility.parkinson_volatility(df,
                                                                window_size=30)
    designmatrix['vol_parkinson_30'] = df['vol_parkinson_30']

    df['vol_parkinson_60'] = fe.volatility.parkinson_volatility(df,
                                                                window_size=60)
    designmatrix['vol_parkinson_60'] = df['vol_parkinson_60']

    df['vol_ctc_30'] = fe.volatility.close_to_close_volatility(df,
                                                               window_size=30)
    designmatrix['vol_ctc_30'] = df['vol_ctc_30']

    df['vol_ctc_60'] = fe.volatility.close_to_close_volatility(df,
                                                               window_size=60)
    designmatrix['vol_ctc_60'] = df['vol_ctc_60']

    flen = len(designmatrix.columns)
    in_range = range(flen)
    pearsonmatrix = np.zeros((flen, flen), dtype=float)
    spearmanmatrix = np.zeros((flen, flen), dtype=float)

    for i in in_range:
        # This works: for l in range(ilen - (ilen - i) + 1):
        for length in in_range:
            pearsonmatrix[i, length] = designmatrix.iloc[:, i].corr(designmatrix.iloc[:, length]) # TODO why to column length?
            spearmanmatrix[i, length] = designmatrix.iloc[:, i].corr(designmatrix.iloc[:, length], method='spearman')

    cm_labels = designmatrix.columns

    # - Pearson
    fig, ax = plt.subplots()
    pm, _ = heatmap.heatmap(pearsonmatrix, cm_labels, cm_labels, ax=ax,
                            cmap="YlGn", cbarlabel="Pearson correlation coefficient")
    heatmap.annotate_heatmap(pm)

    ax.set_title("Heatmap of Pearson correlation matrix of features")
    fig.tight_layout()
    common.savefig(fig, "pearsonmatrix", strategyname)

    # - Spearman
    fig, ax = plt.subplots()
    pm, _ = heatmap.heatmap(spearmanmatrix, cm_labels, cm_labels, ax=ax,
                            cmap="YlGn", cbarlabel="Spearman correlation coefficient")
    heatmap.annotate_heatmap(pm)

    ax.set_title("Heatmap of Spearman correlation matrix of features")
    fig.tight_layout()
    common.savefig(fig, "spearmanmatrix", strategyname)

    # - Multicollinearity

    # variance_inflation_factor() needs this.
    designmatrix.dropna(inplace=True)

    vifs = [(designmatrix.columns.values[i],
             variance_inflation_factor(designmatrix, i))
             for i in range(len(designmatrix.columns))]

    with open(common.generated_file("VIFs.tex", strategyname), "w") as f:
        for name, vif in vifs:
            f.write(name.replace("_", "\\_") + " & " + str(round(vif, 2)) + " \\\\\n")

    return df
