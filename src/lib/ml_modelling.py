import logging
logger = logging.getLogger("BotTau")

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantreo.features_engineering as fe
import scipy.stats

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

    # Volatility Features
    designmatrix['pct_close_futur'] = df['pct_close_futur']

    # df['var'] = df['pct_close_futur'].rolling(window=common.WINDOW_SIZE).var()
    # designmatrix['var'] = df['var']

    # TODO we have 3 DataFrames now: designmatrix, df and df_vol_scaled. Clean up

    df['vol_std'] = df['close'].rolling(window=common.WINDOW_SIZE).std()
    designmatrix['vol_std'] = df['vol_std']

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

    vol_features = ["vol_std",
                    "vol_parkinson_30",
                    "vol_parkinson_60",
                    "vol_ctc_30",
                    "vol_ctc_60"]

    # We got NaNs in the beginning from the partial windows for the vol
    # features.
    df.dropna(inplace=True)

    # PCA
    # Define the train size
    train_size = int(len(df) * 0.8)

    # Chronological split
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Check the result
    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Test set shape : {test_df.shape}")

    # Standardize the features using only the training set
    # This means converting to Z scores with mean 0 and stdev 1.
    scaler = StandardScaler()
    scaler.fit(train_df[vol_features])  # Fit on training data only

    print(train_df[vol_features])

    # Transform both train and full dataset using the same scaler
    train_df_scaled = scaler.transform(train_df[vol_features])
    df_vol_scaled = scaler.transform(df[vol_features])

    # Convert the scaled full dataset to a DataFrame for easier handling
    df_vol_scaled = pd.DataFrame(df_vol_scaled, index=df.index, columns=vol_features)

    pca = KernelPCA(n_components=1)
    pca.fit(train_df_scaled)

    # Apply the PCA on the whole dataset
    df["vol_pca"] = pca.transform(df_vol_scaled)

    vol_features.append("vol_pca")

    plt.figure(figsize=common.FIG_SIZE)
    a = 0.3
    # plt.plot(df_vol_scaled["vol_yang_zhang_30"], label="YZ vol 30", alpha=0.5)
    # plt.plot(df_vol_scaled["vol_rogers_satchell_60"], label="RS vol 60", alpha=0.5)
    plt.plot(df_vol_scaled["vol_std"], label="STDEV", alpha=a)
    plt.plot(df_vol_scaled["vol_ctc_30"], label="CTC vol 30", alpha=a)
    plt.plot(df_vol_scaled["vol_ctc_60"], label="CTC vol 60", alpha=a)
    plt.plot(df_vol_scaled["vol_parkinson_30"], label="Parkinson vol 30", alpha=a)
    plt.plot(df_vol_scaled["vol_parkinson_60"], label="Parkinson vol 60", alpha=a)
    plt.plot(df["vol_pca"], label="PCA vol feature")
    plt.title("PCA Over Volatility Features")
    plt.legend()
    plt.grid()
    common.savefig(plt, "PCA", strategyname)

    # df[vol_features].corr() # TODO to Latex table?

    # TODO Should df_vol_scaled be used now?

    # Entropy

    def rolling_entropy(window_data):
        # 1. Create a histogram to get frequencies.
        counts, _ = np.histogram(window_data, common.WINDOW_SIZE)

        # 2. Calculate Shannon entropy on the counts.
        return scipy.stats.entropy(counts)

    # pct_close_futur
    df["entropy_pct_close_futur"] = df["pct_close_futur"].rolling(window=common.WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Volatility
    df["vol_pct_close_futur"] = df["pct_close_futur"].rolling(window=common.WINDOW_SIZE).std()

    df["entropy_vol_std"] = df["vol_std"].rolling(window=common.WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Skewness
    df["skewness"] = df["pct_close_futur"].rolling(window=common.WINDOW_SIZE).skew()
    df["entropy_skewness"] = df["skewness"].rolling(window=common.WINDOW_SIZE) \
        .apply(rolling_entropy)

    # Plot entropy
    plt.figure()
    plt.plot(df["entropy_pct_close_futur"], label="Entropy pct_close_futur")
    plt.plot(df["entropy_vol_std"], label="Volatility")
    plt.plot(df["entropy_skewness"], label="Skewness")

    plt.legend(["Entropy of Returns",
                "Entropy of Volatility",
                "Entropy of Skewness"])
    plt.title("Entropy of Moments")
    plt.grid()
    common.savefig(plt, "entropy_moments", strategyname)

    # Corr matrix
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
