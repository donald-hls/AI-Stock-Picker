import pandas as pd
import numpy as np
import statsmodels.api as sm
from utility import zscore, rolling_total_return
from statsmodels.stats.sandwich_covariance import cov_hac

# To take away the outliers -> 98%
TARGET_CLIP_BOUNDS = (-2.0, 2.0)

# Step 1: rolling betas (time-series regressions per firm)
def add_factor_betas(df):
    """
    add_factor_betas estimates trailing OLS betas using annual information up to t-1 (no look-ahead).
    The function returns a DataFrame with the factor betas.
    """
    # Systematic Risk Factors 
    factor_cols=("Mkt_RF_ann", "SMB_ann", "HML_ann")
    windows = 5
    min_periods = 3
    # id_col = "SP Identifier"
    # time_col = "period_end"
    target_col = "excess_ret"
    df = df.sort_values(["SP Identifier", "period_end"]).copy()
    # Create the beta columsn - "renaming the factor columns"
    beta_cols = [f"beta_{col.lower().replace('-', '_')}" for col in factor_cols]
    betas = pd.DataFrame(index = df.index, columns=beta_cols)
    for firm_id, firm_panel in df.groupby("SP Identifier"):
        # Time Series of observations 
        firm_panel = firm_panel.sort_values("period_end") # all rows for a given firm sorted by period_end
        # list of row indices
        firm_index = firm_panel.index.to_list() # list of DataFrame index values 
        
        for pos, idx in enumerate(firm_index):
            end = pos
            # no past observations available
            if end == 0:
                continue
            # start index of the window
            start = max(0, end - windows)
            # list of row indices for the window
            window_idx = firm_index[start:end] # get the last 5 
            # DataFrame of the window: 
                # excess returns(target_col) and factors (factor_cols)
            window_frame = firm_panel.loc[window_idx, [target_col, *factor_cols]].dropna()
            if window_frame.shape[0] < min_periods:
                continue
            # Run OLS Regression 
            # vector of excess returns 
            y_window = window_frame[target_col].astype(float).to_numpy()
            # matrix of factor returns 
            X_window = window_frame[list(factor_cols)].astype(float).to_numpy()
            # design matrix: add a column of ones for the intercept
            X_design = np.column_stack([np.ones(len(y_window)), X_window])
            # returns the least-squares solution
            # beta_vector[0] = intercept
            # beta_vector[1] = beta on Market Risk Factor
            # beta_vector[2] = beta on SMB
            # beta_vector[3] = beta on HML
            beta_vector, *_ = np.linalg.lstsq(X_design, y_window, rcond=None)
            for j, factor in enumerate(factor_cols, start=1):
                col_name = f"beta_{factor.lower().replace('-', '_')}"
                betas.at[idx, col_name] = beta_vector[j]

    for col in beta_cols:
        df[col] = betas[col]

    return df


def merge_data(stock_data_path, ff_data_path):
    """
    Merge windsorized characteristics with Fama-French factors on period_end.
    Builds next-period excess returns and contemporaneous excess returns for betas.
    """
    # data from data_selection.py
    windsorized = pd.read_csv(stock_data_path, parse_dates=["period_end"]).sort_values(["SP Identifier", "period_end"])
    # data from Fama-French
    fama_french = pd.read_csv(ff_data_path, parse_dates=["period_end"])
    fama_french = fama_french.drop(columns=["Unnamed: 0"], errors = "ignore")
    merged = pd.merge(windsorized, fama_french, on="period_end", how = "left")
    merged = merged.sort_values(by=["SP Identifier", "period_end"])
    # Future returns: Used to evaluate the performance of the model -> XGBoost
    merged["next_ret_12m"] = (
        merged.groupby("SP Identifier")["ret_1m"].transform(
            lambda s: rolling_total_return(s.shift(-1), 12, 12)
        )
    )
    merged["next_ret_12m"] = merged["next_ret_12m"].clip(*TARGET_CLIP_BOUNDS)
    merged["next_excess_ret"] = merged["next_ret_12m"] - merged["next_rf_12m"]
    # Uses trailing 12 month: feed to add_factor_betas
    merged["excess_ret"] = merged["ret_12m"] - merged["Risk Free"]
    merged = add_factor_betas(merged)
    # Rename the columns:
    merged.rename(columns = {"beta_mkt_rf_ann": "beta_mkt_rf", 
                             "beta_smb_ann": "beta_smb", 
                             "beta_hml_ann": "beta_hml"}, inplace=True)
    model_data = merged.dropna(subset=["next_excess_ret"]).copy()
    model_data.to_csv("model_data.csv", index=False)
    return model_data

merged_data = merge_data("windsorized_data.csv", "fama_french_factors.csv")

# Step 2: cross-sectional regressions and averaging

def fama_macbeth(df, feature_columns, hac_lags=3):
    """
    fama_macbeth runs the classic two-step Fama–MacBeth procedure on the panel.
    The function returns a DataFrame with the Fama-MacBeth results, a DataFrame with the scored data, and a DataFrame with the lambda table.
    """
    target_col="next_excess_ret"
    
    df = df.sort_values(["period_end", "SP Identifier"]).copy()
    feature_columns = [col for col in feature_columns if col in df.columns]
    # containing the list of per period lambda
    lambda_container = []
    kept_features = set()

    for t, group in df.groupby("period_end", sort=True):
        group = group.dropna(subset=[target_col])
        X_z, keep = zscore(group[feature_columns])
        # Helper to standardize the features
        if X_z.empty:
            continue
        # Not enough features to run the regression
        if len(group) < X_z.shape[1] + 3:
            continue
        # Design matrix: add a column of ones for the intercept
        X = sm.add_constant(X_z.astype(float)) # casts to float
        # vector of excess returns 
        y = group[target_col].astype(float) #casts to float 
        # Running Cross-Sectional Regression
        beta = sm.OLS(y, X).fit().params.rename(t)
        lambda_container.append(beta)
        kept_features.update(X_z.columns)

    lambdas = pd.DataFrame(lambda_container)
    feature_order = ["const"] + sorted(kept_features)
    lambdas = lambdas.reindex(columns=feature_order)

    average_lambda = lambdas.mean()
    # HAC t-statistics
    t_statistics = {}
    for col in lambdas.columns:
        series = lambdas[col].dropna()
        if len(series) < 8:
            t_statistics[col] = np.nan
            continue
        # Run a regression on a constant to get the mean
        # Use HAC to account for autocorrelation and heteroskedasticity
        reg = sm.OLS(series.values, np.ones((len(series), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        t_statistics[col] = reg.tvalues[0]

    results = pd.DataFrame({"lambda_mean": average_lambda, "t_stat": pd.Series(t_statistics)}).drop("const", errors="ignore")
    # Build the score board.
    df["FM_signal"] = np.nan
    fm_features = [col for col in feature_order if col != "const"]
    available_lambda = average_lambda.reindex(fm_features).fillna(0.0)

    for t, group in df.groupby("period_end", sort=True):
        X_z, _ = zscore(group[fm_features])
        if X_z.empty:
            continue
        # fill in the missing features with 0.0
        X_z = X_z.reindex(columns=fm_features, fill_value=0.0).astype(float)
        df.loc[group.index, "FM_signal"] = X_z.to_numpy().dot(available_lambda.to_numpy())

    print(df.columns)
    return results, df, lambdas

# Feature set (characteristics + factor betas)
# candidate of pricing factors
feature_lst = [
    "PE",
    "PB",
    "PS",
    "OperatingMargin",
    "EbitdaMargin",
    "DebtToEquity",
    "DebtToAssets",
    "IntCoverage",
    "CurrentRatio",
    "Sales_YoY",
    "EPS_YoY",
    "return_1y",
    "rev_1y",
    "momentum_2y",
    "momentum_3y",
    "vol_3y",
    "vol_5y",
    "beta_mkt_rf",
    "beta_smb",
    "beta_hml",
]

merged_data = merge_data("windsorized_data.csv", "fama_french_factors.csv")
fm_results, fm_scored, lambda_table = fama_macbeth(merged_data, feature_lst)


if __name__ == "__main__":
    print("Fama-MacBeth Regression Results:")
    print(fm_results.head())

    print("\nFama-MacBeth Scored Data:")
    print(fm_scored.head())

    print("\nFama-MacBeth Lambda Table:")
    print(lambda_table.head())

    print("saving the results to a csv file")
    fm_scored.to_csv("model_data_scored.csv", index=False)
    print("data saved...")
