import pandas as pd
import numpy as np 
import statsmodels.api as sm
# HAC provides robust SEs when errors are heteroskedastic and autocorrelated.
# in FM, SE assumes I.I.D Errors, but lambda_k,t typically has non-constant variance. HAC corrects for that.
from statsmodels.stats.sandwich_covariance import cov_hac
# Fama-Macbeth Regression. 
"""
* Skip the Time-Series regression step, because we are not estimating risk exposures. 
Procedure: 
    1. Run a cross-sectional regression at each time period to find the risk premium
    2. Average the risk premia (lambda_hat)
        The average of these time-series risk premia provides the final estimate
        for the risk premium associated with the factor. 
    3. Estimate the Standard Errors.
Purpose: 
    narrow down the features list -> "a filter before feeding to XGBoost" 
"""

# Helper: returns the standardized (z-score) version of the features
def zscore(group):
    """
    Takes in one month's cross-section of features, returns teh standardized version. 
    keep: a mask that indicates which features have sufficient variation.
    """
    x = group.replace([np.inf, -np.inf], np.nan) # replace inf values with NaN
    std = x.std(ddof = 0) # calculate population std (treat cross section as a population)
    keep = std.gt(1e-8)
    x = x.loc[:, keep]
    z = (x - x.mean()) / std[keep] # normalization
    return z.fillna(0), keep 

# - Step 1 in FM-Regression
def add_factor_betas(df, factor_cols=("Mkt_RF", "SMB", "HML"), window=36, min_periods=24, id_col="SP Identifier", time_col="month", target_col="excess_ret"):
    """
    Estimate time-series regressions of the firm's excess returns on factor returns using trailing window (avoid look-ahead bias)
    The calculated betas are firm-level exposures at time t (using info until t-1)
    rolling factor betas for each firm using trailing window OLS. 
    """

    df = df.sort_values([id_col, time_col]).copy()
    # prepares output col_names like beta_mkt_rf, beta_smb, beta_hml
    beta_cols = [f"beta_{col.lower().replace('-', '_')}" for col in factor_cols]
    # creates a empty df to hold the betas
    betas = pd.DataFrame(index=df.index, columns=beta_cols, dtype=float)

    for firm_id, firm_panel in df.groupby(id_col):
        firm_panel = firm_panel.sort_values(time_col)
        firm_index = firm_panel.index.to_list()
        # looping over months within a firm
        for pos, idx in enumerate(firm_index):
            end = pos
            if end == 0:
                continue
            start = max(0, end - window)
            window_idx = firm_index[start:end]
            # extract excess return and the factor columns 
            window_frame = firm_panel.loc[window_idx, [target_col, *factor_cols]].dropna()
            if window_frame.shape[0] < min_periods:
                continue
            # vector of the firm's excess returns over the window
            y_window = window_frame[target_col].astype(float).to_numpy()
            # matrix of factor realizations for the months
            X_window = window_frame[list(factor_cols)].astype(float).to_numpy()
            # creating the intercept (col of 1s)
            X_design = np.column_stack([np.ones(len(y_window)), X_window])
            # beta_vector[0] is teh intercept. 
            beta_vector, *_ = np.linalg.lstsq(X_design, y_window, rcond=None)

            for j, factor in enumerate(factor_cols, start = 1):
                col_name = f"beta_{factor.lower().replace('-', '_')}"
                betas.at[idx, col_name] = beta_vector[j]

    for col in beta_cols:
        df[col] = betas[col]

    return df

# Step 2 
def merge_data(stock_data, ff_data):
    """
    merge_data merges the windsorized data and fama_french_factors data.
    the function merges the data on the month column.
    merge_data(windsorized_data, fama_french_factors): DataFrame, DataFrame -> DataFrame.
    """
    windsorized_data = pd.read_csv(stock_data, parse_dates=["month"])
    fama_french_factors = pd.read_csv(ff_data, parse_dates=["month"])
    fama_french_factors = fama_french_factors.drop(columns=["Unnamed: 0"], errors="ignore") # drop the unnamed column

    merged = pd.merge(windsorized_data, fama_french_factors, on="month", how="left")
    merged = merged.sort_values(by=["SP Identifier", "month"])
    # use .shift(-1) to get the next month's return
    merged["next_mon_ret"] = merged.groupby("SP Identifier")["Monthly Total Return"].shift(-1)
    # to calculate the target: the next month's excess return
    rf_by_month = merged[["month", "Risk Free"]].drop_duplicates().sort_values("month")
    rf_by_month["next_rf"] = rf_by_month["Risk Free"].shift(-1)
    next_rf_map = rf_by_month.set_index("month")["next_rf"]
    merged["next_rf_ret"] = merged["month"].map(next_rf_map)
    merged["next_excess_ret"] = merged["next_mon_ret"] - merged["next_rf_ret"]
    
    # contemporaneous excess return for beta estimation
    merged["excess_ret"] = merged["Monthly Total Return"] - merged["Risk Free"]
    merged = add_factor_betas(merged)

    # drop rows where the essential targets are NaN
    model_data = merged.dropna(subset=["next_excess_ret"]).copy()
    model_data.to_csv("model_data.csv")
    return model_data

merged_data = merge_data("windsorized_data.csv", "fama_french_factors.csv")


def fama_macbeth(df, feature_columns, target_col="next_excess_ret", id_col="SP Identifier", time_col="month", hac_lags=6):
    """
    Fama-Macbeth Regression for panel data.
    Runs a cross-sectional regression at each time period to find the risk premium
    """
    df = df.sort_values([time_col, id_col]).copy()
    feature_columns = [col for col in feature_columns if col in df.columns]
    # to store each month's cross-sectional regression for each month
    lambda_container = []
    # track the features that are kept after standardization
    kept_features = set()
    # group the data by month, each group represents one month's cross-section
    # runs once per unique month, each group is a subset of the df containing only the rows for that month
    for t, group in df.groupby(time_col, sort=True):
        group = group.dropna(subset=[target_col])
        X_z, keep = zscore(group[feature_columns])
        if X_z.empty:
            continue
        # skip months with too few observations relative to usable regressors
        if len(group) < X_z.shape[1] + 3:
            continue
        X_z = X_z.astype(float)
        # adds a col of ones to a dataset, 
        X = sm.add_constant(X_z)
        # y: excess return next month
        y = group[target_col].astype(float)
        # Fit OLS Regression: r_{i,t+1} = λ0_t + Σ_k λ_{k,t} X_{i,k,t} + ε_{i,t+1}
        beta = sm.OLS(y, X).fit().params.rename(t)
        # Save estimated coefficients (λ_t) for this month, labeling them by time (t)
        lambda_container.append(beta)
        kept_features.update(X_z.columns)
    # combine all lambda_t estimates into one df
    lambdas = pd.DataFrame(lambda_container)
    feature_order = ["const"] + sorted(kept_features)
    lambdas = lambdas.reindex(columns=feature_order)
    # Time average the factor premia
    # lambda_hat k = average of lambda_{k, t} across time
    average_lambda = lambdas.mean()
    # t-statistics: 
        # large t: the factor has a statistically relationship with returns
        # small t: the estimated premium could be due to other factors
    t_statistics = {}
    for col in lambdas.columns:
        series = lambdas[col].dropna()
        if len(series) < 12:
            t_statistics[col] = np.nan
            continue
        # OLS regression to estimate the factor premium
        reg = sm.OLS(series.values, np.ones((len(series), 1))).fit(
            cov_type = "HAC", cov_kwds = {'maxlags': hac_lags}
        )
        t_statistics[col] = reg.tvalues[0] # store the t-statistic for the factor premium
    results = pd.DataFrame({
        "lambda_mean": average_lambda, 
        "t_stat": pd.Series(t_statistics)
    }).drop("const", errors= "ignore") # drop the intercept row
    # construct fama-macbeth linear singal for each observation
    df["FM_signal"] = np.nan
    fm_features = [col for col in feature_order if col != "const"]
    available_lambda = average_lambda.reindex(fm_features).fillna(0.0)
    for t, group in df.groupby(time_col, sort=True):
        X_z, _ = zscore(group[fm_features])
        if X_z.empty:
            continue
        X_z = X_z.reindex(columns=fm_features, fill_value=0.0).astype(float)
        df.loc[group.index, "FM_signal"] = X_z.to_numpy().dot(available_lambda.to_numpy())
    
    # estimated the stock's expected return 
    # FM_signals:
        # Positive: the stock has exposures associated with higher expected returns
        # Negative: the stock has exposures associated with lower expected returns 
    # results: a df with the factor premia and t-statistics
    # df: the original df with the FM_signal column
    # lambdas: a df with the factor premia for each month
    return results, df, lambdas

# stock level features: 
feature_lst = [
    "PE", "PB", "PS", "OperatingMargin", "EbitdaMargin",
    "DebtToEquity", "DebtToAssets", "IntCoverage", "CurrentRatio",
    "Sales_YoY", "EPS_YoY", "momentum_12_1", "momentum_6",
    "momentum_3", "rev_1m", "vol_12m",
    "beta_mkt_rf", "beta_smb", "beta_hml"
]
fm_results, fm_scored, lambda_table = fama_macbeth(merged_data, feature_lst)

if __name__ == "__main__":
    print("Fama-Macbeth Regression Results:")
    print(fm_results)
    print("\nFama-Macbeth Scored Data:")
    print(fm_scored)
    print("\nFama-Macbeth Lambda Table:")
    print(lambda_table)
    print("saving the results to a csv file")
    fm_scored.to_csv("model_data_scored.csv", index=False)
    print("data saved...")