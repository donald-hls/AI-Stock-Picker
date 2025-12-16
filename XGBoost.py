from xgboost import XGBRegressor
from config import TRAIN_END_DATE, VALID_END_DATE
import pandas as pd
import numpy as np
from utility import infer_months_per_period
from sklearn.model_selection import RandomizedSearchCV

# Some Helper functions
def rmse(y_true, y_pred):
    """
    Root Mean Squared Error: measures magnitude error
    sensitive to outliers 
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def spearman_rank_corr(y_true, y_pred):
    """
    Spearman Rank Correlation: measures ranking accuracy
    """
    if len(y_true) == 0:
        return np.nan
    return pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

df = pd.read_csv("model_data.csv")
print(df.columns)
df["period_end"] = pd.to_datetime(df["period_end"])
target_col = "next_excess_ret"
months_per_period = infer_months_per_period(df, id_col="SP Identifier", date_col="period_end")

# biasedcols contain future info at t+1, excluding them from X
biasedcols = [col for col in df.columns if col.startswith("next_")]
id_cols = ["SP Identifier", "PERMNO"]
feature_cols = [col for col in df.columns if col not in biasedcols + id_cols + ["period_end"]]
df = df.dropna(subset=[target_col])

# keep numeric features only to avoid object dtype issues
X_full = df[feature_cols]
numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
# Filling the missing values with 0.0
X = X_full[numeric_cols].fillna(0.0)
y = df[target_col].values
train_mask = df["period_end"] <= pd.to_datetime(TRAIN_END_DATE)
valid_mask = (df["period_end"] > pd.to_datetime(TRAIN_END_DATE)) & (df["period_end"] <= pd.to_datetime(VALID_END_DATE))
test_mask = df["period_end"] > pd.to_datetime(VALID_END_DATE)


# Set up some fixed parameters
XGBoost_model = XGBRegressor(
    n_estimators = 3000,
    random_state = 66,
    eval_metric="rmse",
    early_stopping_rounds = 100,
    n_jobs = -1
)
# Use GridSearchCV to tune the remaining parameters
param_grid = {
    # Controls the complexity of the model
    "max_depth": [2, 3,4, 5, 6],
    # Control the learning rate (speed)
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    # Controls the percentage of firms (rows) each individual tree is allowed to see
    "subsample": [0.6, 0.7, 0.8, 0.9],
    # Controls the percentage of features(columns) each individual tree is allowed to use
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    # Regularization parameters - prevent overfitting
    "min_child_weight": [5, 10, 20],
    # L1 Regularization (Lasso) 
    "reg_alpha": [0, 0.1, 1, 10],
    # L2 Regularization (Ridge)   
    "reg_lambda": [0, 1, 10],
}

# Model Trainging
model = RandomizedSearchCV(
    estimator = XGBoost_model,
    param_distributions = param_grid,
    n_iter = 100,
    scoring = "neg_mean_squared_error",
    n_jobs = -1,
    verbose = 1,
    random_state = 66
)

# Train the XGBoost model before running validation metrics
XGBoost_model.fit(
    X[train_mask],
    y[train_mask],
    eval_set=[(X[train_mask], y[train_mask]), (X[valid_mask], y[valid_mask])],
    verbose=True,
)

XGBoost_model = XGBRegressor(
    n_estimators = 3000,
    learning_rate = 0.04,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 88,
    eval_metric="rmse",
    early_stopping_rounds = 200
)

XGBoost_model.fit(
    X[train_mask],
    y[train_mask],
    eval_set=[(X[train_mask], y[train_mask]), (X[valid_mask], y[valid_mask])],
    verbose=True,
)


# Creating Scores for Validation Set
y_valid = y[valid_mask]
valid_pred = XGBoost_model.predict(X[valid_mask])
valid_rmse = rmse(y_valid, valid_pred)
valid_spearman = spearman_rank_corr(y_valid, valid_pred)
directional_accuracy = np.mean(np.sign(valid_pred) == np.sign(y_valid))

print(f"\nValidation RMSE: {valid_rmse:.4f}")
print(f"Validation Spearman IC: {valid_spearman:.4f}")
print(f"Validation direction hit-rate: {directional_accuracy:.4f}")

# Testing Phase:
df.loc[test_mask, "predicted_next_excess_ret"] = XGBoost_model.predict(X[test_mask])

prediction_df = df.loc[
    test_mask,
    ["SP Identifier", "PERMNO", "period_end", target_col, "predicted_next_excess_ret"],
].copy()
prediction_df = prediction_df.dropna(subset=["predicted_next_excess_ret"])
# Rename the ground-truth label column to avoid confusion:
# - target_col == "next_excess_ret" is the *realized* forward excess return in the data (not a prediction)
prediction_df = prediction_df.rename(columns={target_col: "forward_excess_ret"})
# Take the most top ranked stocks
prediction_df["rank"] = prediction_df.groupby("period_end")["predicted_next_excess_ret"].rank(method="first", ascending=False)

Stocks_Top10 = prediction_df[prediction_df["rank"] <= 10].copy()
Stocks_Top10.to_csv("top10_long_only.csv", index=False)
print("--------------------------------")
print(f"\nSaved the top 10 stock selections for {Stocks_Top10['period_end'].nunique()} months.")
print("--------------------------------")

period_returns = Stocks_Top10.groupby("period_end")["forward_excess_ret"].mean().rename("portfolio_return")
# Other Stocks Return: equal-weight average forward excess return across the whole available universe each period
baseline_returns = prediction_df.groupby("period_end")["forward_excess_ret"].mean().rename("Other_Stocks_return")
performance = pd.concat([period_returns, baseline_returns], axis=1).dropna()
performance["excess_vs_baseline"] = performance["portfolio_return"] - performance["Other_Stocks_return"]



if __name__ == "__main__":
    print("-----------(Saving)------------")
    performance.to_csv("model_period_returns.csv", index=True)
    print("-----------(Performance saved)------------")
