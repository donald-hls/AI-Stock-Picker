import pandas as pd 
import numpy as np 
# MonthEnd to target the end of a Month.
from pandas.tseries.offsets import MonthEnd
from pandas_datareader import data as pdr
from utility import ratio_calculation, limiting_extreme_values, rolling_total_return
from config import (
    COLUMN_NAME_MAPPING,
    FINANCIAL_VARS_IN_MILLIONS,
    FUNDAMENTAL_COLS,
    PRICE_COLS,
    PUBLICATION_LAG_MONTHS,
    FAR_FUTURE_DATE,
    MILLION_TO_DOLLARS,
    THOUSAND_TO_ACTUAL,
)

def clean_data(csv_file):
    """
    clean_data reads in a csv file and cleans the data by
    removing/filling in the missing values and converting the data to the 
    correct format for the model. 
    The function returns a pandas dataframe with selected columns. 
    The data is obtained from Wharton Research Data Services. 
    clean_data(csv_file): CSV_File -> DataFrame
    """
    df = pd.read_csv(csv_file)
    # Rename the columns based on the mapping
    df.rename(columns= COLUMN_NAME_MAPPING, inplace=True)
    # Select the columns based on the mapping and makes a copy
    df = df[[val for val in COLUMN_NAME_MAPPING.values()]].copy()
    # Convert to Datetime objects 
    df["Monthly Calendar Date"] = pd.to_datetime(df["Monthly Calendar Date"], errors="coerce")
    # STANDARDIZE ALL FINANCIAL VALUES TO DOLLARS (not millions)
    # Convert all financial statement data from millions to actual dollars
    for item in FINANCIAL_VARS_IN_MILLIONS:
        df[item] = df[item] * MILLION_TO_DOLLARS
    # marks the month-end date for each observation. - helps sorting
    df["period_end"] = df["Monthly Calendar Date"].dt.to_period("M").dt.to_timestamp("M")
    fund = df.drop_duplicates(subset=["SP Identifier","Fiscal Year"])[[col for col in FUNDAMENTAL_COLS if col in df.columns]].copy()
    fund["fiscalYear_end"] = pd.to_datetime(fund["Fiscal Year"].astype(str) + "-12-31")
    # Create a 4-month publication lag window per fiscal year
    fund["avail_from"] = (fund["fiscalYear_end"] + pd.offsets.MonthEnd(PUBLICATION_LAG_MONTHS))
    # avail_to: the last day this snapshot remains the "latest available" before it's superseded.
    fund["avail_to"] = fund.groupby("SP Identifier")["avail_from"].shift(-1) - pd.Timedelta(days=1)
    fund["avail_to"] = fund["avail_to"].fillna(pd.Timestamp(FAR_FUTURE_DATE)) 
    # Filter & Drop duplicates
    px = df.dropna(subset=["Monthly Price","Shares Outstanding (CRSP)"])[PRICE_COLS].drop_duplicates(subset=["PERMNO","period_end"])
    # Join fundamentals to each month using the availability window 
    fundamental_cols_clean = [col for col in FUNDAMENTAL_COLS if col in fund.columns and col not in ["SP Identifier"]]
    # Keep the timing window and the fundamental cols
    fund_long = fund[["SP Identifier", "avail_from","avail_to"] + fundamental_cols_clean].copy()
    merged = px.merge(fund_long, on = "SP Identifier", how = "left")
    # Filter by Month within the timing window 
    mask = (merged["period_end"] >= merged["avail_from"]) & (merged["period_end"] <= merged["avail_to"])
    merged = merged[mask].copy()
    merged["Monthly Market Cap"] = merged["Monthly Price"].abs() * merged["Shares Outstanding (CRSP)"] * THOUSAND_TO_ACTUAL
    merged = merged.sort_values(["SP Identifier", "period_end"])
    merged["ret_12m"] = merged.groupby("SP Identifier")["Monthly Price"].pct_change(1)
    # Some ratio calculations:
    # Price to Earnings Ratio: Market Cap / Net Income
    merged["P/E"] = ratio_calculation(merged["Monthly Market Cap"], merged["Net Income"])
    # P/B: Price/Book Value: Market Cap / Book Equity
    merged["P/B"] = ratio_calculation(merged["Monthly Market Cap"], merged["Common Equity"])
    # P/S: Price/Sales: Market Cap / Sales
    merged["P/S"] = ratio_calculation(merged["Monthly Market Cap"], merged["Sales"])
    # ROE: Return on Equity: Net Income / Common Equity
    merged["ROE"] = ratio_calculation(merged["Net Income"], merged["Common Equity"])
    # ROA: Return on Assets
    merged["ROA"] = ratio_calculation(merged["Net Income"], merged["Total Assets"])
    # Operating Margin: Operating Income After Depreciation / Sales
    merged["Operating Margin"] = ratio_calculation(merged["Operating Income After Depreciation"], merged["Sales"])
    # EBITDA Margin: EBITDA / Sales
    merged["EBITDA Margin"] = ratio_calculation(merged["EBITDA"], merged["Sales"])
    # Debt Total: Debt in Current Liabilities + Debt in Long-Term Debt
    merged["Debt Total"] = merged["Debt in Current Liabilities"] + merged["Debt in Long-Term Debt"]
    # Debt/Equity: Debt Total / Common Equity
    merged["Debt/Equity"] = ratio_calculation(merged["Debt Total"], merged["Common Equity"])
    # Debt/Assets: Debt Total / Total Assets
    merged["Debt/Assets"] = ratio_calculation(merged["Debt Total"], merged["Total Assets"])
    # Interest Coverage: EBIT / Interest Related Expense
    merged["Interest Coverage"] = ratio_calculation(merged["EBIT"], merged["Interest Related Expense"])
    # Current Ratio: Current Assets / Current Liabilities
    merged["Current Ratio"] = ratio_calculation(merged["Current Assets"], merged["Current Liabilities"])
    return merged

annual_data = clean_data("new_data.csv")

def get_macro_data(start_date="1900-01-01", end_date="2025-10-01"):
    """
    Fetch monthly macroeconomic series from FRED.
    Returns a DataFrame of VIX, Treasury yields, unemployment, and IG OAS resampled to month-end.
    """
    # VIX
    VIX = pdr.get_data_fred("VIXCLS", start=start_date, end=end_date)
    # 10Y Treasury Yield
    Yield10 = pdr.get_data_fred("DGS10", start=start_date, end=end_date)
    # 3M Treasury Yield
    Yield3M = pdr.get_data_fred("TB3MS", start=start_date, end=end_date)
    # Unemployment Rate
    unemploy = pdr.get_data_fred("UNRATE", start=start_date, end=end_date)
    # IG OAS
    ig_oas = pdr.get_data_fred("BAMLC0A0CM", start=start_date, end=end_date)["BAMLC0A0CM"]
    monthly_data = pd.DataFrame({
            "VIX": VIX["VIXCLS"].resample("ME").last(),
            "Yield10": Yield10["DGS10"].resample("ME").last(),
            "Yield3M": Yield3M["TB3MS"].resample("ME").last(),
            "unemploy": unemploy["UNRATE"].resample("ME").last(),
            "IG_OAS": ig_oas.resample("ME").last(),
        }).dropna(how="all")
    monthly_data = monthly_data.reset_index().rename(columns={"DATE": "period_end"})
    monthly_data["3Mon-10Y"] = monthly_data["Yield10"] - monthly_data["Yield3M"]
    monthly_data["Δ in Unemploy"] = monthly_data["unemploy"].diff()
    return monthly_data
        
macro_data = get_macro_data(start_date= "2001-04-30", end_date= "2024-12-31")

def add_macro_features(ann_data, macro_data, lag_periods = 1):
    
    """
    add_macro_features adds macroeconomic features to the monthly data.
    The function returns a pandas dataframe with the macroeconomic features.
    
    By default, the function sets the lag_month = 1. (not knowing the economic data until the next month)
    """
    macro_data_copy = macro_data.sort_values(by ="period_end").copy()
    # Apply time shift to macro variables (not including Month column)
    for col in macro_data_copy.columns:
        if col != "period_end":
            macro_data_copy[col] = macro_data_copy[col].shift(lag_periods)
    return ann_data.merge(macro_data_copy, on="period_end", how="left")
    
retval = add_macro_features(annual_data, macro_data)

def compute_features(monthly_df):
    """
    Compute some new features for the monthly data, and puts a lag on the 
    newly added features such as YoY growth, momentum, volatility, and beta.
    compute_features(monthly_df): DataFrame -> DataFrame, list
    """
    monthly_df = monthly_df.sort_values(by=["SP Identifier", "period_end"]).copy()
    # Rename Mapping rule 
    RENAME = {
        "P/E":"PE","P/B":"PB","P/S":"PS",
        "Operating Margin":"OperatingMargin","EBITDA Margin":"EbitdaMargin",
        "Debt/Equity":"DebtToEquity","Debt/Assets":"DebtToAssets",
        "Interest Coverage":"IntCoverage","Current Ratio":"CurrentRatio",
        "Monthly Market Cap":"mcap"
    }
    for key, val in RENAME.items():
        if key in monthly_df.columns:
            monthly_df.rename(columns={key: val}, inplace = True)
    
    # forward fill annual fundamentals with availability window
    for col in ["Sales","Net Income","Operating Income After Depreciation",
                "EBITDA","Common Equity","Total Assets","EPS","COGS"]:
        if col in monthly_df.columns:
            monthly_df[col + "_ff"] = monthly_df.groupby("SP Identifier")[col].ffill()
            
    # Year over Year Growth (YOY)
    monthly_df["Sales_YoY"] = monthly_df.groupby("SP Identifier")["Sales_ff"].pct_change(1, fill_method=None)
    monthly_df["EPS_YoY"] = monthly_df.groupby("SP Identifier")["EPS_ff"].pct_change(1, fill_method=None)

    # Momentum / reversal and risk features derived from annual returns
    ret_12m_grouped = monthly_df.groupby("SP Identifier")["ret_12m"]
    # Buils annual momentum and volatility
    monthly_df["return_1y"] = monthly_df["ret_12m"]
    monthly_df["rev_1y"] = -monthly_df["ret_12m"]
    monthly_df["momentum_2y"] = ret_12m_grouped.transform(lambda s: rolling_total_return(s, window=2, min_periods=2))
    monthly_df["momentum_3y"] = ret_12m_grouped.transform(lambda s: rolling_total_return(s, window=3, min_periods=2))
    monthly_df["vol_3y"] = ret_12m_grouped.transform(lambda s: s.rolling(window=3, min_periods=2).std())
    monthly_df["vol_5y"] = ret_12m_grouped.transform(lambda s: s.rolling(window=5, min_periods=3).std())
    # put lag on the newly added features. 
    accounting_cols = ["PE","PB","PS","OperatingMargin","EbitdaMargin","DebtToEquity","DebtToAssets","IntCoverage","CurrentRatio"]
    calculated_cols = ["Sales_YoY","EPS_YoY","return_1y","rev_1y","momentum_2y","momentum_3y","vol_3y","vol_5y"]
    combined_cols = accounting_cols + calculated_cols
    
    for col in combined_cols:
        if col in monthly_df.columns:
            monthly_df[col] = monthly_df.groupby("SP Identifier")[col].shift(1)
            
    FEATURE_COLS = [c for c in accounting_cols + calculated_cols if c in monthly_df.columns]
    
    return monthly_df, FEATURE_COLS

computed_data, feature_cols = compute_features(retval)


def cross_sectional_windsorize(monthly_df, feature_cols):
    """
    cross_sectional_windsorize windsorizes the data across all companies in a given month.
    the function calls the helper function limiting_extreme_values to windsorize the data in Series. 
    
    cross_sectional_windsorize(monthly_df, feature_cols): DataFrame, list -> DataFrame
    """
    retval = monthly_df.copy()
    # partitions rows by fiscal period  
    for period, idx in retval.groupby("period_end").groups.items():
        block = retval.loc[idx, feature_cols]
        blockWindsorized = block.apply(limiting_extreme_values, axis = 0)
        # normalize the data
        retval.loc[idx, feature_cols] = (blockWindsorized - blockWindsorized.mean()) / blockWindsorized.std()
    return retval 
    
windsorized_data = cross_sectional_windsorize(computed_data, feature_cols)
windsorized_data.to_csv("windsorized_data.csv", index = False)

if __name__ == "__main__":
    # Print a short summary: 
    print(f"Annual dataset created: {len(annual_data):,} observations")
    # Print the date range
    print(f"Date range: {annual_data['period_end'].min()} to {annual_data['period_end'].max()}")
    # Print the number of companies
    print(f"Number of companies: {annual_data['SP Identifier'].nunique():,}")
    # save the dataframe to an excel file to check data integrity
    # monthly_data.to_excel("monthly_data.xlsx", index = False)
    
    # Additional dataset coverage info
    print("=== DATA RANGES ===")
    print(f"Prices/Fundamentals (annual_data): {annual_data['period_end'].min().date()} → {annual_data['period_end'].max().date()} ({len(annual_data):,} rows)")
    print(f"Macro (macro_data): {macro_data['period_end'].min().date()} → {macro_data['period_end'].max().date()} ({len(macro_data):,} rows)")
    print(f"Windsorized (windsorized_data): {windsorized_data['period_end'].min().date()} → {windsorized_data['period_end'].max().date()} ({len(windsorized_data):,} rows)")
