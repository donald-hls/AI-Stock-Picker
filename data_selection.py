import pandas as pd 
import numpy as np 

"""
Units of the CRSP Variables:
    (fyear) Fiscal data year — year number.
    (fyr) Fiscal year-end month — 1 – 12.
    (epspx) EPS (Basic, excl. extraordinary) — USD/share. 
    (csho) Common shares outstanding — millions of shares (used with PRCC_* to compute market cap in $mm). 
    (ceq) Common equity, total — USD millions. 
    (ebitda) EBITDA — USD millions. 
    (ebit) EBIT — USD millions. 
    (dltt) Long-term debt, total — USD millions. 
    (dlc) Debt in current liabilities — USD millions. 
    (che) Cash & short-term investments — USD millions. 
    (dvpsx_f) Dividends per share (ex-date, fiscal) — USD/share. 
    (dvt) Dividends, total — USD millions. 
    (sale) Sales/Turnover (net) — USD millions (explicitly stated). 
    (cogs) Cost of goods sold — USD millions (explicit note). 
    (oiadp) Operating income after depreciation — USD millions. 
    (ni) Net income (loss) — USD millions. 
    (lt) Liabilities, total — USD millions. 
    (at) Assets, total — USD millions. 
    (oancf) Net cash flow from operating activities — USD millions. 
    (capx) Capital expenditures — USD millions. 
    (act) Current assets — USD millions. 
    (lct) Current liabilities — USD millions. 
    (xint) Interest and related expense — USD millions. 
    (prcc_f) Price close (annual, fiscal) — USD/share. 
    (prcc_c) Price close (annual, calendar) — USD/share. 
    (yyyymm) Month key — YYYYMM integer.
    (mthprc) Month-end price — USD/share. 
    (mthret) Monthly total return (with dividends) — decimal (e.g., 0.05 = 5%). 
    (mthretx) Monthly return without dividends — decimal. 
    (shrout) Shares outstanding — thousands of shares (CRSP convention).
"""

def ratio_calculation(S1, S2):
    """
        Helper function to calculate the ratio of 2 Sereis Objects,
        used for data cleaning functions.
    
    ratio_calculation(S1, S2) -> Series
    
    """
    ratio = S1 / S2
    # if not finite 
    cleaned = ratio.mask(~np.isfinite(ratio), np.nan)
    return cleaned


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
    column_name_mapping = {
        "gvkey": "SP Identifier",
        "PERMNO": "PERMNO",
        "fyear": "Fiscal Year",
        "MthCalDt": "Monthly Calendar Date",
        "MthRet": "Monthly Total Return",
        "MthRetx": "Monthly Total Return Excluding Dividends",
        "conm": "company_name",
        "epspx": "EPS",
        "sale": "Sales",
        "ebit": "EBIT",
        "cogs": "COGS",
        "oiadp": "Operating Income After Depreciation",
        "ebitda": "EBITDA",
        "capx": "Capital Expenditures",
        "xint": "Interest Related Expense",
        "at": "Total Assets",
        "lt": "Total Liabilities",
        "act": "Current Assets",
        "lct": "Current Liabilities",
        "che": "Cash and Short-Term Investments",
        "dlc": "Debt in Current Liabilities",
        "dltt": "Debt in Long-Term Debt",
        "ceq": "Common Equity",
        "ni": "Net Income",
        "dvpsx_f": "Dividends per Share",
        "ShrOut": "Shares Outstanding (CRSP)",
        "MthPrc": "Monthly Price"
    }
    # Rename the columns based on the mapping
    df.rename(columns = column_name_mapping, inplace=True)

    # Select the columns based on the mapping and makes a copy
    df = df[[val for val in column_name_mapping.values()]].copy()
    # Convert to Datetime objects 
    df["Monthly Calendar Date"] = pd.to_datetime(df["Monthly Calendar Date"], errors="coerce")
    
    # STANDARDIZE ALL FINANCIAL VALUES TO DOLLARS (not millions)
    # Convert all financial statement data from millions to actual dollars
    financial_vars = ["Total Assets", "Total Liabilities", "Sales", "EBIT", "COGS", 
                     "Operating Income After Depreciation", "EBITDA", "Capital Expenditures",
                     "Interest Related Expense", "Current Assets", "Current Liabilities", 
                     "Cash and Short-Term Investments", "Debt in Current Liabilities",
                     "Debt in Long-Term Debt", "Common Equity", "Net Income"]
    
    for item in financial_vars:
            df[item] = df[item] * 1_000_000  # Convert millions to dollars
    # Using Monthly Data 
        # Why ? 
            # 1. More data points, lower estimation error 
            # 2. Better covariance / factor signals - Volatility clusters and correlations move through the year.
            # 3. Monthly returns capture regime shifts and co-movements that annual aggregation washes out.
    # Month key
    df["month"] = df["Monthly Calendar Date"].dt.to_period("M").dt.to_timestamp("M")

    # Build a fundamentals frame (annual) 
    # Build a prices frame (monthly)
    fundamental_cols = ["SP Identifier","Fiscal Year","company_name","EPS","Sales","EBIT","COGS",
                 "Operating Income After Depreciation","EBITDA","Capital Expenditures",
                 "Interest Related Expense","Total Assets","Total Liabilities","Current Assets",
                 "Current Liabilities","Cash and Short-Term Investments","Debt in Current Liabilities",
                 "Debt in Long-Term Debt","Common Equity","Net Income","Dividends per Share"]
    # Using the fundamental_cols, drop duplicates based on the SP Identifier and Fiscal Year
    fund = df.drop_duplicates(subset=["SP Identifier","Fiscal Year"])[[col for col in fundamental_cols if col in df.columns]].copy()

    # Create a 4-month publication lag window per fiscal year
    # Assume FY ends in calendar year Fiscal Year with FY-end = Dec 31 of that year (adjust if you have FYR).
    fund["fiscalYear_end"] = pd.to_datetime(fund["Fiscal Year"].astype(str) + "-12-31")
    # Use them only when the market could actually "know" them
    # avail_from: the first month-end when a given fiscal-year snapshot is considered public/usable.
    # this is to block "look ahead bias"
    fund["avail_from"] = (fund["fiscalYear_end"] + pd.offsets.MonthEnd(4))  # FY + 4 months
    # avail_to: the last day this snapshot remains the “latest available” before it’s superseded.
    fund["avail_to"]   = fund.groupby("SP Identifier")["avail_from"].shift(-1) - pd.Timedelta(days=1)
    fund["avail_to"]   = fund["avail_to"].fillna(pd.Timestamp("2200-12-31")) #set a far date to avoid any issues. 
    
    # Monthly prices/returns frame
    px_cols = ["SP Identifier","PERMNO","month","Monthly Price","Monthly Total Return",
               "Monthly Total Return Excluding Dividends","Shares Outstanding (CRSP)"]
    # Filter & Drop duplicates based on the PERMNO and month
    px = df.dropna(subset=["Monthly Price","Shares Outstanding (CRSP)"])[px_cols].drop_duplicates(subset=["PERMNO","month"])
    # Join fundamentals to each month using the availability window
    # Remove any potential duplicate columns before merge
    fundamental_cols_clean = [col for col in fundamental_cols if col in fund.columns and col not in ["SP Identifier"]]
    # Keep the timing window and the fundamental cols
    fund_long = fund[["SP Identifier", "avail_from","avail_to"] + fundamental_cols_clean].copy()

    merged = px.merge(fund_long, on = "SP Identifier", how = "left")
    # Filter by Month within the timing window 
    mask = (merged["month"] >= merged["avail_from"]) & (merged["month"] <= merged["avail_to"])
    merged = merged[mask].copy() # Make a copy of the filtered dataframe
    
    # Calculate the Monthly Market Cap
    # Use abs because CRSP sometimes stores negative prices as a convention.
    merged["Monthly Market Cap"] = merged["Monthly Price"].abs() * merged["Shares Outstanding (CRSP)"] * 1000
    # Calculate some financial ratios (Monthly)
    
    # P/E: Price/Earnings: Market Cap / Net Income
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

    # Print a short summary: 
    print(f"Monthly dataset created: {len(merged):,} observations")
    # Print the date range
    print(f"Date range: {merged['month'].min()} to {merged['month'].max()}")
    # Print the number of companies
    print(f"Number of companies: {merged['SP Identifier'].nunique():,}")
    
    # save the dataframe to an excel file to check data integrity
    merged.to_excel("monthly_data.xlsx", index = False)
    # Return the merged dataframe
    return merged

monthly_data = clean_data("data.csv")

print(monthly_data.head())

print(monthly_data.columns)