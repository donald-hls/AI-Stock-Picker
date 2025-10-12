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
        "prcc_f": "Annual Price Close (Fiscal)",
        "ShrOut": "Shares Outstanding (CRSP)"
    }
    # Rename the columns based on the mapping
    df.rename(columns = column_name_mapping, inplace=True)
    # Select the columns based on the mapping
    df = df[[val for val in column_name_mapping.values()]]
    # STANDARDIZE ALL FINANCIAL VALUES TO DOLLARS (not millions)
    # Convert all financial statement items from millions to actual dollars
    financial_vars = ["Total Assets", "Total Liabilities", "Sales", "EBIT", "COGS", 
                     "Operating Income After Depreciation", "EBITDA", "Capital Expenditures",
                     "Interest Related Expense", "Current Assets", "Current Liabilities", 
                     "Cash and Short-Term Investments", "Debt in Current Liabilities",
                     "Debt in Long-Term Debt", "Common Equity", "Net Income"]
    
    for var in financial_vars:
        if var in df.columns:
            df[var] = df[var] * 1_000_000  # Convert millions to dollars
    
    # Calculate Market Cap using CRSP method (following Wharton documentation)
    # MarketCap = PRC × SHROUT × 1000 (where SHROUT is in thousands, multiply by 1000 for actual shares)
    df["Market Cap"] = df["Annual Price Close (Fiscal)"] * df["Shares Outstanding (CRSP)"] * 1000
    # Adding some key financial ratios and metrics 
    
    # VALUATION METRICS
    df["P/E Ratio"] = df["Market Cap"] / df["Net Income"]  # Price-to-Earnings
    df["P/B Ratio"] = df["Market Cap"] / df["Common Equity"]  # Price-to-Book
    df["P/S Ratio"] = df["Market Cap"] / df["Sales"]  # Price-to-Sales
    # PROFITABILITY METRICS
    df["ROE"] = df["Net Income"] / df["Common Equity"]  # Return on Equity
    df["ROA"] = df["Net Income"] / df["Total Assets"]  # Return on Assets
    df["Operating Margin"] = df["Operating Income After Depreciation"] / df["Sales"]
    df["Net Profit Margin"] = df["Net Income"] / df["Sales"]
    df["EBITDA Margin"] = df["EBITDA"] / df["Sales"]
    # LEVERAGE & SOLVENCY METRICS
    df["Total Debt"] = df["Debt in Current Liabilities"] + df["Debt in Long-Term Debt"]
    df["Debt-to-Equity"] = df["Total Debt"] / df["Common Equity"]
    df["Debt-to-Assets"] = df["Total Debt"] / df["Total Assets"]
    df["Interest Coverage"] = df["EBIT"] / df["Interest Related Expense"]
    df["Current Ratio"] = df["Current Assets"] / df["Current Liabilities"]
    
    
    # PER-SHARE METRICS (using actual shares from ShrOut * 1000)
    actual_shares = df["Shares Outstanding (CRSP)"] * 1000
    df["Book Value per Share"] = df["Common Equity"] / actual_shares
    df["Sales per Share"] = df["Sales"] / actual_shares
    df["EPS (Calculated)"] = df["Net Income"] / actual_shares
    
    print(df.head())
clean_data("data.csv")
    