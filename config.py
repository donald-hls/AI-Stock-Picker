"""
Tracking defined Variables / Constants, etc..
Contains constants, column mappings, and other configuration parameters.
"""

# Column name mapping from WRDS raw data to clean names
COLUMN_NAME_MAPPING = {
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

# Financial variables that need to be converted from millions to dollars
FINANCIAL_VARS_IN_MILLIONS = [
    "Total Assets", 
    "Total Liabilities", 
    "Sales", 
    "EBIT", 
    "COGS",
    "Operating Income After Depreciation", 
    "EBITDA", 
    "Capital Expenditures",
    "Interest Related Expense", 
    "Current Assets", 
    "Current Liabilities",
    "Cash and Short-Term Investments", 
    "Debt in Current Liabilities",
    "Debt in Long-Term Debt", 
    "Common Equity", 
    "Net Income"
]

# Fundamental columns to keep for annual data
FUNDAMENTAL_COLS = [
    "SP Identifier",
    "Fiscal Year",
    "company_name",
    "EPS",
    "Sales",
    "EBIT",
    "COGS",
    "Operating Income After Depreciation",
    "EBITDA",
    "Capital Expenditures",
    "Interest Related Expense",
    "Total Assets",
    "Total Liabilities",
    "Current Assets",
    "Current Liabilities",
    "Cash and Short-Term Investments",
    "Debt in Current Liabilities",
    "Debt in Long-Term Debt",
    "Common Equity",
    "Net Income",
    "Dividends per Share"
]

# Monthly price columns
PRICE_COLS = [
    "SP Identifier", 
    "PERMNO", 
    "month", 
    "Monthly Price", 
    "Monthly Total Return",
    "Monthly Total Return Excluding Dividends", 
    "Shares Outstanding (CRSP)"
]

# FRED API series IDs for macroeconomic data
FRED_SERIES = {
    "VIX": "VIXCLS",  # Volatility Index
    "Yield10": "DGS10",  # 10-Year Treasury Yield
    "Yield3M": "TB3MS",  # 3-Month Treasury Yield
    "Unemployment": "UNRATE",  # Unemployment Rate
    "Corporate_OAS": "BAMLC0A0CM"  # ICE BofA US Corporate Index Option-Adjusted Spread
}

# Data collection parameters
PUBLICATION_LAG_MONTHS = 4  # Months to wait for financial statement publication
START_DATE = "2010-01-01"  # Start date for macro data collection
FAR_FUTURE_DATE = "2200-12-31"  # Placeholder for "no expiration"

# Unit conversion factors
MILLION_TO_DOLLARS = 1_000_000
THOUSAND_TO_ACTUAL = 1_000  # For CRSP shares outstanding
