import numpy as np
import pandas as pd
"""
The file stores all the helper functions
"""

def ratio_calculation(S1, S2, eps = 1e-9):
    """
        Helper function to calculate the ratio of 2 Sereis Objects,
        used for data cleaning functions.
    
    ratio_calculation(S1, S2) -> Series
    
    """
    denominator = S2.mask(S2.abs() < eps, np.nan)
    ratio = S1.divide(denominator)
    return ratio.where(np.isfinite(ratio))

def limiting_extreme_values(s, p=0.01):
    """
    this is the process of windsorizing the data to remove any extreme values.
    limiting_extreme_values(s, p=0.01): Series, float -> Series
    """
    # if not enough data, return the original series
    if s.notna().sum() < 15: 
        return s
    lower_quantile = s.quantile(p)
    upper_quantile = s.quantile(1-p)
    # Assigns values outside boundary to boundary values. (to trim inputs)
    return s.clip(lower_quantile, upper_quantile)
    
def rolling_total_return(series, window, min_periods):
    """
    rolling_total_return calculates the rolling total return for a given series.
    The function returns a Series with the rolling total return
    rolling_total_return(series, window, min_periods): Series, int, int -> Series
    e.g: (1+r_1) * (1+r_2) * ... * (1+r_n) - 1
    """
    return series.rolling(window, min_periods).apply(
        lambda array: np.prod(1.0 + array) - 1.0, raw=True)
    
def zscore(group):
    """
    zscore standardises each cross-sectional slice (mean 0, std 1).
    The function returns a DataFrame with the z-scored data and a Series with the columns to keep.
    """
    x = group.replace([np.inf, -np.inf], np.nan)
    std = x.std(ddof=0)
    # Documents which features survivied the filter.
    keep = std.gt(1e-8)
    x = x.loc[:, keep]
    z = (x - x.mean()) / std[keep]
    return z.fillna(0.0), keep
