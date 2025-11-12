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
    """
    return series.rolling(window = window, min_periods = min_periods).apply(
        lambda arr: np.prod(1.0 + arr) - 1.0, raw=True)