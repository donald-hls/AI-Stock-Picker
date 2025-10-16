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
    