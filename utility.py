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