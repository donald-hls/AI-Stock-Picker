# AI-Stock-Picker

## This project leverages a hybrid approach—combining classical financial econometrics (Fama-French) with modern machine learning (XGBoost)— to identifies opportunities by analyzing over 20 years of historical fundamental, market, and macroeconomic data (2001–2024).. By integrating fundamental metrics and macroeconomic indicators from FRED, the model selects a high-conviction 5-stock portfolio designed to evalluate how much they outperformed the broader market."

##  Data Processing
- Obtained historical equity data from Wharton Research Data Services
- Obtained historical macro data from Federal Reserve Economic Data
- Used Historic Data obtained from 2000 to 2024
- The system implements a 4-month publication lag for fundamental data to prevent "look-ahead bias," ensuring the model only trains on information that was actually available to investors at the time.

## Modeling & Ensembling

- Fama-French Three Factor Model (Market Risk (Market Risk Premium), Size (SMB), Value (HML))
    - Used standard 50-50 split for Market Equity (size)
    - Used standard 30-70 split for Book to Equity (value)
    - A custom engine constructs SMB (Size) and HML (Value) factors using a 2x3 portfolio grid.
    - Used Fama-French factors as features in the XGBoost model

- XgBoost (Tree Models)
    - Uses RandomizedSearchCV to tune hyperparameters (learning rate, tree depth, and regularization) across 500 different iterations to ensure robust generalization.
    - Picked 5 Stocks with the model



