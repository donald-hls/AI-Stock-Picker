# AI-Stock-Picker

##  Data Processing
- Obtained historical data from Wharton Research Data Services
- Used Historic Data obtained from 2001 to 2024 

- Collected and computed fundamental metrics / Macro data from Federal Reserve Economic Data

## Modeling & Ensembling

- Fama-French Three Factor Model (Market Risk (Market Risk Premium), Size (SMB), Value (HML))
    - Used standard 50-50 split for Market Equity (size)
    - Used standard 30-70 split for Book to Equity (value)

- XgBoost (Tree Models)
    - Tuned Hyper-Parameters with GridSearchCV



