# ESG News (MarketPsych) to Stock Volatility Prediction

## Guide to utilise the repository

### Experimentation Notebook Flow:

1. Pull the ESG Data (Internal API and Library)

> 1-ESG Scores Collection.ipynb

2. Pull the OHLC Price Data (Internal API and Library)

> 1.1-Price Collection.ipynb (FTSE100 Constituents)
> 1.1-Price Index Collection.ipynb (FTSE100 Index)

3. Check the "Robustness" of the OHLC Price Data

> 1.1.2-Price_Robustness_v2.ipynb

4. Calculate the Yang-Zhang Volatility

> 1.2-Volatility-Calculation_v3.ipynb

5. Merge Data ESG and Volatility Data

> 1.3.1-Merge-Fillna-Data.ipynb

6. Train - Hyperparameter Tune - Test of HAR, ElasticNet

> 3.1.6-EN-Hyperparameter-Tuning-and-Test.ipynb
> 3.1.6-all_model_new_data.ipynb

6. Training and Tuning LSTM

> 3.1.5-LSTM.ipynb

7. Paper's Presentation 

> Presentation.ipynb

---