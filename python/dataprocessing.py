from datetime import timedelta
from math import ceil
from numpy import asarray, log1p, expm1
from numpy import number
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import time
import warnings
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

class DataProcessing():
    '''
    Class to train and infer stock price for one particular stock
    '''
    def __init__(self, mt_start, mt_end, 
                 daily = False, 
                 predict_days = 200,
                 index_path = '../data/1.1-FTSE-IDX_VOL30-PRICES_2006-2023.csv',
                 price_path = '../data/1.1-FTSE_VOL30-PRICES_2006-2023.csv',
                 esg_path = '../data/1.2-FTSE_ESG_COR_2006-2023.csv'):

        self.mt_start = mt_start
        self.mt_end = mt_end
        self.predict_days = predict_days
        self.index_path = index_path
        self.price_path = price_path
        self.esg_path = esg_path
        self.daily = daily

    def count_train_test(self, train_df, test_df):
        master_df = pd.DataFrame()

        assets = train_df.Asset.unique().tolist()

        for _, asset in enumerate(assets): 
            df_train = train_df[train_df.Asset == asset]
            df_test = test_df[test_df.Asset == asset]

            master_df.loc[_ , 'Asset'] = asset
            master_df.loc[_ , 'Train Length'] = df_train.shape[0]
            master_df.loc[_ , 'Test Length'] = df_test.shape[0]
            master_df.loc[_ , 'Total Length'] = df_train.shape[0] + df_test.shape[0]

        return master_df
    
    def min_data_threshold(self, df):

        if self.daily:
            # At least 720 days or 2 years datapoints
            threshold = 360*2 

        else:
            # At least 24 months or 2 years datapoints.
            threshold = 24

        return df[df['Total Length'] >= threshold]['Asset'].tolist()

    def clean_count_missing_rows_assets(self, df):
        master_df = pd.DataFrame()

        for i, asset in enumerate(df.Asset.unique()):
            df_temp = df[df.Asset == asset]
            df_clean = df_temp.dropna()
            
            temp2_df = pd.DataFrame({ 'num_assets': asset,
                            'missing_rows': df_temp.shape[0] - df_clean.shape[0],
                            'perc_missing': (df_temp.shape[0] - df_clean.shape[0]) / df_temp.shape[0],
                            'total_rows': df_temp.shape[0]
                        }, index= [i])
            
            master_df = pd.concat([master_df, temp2_df])

        return master_df

    def clean_filna_assets_df(self, df):
        '''
        Version 1 of fill Null value, the first performance is using this function.
        '''

        Assets = df.Asset.unique()
        master_df = pd.DataFrame()

        for asset in Assets:
            temp_df = df[df['Asset'] == asset]

            # Select numerical columns
            numerical_columns = temp_df.select_dtypes(include=[number]).columns
            # Check for any missing values in these numerical columns
            missing_numerical = [col for col in numerical_columns if temp_df[col].isna().any()]
            
            for col in missing_numerical:
                mrows = temp_df[col].isna().sum() + 1
                roll_mean = temp_df[col][::-1].rolling(window=mrows, min_periods=1).mean()
                temp_df[col] = temp_df[col].fillna(roll_mean)
                temp_df[col] = temp_df[col].fillna(method='ffill')

            temp_df = temp_df.reset_index(drop=True)
            master_df = pd.concat([master_df, temp_df])

        return master_df

    def monthly_last_trading_date(self):

        date_list = []

        df = pd.read_csv(self.index_path)
        dt_trades = df.loc[:,['Date']]
        dt_trades.Date = pd.to_datetime(dt_trades.Date)
        dt_trades.loc[:, 'Month_Key'] = dt_trades.Date.apply(lambda x: x.strftime('%Y-%m-01'))
        dt_trades.Month_Key = pd.to_datetime(dt_trades.Month_Key)
        dt_trades = dt_trades[(dt_trades['Month_Key'] >= self.mt_start) & (dt_trades['Month_Key'] <= self.mt_end)]

        for date in dt_trades.Month_Key.unique():
            temp_df = dt_trades[dt_trades.Month_Key == date].copy()
            temp_df = temp_df.sort_values(by= 'Date', ascending=True)
            dt = temp_df.iloc[-1, 0]
            date_list.append(dt)

        self.date_list = date_list

    def data_preprocessing_price(self):

        select_cols = ['month_key', 'Date', 'Asset', 'Open', 'High', 'Low', 'Close', 'Return', 'V^CC', 'V^RS', 'V^YZ']
        
        price_df = pd.read_csv(self.price_path)
        price_df = price_df.rename(columns={'Month':'month_key'})
        price_df.Date = pd.to_datetime(price_df.Date)
        price_df.month_key = pd.to_datetime(price_df.month_key)
        price_df.Asset = price_df.Asset.astype(int)
        price_df = price_df[select_cols]
        price_df = price_df.dropna()

        if self.daily:
            price_df.loc[:, 'col_merge'] = price_df.Date.apply(lambda x: x - timedelta(days=1))
            price_df.col_merge = pd.to_datetime(price_df.col_merge)
        else:
            price_df.loc[:, 'col_merge'] = price_df.month_key.apply(lambda x: x - timedelta(days=10))
            price_df.loc[:, 'col_merge'] = price_df.col_merge.apply(lambda x: x.strftime('%Y-%m-01'))
            price_df.col_merge = pd.to_datetime(price_df.col_merge)
            price_df = price_df[price_df.Date.isin(self.date_list)].reset_index(drop=True)

        self.price_df = price_df


    def data_preprocessing_esg(self):

        esg_df = pd.read_csv(self.esg_path)

        # set-up df
        esg_df.Asset = esg_df.Asset.astype(int)
        esg_df.Date = pd.to_datetime(esg_df.Date)
        esg_df = esg_df.drop(['windowTimestamp'], axis=1)

        if not self.daily:
            # set-up month_key column
            esg_df = esg_df[esg_df.Date.isin(self.date_list)].reset_index(drop=True)

        esg_df['month_key'] = esg_df.Date.apply(lambda x: x.strftime('%Y-%m-01'))
        esg_df.month_key = pd.to_datetime(esg_df.month_key)

        self.esg_df = esg_df

    def func_train_test_split(self):
        '''
        '''
        
        df = self.clean_df
        
        
        if self.daily:
            lag_1, lag_2, lag_3 = 1, 7, 30
        else:
            # lag for monthly
            lag_1, lag_2, lag_3 = 1, 3, 12
        
        df.index = df['col_merge']

        train_df, test_df = pd.DataFrame(), pd.DataFrame()
        asset_lists = df.Asset.unique()

        for asset in asset_lists:
            # subset dataframe
            temp_df = df[df['Asset'] == asset].copy()

            # parameters
            predict_days = self.predict_days

            # setting up volatility lag to a dataframe
            vol_df = pd.DataFrame({
            'vol_series_daily' : temp_df['V^YZ'].shift(lag_1),
            'vol_series_weekly' : temp_df['V^YZ'].rolling(lag_2).mean().shift(1),
            'vol_series_monthly' : temp_df['V^YZ'].rolling(lag_3).mean().shift(1)
            })

            temp_df = pd.merge(temp_df, vol_df, how = 'left', left_index=True, right_index=True)

            # split the subset into train_df
            train_df = pd.concat([temp_df.iloc[:-(predict_days)], train_df])
            test_df = pd.concat([temp_df.iloc[-(predict_days):], test_df])

        
        master_df = self.count_train_test(train_df, test_df) # count the total rows of each assets
        used_assets = self.min_data_threshold(master_df)     # filter out assets that has least data points

        train_df = train_df[train_df.Asset.isin(used_assets)]
        test_df = test_df[test_df.Asset.isin(used_assets)]

        return train_df, test_df

    def merge_data(self):
        '''
        '''
        self.monthly_last_trading_date()
        self.data_preprocessing_price()
        self.data_preprocessing_esg()

        if not self.daily:
            merge_df = pd.merge(self.price_df, self.esg_df, how = 'left', left_on = ['col_merge', 'Asset'],
                                    right_on = ['month_key', 'Asset'])
            
            # output column arrangement
            merge_df.drop(['month_key_x', 'month_key_y', 'Date_y'], axis = 1, inplace = True)
            merge_df = merge_df.rename(columns={
                            'Date_x': 'date_key',
                            })
            
        else:
            merge_df = pd.merge(self.price_df, self.esg_df, how = 'left', left_on = ['col_merge', 'Asset'],
                        right_on = ['Date', 'Asset'])
            merge_df.drop(['month_key_x', 'month_key_y', 'Date_y'], axis = 1, inplace = True)
            merge_df = merge_df.rename(columns={
                            'Date_x': 'date_key',
                            })

        self.merge_df = merge_df

        return self.merge_df
    
    def clean_final(self, fillna = None):
        '''
        The excluded columns: ResourceUse, HumanRights, CSRStrategy, and Emissions were selected
        Based on columns that mostly contribute null to the FTSE assets.
        '''

        self.merge_data()

        #filter exclude columns # should be more dynamic.
        clean_df = self.merge_df.drop(['ResourceUse', 'HumanRights', 'CSRStrategy', 'Emissions'], axis=1)

        #count how many missing values and total observation
        #exclude missing value that more than 50% and obs less than 24
        cnt_miss_rws = self.clean_count_missing_rows_assets(clean_df)
        exc = cnt_miss_rws[(cnt_miss_rws.perc_missing > 0.5)].num_assets.tolist()
        clean_df = clean_df[~(clean_df.Asset.isin(exc))]

        # fill null value with Original filling method
        if fillna:
            clean_df = self.clean_filna_assets_df(clean_df)

        self.clean_df = clean_df

        train_df, test_df = self.func_train_test_split()
        
        return clean_df, train_df, test_df