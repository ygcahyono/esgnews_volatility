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
                #  price_path = '../data/1.1-FTSE_VOL30-PRICES_2006-2022.csv', # new threshold with 100 ftse only
                #  price_path = '../data/1.1-FTSE_VOL30-PRICES_2006-2023.csv',
                 price_path = '../data/1.1-FTSE_VOL21-PRICES_2006-2022.csv',
                 esg_path = '../data/1.2-FTSE_ESG_COR_2006-2023.csv'):

        self.mt_start = mt_start
        self.mt_end = mt_end
        self.predict_days = predict_days
        self.index_path = index_path
        self.price_path = price_path
        self.esg_path = esg_path
        self.daily = daily

    # Define function to generate noise
    def generate_noise(self, T, beta, gamma, RV):
        epsilon = np.random.randn(T)
        omega = gamma * RV.std() * np.sqrt(1 - beta**2)
        u = np.zeros(T)
        
        for t in range(1, T):
            u[t] = beta * u[t-1] + omega * epsilon[t]
            
        return u

    def count_train_test(self, train_df, test_df):
        master_df = pd.DataFrame()

        assets = train_df.Asset.unique().tolist()

        for _, asset in enumerate(assets): 
            df_train = train_df[train_df.Asset == asset].dropna()
            df_test = test_df[test_df.Asset == asset].dropna()

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
    
    def clean_first_valid_index(self, df):
        master_df = pd.DataFrame()

        for i, asset in enumerate(df.Asset.unique()):
            df_temp = df[df.Asset == asset]
            missing_numerical = [col for col in df_temp.columns if df_temp[col].isna().any()]

            # Loop over all columns
            for column in missing_numerical:
                # find the index of the first non-null observation in the column
                first_valid_index = df_temp[column].first_valid_index()

                # drop rows before the first non-null observation
                df_temp = df_temp.loc[first_valid_index:]

            master_df = pd.concat([master_df, df_temp])
            # print(f'clean first valid index done: {i} out of {len(df.Asset.unique())}')

        return master_df

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
    
    def forward_fill_with_rolling_mean(self, s, window):
        for i in range(len(s)):
            if pd.isnull(s[i]):
                s[i] = s[:i].tail(window).mean()
        return s

    def clean_filna_assets_df(self, df):
        '''
        Version 1 of fill Null value, the first performance is using this function.
        '''

        Assets = df.Asset.unique()
        master_df = pd.DataFrame()
        print(len(Assets))

        for asset in Assets:
            temp_df = df[df['Asset'] == asset].reset_index(drop=True)

            # Select numerical columns
            numerical_columns = temp_df.select_dtypes(include=[number]).columns
            # Check for any missing values in these numerical columns
            missing_numerical = [col for col in numerical_columns if temp_df[col].isna().any()]
            
            for col in missing_numerical:
                # forward filling with rolling mean 30 days
                temp_df[col] = self.forward_fill_with_rolling_mean(temp_df[col], 30)
                

            temp_df = temp_df.reset_index(drop=True)
            master_df = pd.concat([master_df, temp_df])

        print(master_df.shape)
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

        select_cols = ['month_key', 'Date', 'Date (shifted)', 'Asset', 'Open', 'High', 'Low', 'Close', 'Return', 'V^CC', 'V^RS', 'V^YZ']
        
        price_df = pd.read_csv(self.price_path)
        price_df = price_df.rename(columns={'Month':'month_key'})
        price_df.Date = pd.to_datetime(price_df.Date)
        price_df.month_key = pd.to_datetime(price_df.month_key)
        price_df.Asset = price_df.Asset.astype(int)
        price_df = price_df[select_cols]
        price_df = price_df.dropna()

        if self.daily:
            # I change the days = 1 to days = 30 as per 31 July 2023.
            # price_df.loc[:, 'col_merge'] = price_df.Date.apply(lambda x: x - timedelta(days=30))
            price_df.loc[:, 'col_merge'] = price_df.Date
            price_df.col_merge = pd.to_datetime(price_df.col_merge)
            price_df.drop(['Date'], axis=1, inplace=True)
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

    def generate_noise_robustness(self, stop = 10):
        # Noise generation calibrations
        df = self.clean_df.copy()
        betas = [0.0, 0.5, 0.75, 0.90]
        gammas = [0.25, 0.50, 1.00]
        master_df = pd.DataFrame()
        
        for asset in df.Asset.unique():
            sub_df = df[df['Asset'] == asset]
            T = sub_df.shape[0]

            RV = sub_df['V^YZ']
            cnt = 0
            for beta in betas:
                for gamma in gammas:
                    sub_df[f'noise_beta_{beta}_gamma_{gamma}'] = self.generate_noise(T, beta, gamma, RV)
                    cnt+=1
                    if cnt == stop:
                        break
                if cnt == stop:
                    break
                    
                    
            master_df = pd.concat([master_df, sub_df])
        
        self.clean_df = master_df

    def func_train_test_split(self):
        '''
        '''
        
        df = self.clean_df
        
        
        if self.daily:
            # lag_1, lag_2, lag_3 = 1*30, 3*30, 12*30
            lag_1, lag_2, lag_3 = 1, 5, 21
        else:
            # lag for monthly
            lag_1, lag_2, lag_3 = 1, 3, 12
        
        df['col_merge'] = pd.to_datetime(df['col_merge'])
        df['Date (shifted)'] = pd.to_datetime(df['Date (shifted)'])

        df.index = df['col_merge']

        train_df, test_df = pd.DataFrame(), pd.DataFrame()
        asset_lists = df.Asset.unique()

        for asset in asset_lists:
            # subset dataframe
            temp_df = df[df['Asset'] == asset].copy()
            temp_df.to_excel('test_csvcsv.xlsx')
            
            # temp_df.to_csv('test_sample.csv')
            # break
            # parameters
            predict_days = self.predict_days

            # setting up volatility lag to a dataframe
            vol_df = pd.DataFrame({
            'vol_series_daily' : temp_df['V^YZ'].shift(lag_1),
            'vol_series_weekly' : temp_df['V^YZ'].rolling(lag_2).mean().shift(1),
            'vol_series_monthly' : temp_df['V^YZ'].rolling(lag_3).mean().shift(1)
            })

            # additional feature
            vol_df = vol_df.shift(lag_3)

            pointer = temp_df.index[0] + timedelta(days = 30)
            target_vol = temp_df[temp_df.index >= pointer]

            temp_df = pd.merge(target_vol, vol_df, how = 'left', left_index=True, right_index=True).dropna()
            # display(temp_df)

            # split the subset into train_df
            train_df = pd.concat([temp_df.iloc[:-(predict_days)], train_df])
            test_df = pd.concat([temp_df.iloc[-(predict_days):], test_df])

        
        master_df = self.count_train_test(train_df, test_df) # count the total rows of each assets
        # display(master_df)
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
            merge_df.drop(['month_key_x', 'month_key_y'], axis = 1, inplace = True)
            merge_df = merge_df.rename(columns={
                            'Date': 'date_key',
                            })
            
        else:
            merge_df = pd.merge(self.price_df, self.esg_df, how = 'left', left_on = ['col_merge', 'Asset'],
                        right_on = ['Date', 'Asset'])
            # display(merge_df.head())
            merge_df.drop(['month_key_x', 'month_key_y'], axis = 1, inplace = True)
            merge_df = merge_df.rename(columns={
                            'Date': 'date_key',
                            })

        self.merge_df = merge_df

        return self.merge_df

    def selection_criteria_assets(self, df):
        b4_2010 = pd.DataFrame(df[df['date_key'] < '2016-01-01'].Asset.unique())
        af_2020 = pd.DataFrame(df[df['date_key'] > '2019-12-31'].Asset.unique())

        b4af_df = pd.concat([b4_2010, af_2020], axis=0)
        b4af_df.columns = ['Asset']
        b4af_df['flag'] = 1
        b4af_df = b4af_df.groupby(['Asset'], as_index = False).count()
        b4af_df = b4af_df[b4af_df['flag'] == 2]

        return b4af_df.Asset.tolist()
    
    def clean_final(self, fillna = None):
        '''
        The excluded columns: ResourceUse, HumanRights, CSRStrategy, and Emissions were selected
        Based on columns that mostly contribute null to the FTSE assets.
        '''

        self.merge_data()

        #filter exclude columns # should be more dynamic.
        clean_df = self.merge_df.drop(['ResourceUse', 'HumanRights', 'CSRStrategy', 'Emissions'], axis=1)

        clean_df = self.clean_first_valid_index(clean_df)
        #count how many missing values and total observation
        #exclude missing value that more than 50% and obs less than 24
        # cnt_miss_rws = self.clean_count_missing_rows_assets(clean_df)
        # exc = cnt_miss_rws[(cnt_miss_rws.perc_missing > 0.5)].num_assets.tolist()
        # clean_df = clean_df[~(clean_df.Asset.isin(exc))]

        # fill null value with Original filling method
        if fillna:
            clean_df = self.clean_filna_assets_df(clean_df)

        

        # select the criteria of asset that exists before 2016 and after 2020.
        # select_assets = np.load('../data/asset_selection_criteria_2015-2020.npy')
        select_assets = self.selection_criteria_assets(clean_df)
        clean_df = clean_df[clean_df.Asset.isin(select_assets)]

        self.clean_df = clean_df

        self.generate_noise_robustness()
        train_df, test_df = self.func_train_test_split()
        
        return clean_df, train_df, test_df