from datetime import timedelta
import pandas as pd
import warnings
from numpy import number
warnings.filterwarnings('ignore')


class Data_Processing():
    '''
    Class to train and infer stock price for one particular stock
    '''
    def __init__(self, mt_start, mt_end, 
                 index_path = '../data/1.1-FTSE-IDX_VOL30-PRICES_2006-2023.csv',
                 price_path = '../data/1.1-FTSE_VOL30-PRICES_2006-2023.csv',
                 esg_path = '../data/1.2-FTSE_ESG_COR_2006-2023.csv'):

        self.mt_start = mt_start
        self.mt_end = mt_end
        self.index_path = index_path
        self.price_path = price_path
        self.esg_path = esg_path

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

        price_df.loc[:, 'month_lag1'] = price_df.month_key.apply(lambda x: x - timedelta(days=10))
        price_df.loc[:, 'month_lag1'] = price_df.month_lag1.apply(lambda x: x.strftime('%Y-%m-01'))
        price_df.month_lag1 = pd.to_datetime(price_df.month_lag1)
        price_df = price_df[price_df.Date.isin(self.date_list)].reset_index(drop=True)

        self.price_df = price_df
        # month_df.to_csv('files/1.1-FTSE_VOL30-MONTHLY-PRICES_2006-2023.csv', index=None)


    def data_preprocessing_esg(self):

        esg_df = pd.read_csv(self.esg_path)

        # set-up df
        esg_df.Asset = esg_df.Asset.astype(int)
        esg_df.Date = pd.to_datetime(esg_df.Date)
        esg_df = esg_df.drop(['windowTimestamp'], axis=1)

        # set-up month_key column
        esg_df = esg_df[esg_df.Date.isin(self.date_list)].reset_index(drop=True)
        esg_df['month_key'] = esg_df.Date.apply(lambda x: x.strftime('%Y-%m-01'))
        esg_df.month_key = pd.to_datetime(esg_df.month_key)

        self.esg_df = esg_df

    def merge_data(self):
        '''
        '''
        self.monthly_last_trading_date()
        self.data_preprocessing_price()
        self.data_preprocessing_esg()
        merge_df = pd.merge(self.price_df, self.esg_df, how = 'left', left_on = ['month_lag1', 'Asset'],
                                 right_on = ['month_key', 'Asset'])
        
        # output column arrangement
        merge_df.drop(['month_key_x', 'month_key_y', 'Date_y'], axis = 1, inplace = True)
        merge_df = merge_df.rename(columns={
                         'Date_x': 'date_key',
                         'month_lag1': 'month_key'
                         })
        self.merge_df = merge_df

        return self.merge_df
    
    def cleansing_final1(self, fillna = None):
        '''
        The excluded columns: ResourceUse, HumanRights, CSRStrategy, and Emissions were selected
        Based on columns that mostly contribute null to the FTSE assets.
        '''
        self.merge_data()

        #filter exclude columns
        clean_df = self.merge_df.drop(['ResourceUse', 'HumanRights', 'CSRStrategy', 'Emissions'], axis=1)

        #count how many missing values and total observation
        #exclude missing value that more than 50% and obs less than 24
        cnt_miss_rws = self.clean_count_missing_rows_assets(clean_df)
        exc = cnt_miss_rws[(cnt_miss_rws.perc_missing > 0.5) | (cnt_miss_rws.total_rows <= 24)].num_assets.tolist()
        clean_df = clean_df[~(clean_df.Asset.isin(exc))]

        # fill null value with Original filling method
        if fillna:
            clean_df = self.clean_filna_assets_df(clean_df)

        self.clean_df = clean_df
        return clean_df