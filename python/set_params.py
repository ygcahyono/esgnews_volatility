from math import ceil
from pandas import DataFrame, read_csv, concat, to_datetime, merge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def count_train_test(train_df, test_df):
    master_df = DataFrame()

    assets = train_df.Asset.unique().tolist()

    for _, asset in enumerate(assets): 
        df_train = train_df[train_df.Asset == asset]
        # df_valid = valid_df[valid_df.Asset == asset]
        df_test = test_df[test_df.Asset == asset]

        master_df.loc[_ , 'Asset'] = asset
        master_df.loc[_ , 'Train Length'] = df_train.shape[0]
        # master_df.loc[_ , 'Valid Length'] = df_valid.shape[0]
        master_df.loc[_ , 'Test Length'] = df_test.shape[0]
        master_df.loc[_ , 'Total Length'] = df_train.shape[0] + df_test.shape[0]

    return master_df

def min_data_threshold(df, threshold = 24):

    return df[df['Total Length'] >= threshold]['Asset'].tolist()


def func_garch_train_test_split(validation = False, threshold = 24):
    '''
    '''
    train_rows = .7
    df = read_csv('../data/1.3-FTSE_Monthly_ESG_Volatility_Final_v2.csv')
    df = df.rename(columns={'Date_x':'date_key'})
    
    df.date_key = to_datetime(df.loc[:, 'date_key'])
    df.month_key = to_datetime(df.loc[:, 'month_key'])
    df.index = df.month_key

    train_df, valid_df, test_df = DataFrame(), DataFrame(), DataFrame()
    asset_lists = df.Asset.unique()

    for asset in asset_lists:
        temp_df = df[df['Asset'] == asset].copy()

        rows = temp_df.shape[0]
        train_len = ceil(rows*train_rows)

        train_df = concat([temp_df.iloc[:train_len], train_df])
        if validation:
            valid_len = int(rows*.2)

            valid_df = concat([temp_df.iloc[train_len:(train_len+valid_len)], valid_df])
            valid_df = concat([temp_df.iloc[(train_len+valid_len):], valid_df])

        else:
            test_df = concat([temp_df.iloc[train_len:], test_df])

    master_df = count_train_test(train_df, test_df)
    used_assets = min_data_threshold(master_df, threshold = threshold)

    train_df = train_df[train_df.Asset.isin(used_assets)]
    # valid_df = valid_df[valid_df.Asset.isin(used_assets)]
    test_df = test_df[test_df.Asset.isin(used_assets)]

    return train_df, valid_df, test_df

def func_train_test_split(validation = False, threshold = 24):
    '''
    '''
    train_rows = .7
    df = read_csv('../data/1.3-FTSE_Monthly_ESG_Volatility_Final_v2.csv')
    df = df.rename(columns={'Date_x':'date_key'})
    
    df.date_key = to_datetime(df.loc[:, 'date_key'])
    df.month_key = to_datetime(df.loc[:, 'month_key'])
    df.index = df.month_key

    train_df, valid_df, test_df = DataFrame(), DataFrame(), DataFrame()
    asset_lists = df.Asset.unique()

    for asset in asset_lists:
        # subset dataframe
        temp_df = df[df['Asset'] == asset].copy()

        # parameters
        rows = temp_df.shape[0]
        train_len = ceil(rows*train_rows)

        # setting up volatility lag to a dataframe
        vol_df = DataFrame({
        'vol_series_daily' : temp_df['V^YZ'].shift(1),
        'vol_series_weekly' : temp_df['V^YZ'].rolling(3).mean().shift(1),
        'vol_series_monthly' : temp_df['V^YZ'].rolling(12).mean().shift(1)
        })

        temp_df = merge(temp_df, vol_df, how = 'left', left_index=True, right_index=True)

        # split the subset into train_df
        train_df = concat([temp_df.iloc[:train_len], train_df])
        test_df = concat([temp_df.iloc[train_len:], test_df])

        if validation:
            # if yes validation has 20% of the portion.
            valid_len = int(rows*.2)

            valid_df = concat([temp_df.iloc[train_len:(train_len+valid_len)], valid_df])
            valid_df = concat([temp_df.iloc[(train_len+valid_len):], valid_df])

    
    master_df = count_train_test(train_df, test_df) # count the total rows of each assets
    used_assets = min_data_threshold(master_df)     # filter out assets that has least data points

    train_df = train_df[train_df.Asset.isin(used_assets)]
    # valid_df = valid_df[valid_df.Asset.isin(used_assets)]
    test_df = test_df[test_df.Asset.isin(used_assets)]

    return train_df, valid_df, test_df


def series_to_supervised(data, n_in=1, n_out=1, target = 'y',dropnan=True):
    '''
    transform a time series dataset into a supervised learning dataset
    '''
    cols = list()
    colname = data.columns
    dropcols = [col for col in colname if col not in target]
    # print('dropping columns:', dropcols)
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        temp_df = data.shift(i)
        colname = temp_df.columns + f'_s{i}'
        temp_df.columns = colname
        cols.append(temp_df)
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        
    # put it all together
    agg = concat(cols, axis=1)
    agg = DataFrame(agg)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg.drop(dropcols, axis=1).values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    '''
    train test split based on refer to the array set, with the same style as the random forest.
    '''
    return data[:-n_test, :], data[-n_test:, :]


def vif_check():

    merge_all_fill_df = read_csv('../data/1.3-FTSE_Monthly_ESG_Volatility_Final.csv')
    train_df, valid_df, test_df = func_train_test_split(validation = False)
    merge_all_fill_df = concat([train_df, test_df])
    merge_all_fill_df = merge_all_fill_df.dropna()

    cols = [
        'buzz','ESG','ESGCombined','ESGControversies','EnvironmentalPillar','GovernancePillar','SocialPillar'
                    ,'CSRStrategy','Community','Emissions','EnvironmentalInnovation','HumanRights','Management','ProductResponsibility'
                    ,'ResourceUse','Shareholders','Workforce', 'vol_series_daily','vol_series_weekly','vol_series_monthly', 'V^YZ']
    
        
    # version 2
    cols = ['buzz','ESG','ESGCombined','ESGControversies','EnvironmentalPillar','GovernancePillar','SocialPillar','Community',
            'EnvironmentalInnovation','Management','ProductResponsibility','Shareholders','Workforce', 'V^YZ']

    
    merge_all_fill_df = merge_all_fill_df[cols]
    merge_all_fill_df = add_constant(merge_all_fill_df)
    vif = DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(merge_all_fill_df.values, i) for i in range(merge_all_fill_df.shape[1])]
    vif["features"] = merge_all_fill_df.columns
    
    non_multicollinear_features = vif[vif['VIF Factor'] <= 5.0]
    
    return non_multicollinear_features['features'].tolist()


def vif_dynamic_check(df):
    
    df = add_constant(df)
    vif = DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif["features"] = df.columns
    
    non_multicollinear_features = vif[vif['VIF Factor'] <= 5.0]
    display(vif)
    
    return non_multicollinear_features['features'].tolist()


class Data_Processing():
    '''
    Class to train and infer stock price for one particular stock
    '''
    def __init__(self, dt_start, dt_end, df_path = '../data/1.1-FTSE-IDX_VOL30-PRICES_2006-2023.csv'):

        self.start = dt_start
        self.end = dt_end
        self.df_path = df_path


    def data_last_trading_month(self):

        date_list = []

        df = pd.read_csv(self.df_path)
        dt_trades = df.loc[:,['Date']]
        dt_trades.Date = pd.to_datetime(dt_trades.Date)
        dt_trades.loc[:, 'Month_Key'] = dt_trades.Date.apply(lambda x: x.strftime('%Y-%m-01'))
        dt_trades.Month_Key = pd.to_datetime(dt_trades.Month_Key)

        for date in dt_trades.Month_Key.unique():
            temp_df = dt_trades[dt_trades.Month_Key == date].copy()
            temp_df = temp_df.sort_values(by= 'Date', ascending=True)
            dt = temp_df.iloc[-1, 0]
            date_list.append(dt)

        return date_list


    def data_preprocessing_price():
        print()