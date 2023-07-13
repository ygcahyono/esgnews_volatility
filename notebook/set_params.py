from math import ceil
from pandas import DataFrame, read_csv, concat, to_datetime, merge

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
    df = read_csv('../data/1.3-FTSE_Monthly_ESG_Volatility_Final.csv')
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
    df = read_csv('../data/1.3-FTSE_Monthly_ESG_Volatility_Final.csv')
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