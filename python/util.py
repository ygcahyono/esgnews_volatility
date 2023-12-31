# adding a feature of daily data

from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import time

# Method #3 Regularisation Model
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from datetime import timedelta
from numpy import asarray, log1p, expm1
from numpy import number
from sklearn.metrics import mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import warnings
import numpy as np
import sys

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

def calculate_iqr(values):
    # Calculate Q1
    Q1 = np.percentile(values, 25)
    # Calculate Q3
    Q3 = np.percentile(values, 75)
    # Calculate IQR
    IQR = Q3 - Q1
    return IQR

def detect_outliers_iqr(values):
    # Calculate the IQR of the values
    IQR = calculate_iqr(values)
    # Calculate Q1 and Q3
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    # Define the lower and upper bound for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Return a boolean array: True if the value is an outlier, False otherwise
    return lower_bound, upper_bound

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
    agg = pd.concat(cols, axis=1)
    agg = pd.DataFrame(agg)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg.drop(dropcols, axis=1).values

# split a univariate dataset into train/test sets
def spv_train_test_split(data, n_test):
    '''
    train test split based on refer to the array set, with the same style as the random forest.
    '''
    return data[:-n_test, :], data[-n_test:, :]

class Data_Processing():
    '''
    Class to train and infer stock price for one particular stock
    '''
    def __init__(self, mt_start, mt_end, 
                 validation = False,
                 threshold = 24,
                 daily = False, 
                 train_perc = .7,
                 index_path = '../data/1.1-FTSE-IDX_VOL30-PRICES_2006-2023.csv',
                 price_path = '../data/1.1-FTSE_VOL30-PRICES_2006-2023.csv',
                 esg_path = '../data/1.2-FTSE_ESG_COR_2006-2023.csv'):

        self.mt_start = mt_start
        self.mt_end = mt_end
        self.train_perc = train_perc
        self.index_path = index_path
        self.price_path = price_path
        self.esg_path = esg_path
        self.validation = validation
        self.threshold = threshold
        self.daily = daily

    def count_train_test(self, train_df, test_df):
        master_df = pd.DataFrame()

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
    
    def min_data_threshold(self, df):
        
        threshold = self.threshold

        if self.daily:
            threshold = 360*2 # at least it has 2 years of datapoints.

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
        
        validation = self.validation
        train_rows = self.train_perc
        df = self.clean_df
        
        lag_1, lag_2, lag_3 = 1, 3, 12
        if self.daily:
            lag_1, lag_2, lag_3 = 1, 7, 30
        
        df.index = df.col_merge

        train_df, valid_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        asset_lists = df.Asset.unique()

        for asset in asset_lists:
            # subset dataframe
            temp_df = df[df['Asset'] == asset].copy()

            # parameters
            rows = temp_df.shape[0]
            train_len = ceil(rows*train_rows)

            # setting up volatility lag to a dataframe
            vol_df = pd.DataFrame({
            'vol_series_daily' : temp_df['V^YZ'].shift(lag_1),
            'vol_series_weekly' : temp_df['V^YZ'].rolling(lag_2).mean().shift(1),
            'vol_series_monthly' : temp_df['V^YZ'].rolling(lag_3).mean().shift(1)
            })

            temp_df = pd.merge(temp_df, vol_df, how = 'left', left_index=True, right_index=True)

            # split the subset into train_df
            train_df = pd.concat([temp_df.iloc[:train_len], train_df])
            test_df = pd.concat([temp_df.iloc[train_len:], test_df])

            if validation:
                # if yes validation has 20% of the portion.
                valid_len = int(rows*.2)

                valid_df = pd.concat([temp_df.iloc[train_len:(train_len+valid_len)], valid_df])
                valid_df = pd.concat([temp_df.iloc[(train_len+valid_len):], valid_df])

        
        master_df = self.count_train_test(train_df, test_df) # count the total rows of each assets
        used_assets = self.min_data_threshold(master_df)     # filter out assets that has least data points

        train_df = train_df[train_df.Asset.isin(used_assets)]
        # valid_df = valid_df[valid_df.Asset.isin(used_assets)]
        test_df = test_df[test_df.Asset.isin(used_assets)]

        return train_df, valid_df, test_df

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
    
    def cleansing_final1(self, fillna = None):
        '''
        The excluded columns: ResourceUse, HumanRights, CSRStrategy, and Emissions were selected
        Based on columns that mostly contribute null to the FTSE assets.
        '''
        validation = self.validation

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

        train_df, valid_df, test_df = self.func_train_test_split()
        
        return clean_df, train_df, valid_df, test_df


class Run_Algorithms():
    '''
    Initialize parameters for the class
    :param train_df: train's dataset
    :param test_df: test's dataset
    :param algorithms: algorithm: GARCH, HAR, EN, RF
    :param features: "m1", "m2", or "m3"
    :param cap: caping the nonsense prediction value. i.e. below 0 or ultra-high spike
    :param plot_export: export plot to "outputs" folder
    :param res_export: export result of algo to results file
    '''


    def __init__(self, train_df, test_df, algorithms = 'HAR',
                 vif_factor = 5, sample = True, features = 'm1', cap = False, plot_export = False, res_export = False):

        self.train_df = train_df
        self.test_df = test_df        
        self.algorithms = algorithms 
        self.vif_factor = vif_factor
        self.sample = sample
        self.features = features
        self.cap = cap
        self.plot_export = plot_export
        self.res_export = res_export

    def get_asset_name(self):

        # sys.path.append('../data')

        coverage_df = pd.read_csv('../data/coverage_dataframe.csv')
        coverage_df.PermID = coverage_df.PermID.astype(int)
        coverage_df = coverage_df[['PermID', 'Name']]
        coverage_df = coverage_df.rename(columns={'PermID':'Asset'})

        return coverage_df
    
    # fit an random forest model and make a one step prediction
    def elasticnet_forecast(self, train, testX, l1_ratio = 0.25):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = ElasticNet(l1_ratio= l1_ratio)
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]

    # fit an random forest model and make a one step prediction
    def random_forest_forecast(self, train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = RandomForestRegressor(n_estimators=100, n_jobs = -1)
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]
    
    # walk-forward validation for univariate data
    def walk_forward_validation(self, data, n_test, algorithm, verbose = True):
        predictions = list()
        # split dataset
        train, test = spv_train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # print(history)
        # step over each time-step in the test set
        
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]

            if algorithm == 'EN':
                # fit model on history and make a prediction
                yhat = self.elasticnet_forecast(history, testX)

            elif algorithm == 'RF':
                # fit model on history and make a prediction
                yhat = self.random_forest_forecast(history, testX)

            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
            # summarize progress
            # print('>expected=%.4f, predicted=%.4f' % (testy, yhat))
            
        # estimate prediction error
        error = mean_squared_error(test[:, -1], predictions) * 10**3
        # error = mean_absolute_error(test[:, -1], predictions)
        
        return test[:, -1], predictions, error

    def vif_check(self, feature_version = 2):

        train_df = self.train_df
        test_df = self.test_df
        vif_factor = self.vif_factor
        
        merge_df = pd.concat([train_df, test_df])
        merge_df = merge_df.dropna()

        cols = self.features_selections(features = 'm3', feature_version = feature_version)

        merge_df = merge_df[cols]
        merge_df = add_constant(merge_df)

        vif = pd.DataFrame()
        vif['VIF Factor'] = [variance_inflation_factor(merge_df.values, i) for i in range(merge_df.shape[1])]
        vif['features'] = merge_df.columns
        
        features = vif[vif['VIF Factor'] <= vif_factor]['features'].tolist()
        
        return features

    def asset_selections(self, sample = True):

        assets = self.train_df.Asset.unique().tolist()
        if sample: 
            # 3rd and 4th most volatile stocks in FTSE 2006 to 2022
            assets = [8589934212, 8589934254] 

        self.assets = assets
        return assets

    def features_selections(self, features = 'm1', feature_version = 2):
        '''
        Current version #2 is the version after pre-processing and removing four columns and does not including the variables
        that come from HAR algorithms
        '''

        if features == 'm1':
            cols = ['vol_series_daily', 'vol_series_weekly', 'vol_series_monthly', 'V^YZ']

        elif features == 'm3':

            # version 1
            cols = [
                'buzz','ESG','ESGCombined','ESGControversies','EnvironmentalPillar','GovernancePillar','SocialPillar'
                            ,'CSRStrategy','Community','Emissions','EnvironmentalInnovation','HumanRights','Management','ProductResponsibility'
                            ,'ResourceUse','Shareholders','Workforce', 'vol_series_daily','vol_series_weekly','vol_series_monthly', 'V^YZ']
        
            if feature_version == 2:
                # version 2
                cols = ['buzz','ESG','ESGCombined','ESGControversies','EnvironmentalPillar','GovernancePillar','SocialPillar','Community',
                        'EnvironmentalInnovation','Management','ProductResponsibility','Shareholders','Workforce', 'V^YZ']
        
        else:
            cols = self.vif_check(feature_version = feature_version)

        self.cols = cols
        return cols

    def vis_line_plot_results(self, y_pred, y_test, algorithms, name, r, features):

        plt.figure(figsize=(10,4))
        true, = plt.plot(y_test)
        preds, = plt.plot(y_pred)
        plt.title(f'{algorithms}-{features}-{name}', fontsize=15)
        plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=9)
        plt.xticks(rotation=0)
        plt.savefig(f'../outputs/{algorithms}-{features}/{str(r+1).zfill(3)}-{algorithms}-{name}.png')
        plt.close()

    def run_ols(self, train_df, test_df, asset, cols, cap = False, target = 'V^YZ'):

        # (df_train, df_test, asset, target = 'V^YZ')
        df_train = train_df[train_df.Asset == asset][cols].dropna()
        df_test = test_df[test_df.Asset == asset][cols].dropna()
        
        df_train = log1p(df_train)
        df_test = log1p(df_test)
        
        X_train = df_train.drop([target], axis=1)
        X_test = df_test.drop([target], axis=1)
        
        y_train = df_train[target]
        y_test = df_test[target]

        # if algorithms == 'HAR':
        # deactivate the if else of HAR
        X_train = sm.add_constant(X_train)
        X_test.loc[:, 'const'] = 1
        X_test = X_test[X_train.columns]

        # Fit the model
        model = OLS(y_train, X_train)
        model_fit = model.fit()

        # display(X_test, X_train)
        y_pred = model_fit.predict(X_test)

        y_test = expm1(y_test)
        y_pred = expm1(y_pred)

        if cap:
            y_pred = y_pred.clip(lower = 0)

        return y_test, y_pred
    
    def run_garch(self, train_df, test_df, asset, cap = False
                  , target = 'V^YZ'
                  , p = 1
                  , q = 1):
        
        y_pred = []
        test_size = test_df[test_df['Asset'] == asset].shape[0]
        
        y_volatility = train_df[train_df['Asset'] == asset][target]
        y_volatility.append(test_df[test_df['Asset'] == asset][target])
        y_test = y_volatility[-test_size:]
        
        for i in range(test_size):
            # train data
            y_train = y_volatility[-(test_size-i):]
            model = arch_model(y_train, p=p, q=q)
            model_fit = model.fit(disp='off')
            # test data
            pred = model_fit.forecast(horizon=1)
            pred = np.sqrt(pred.variance.values[-1,:][0])
            
            # capping the outliers
            if cap:
                lower_bound, upper_bound = detect_outliers_iqr(y_train)
                if upper_bound < pred:
                    pred = upper_bound
            
            y_pred.append(pred)

        indices = y_test.index
        y_pred = pd.Series(y_pred, index=indices)

        return y_test, y_pred, test_size
    
    def run_supervised(self, train_df, test_df, asset, cols, name, 
                       algorithm, 
                       cap = False, target = 'V^YZ', test_perc = .3):

        df_train = train_df[train_df.Asset == asset][cols].dropna()
        df_test = test_df[test_df.Asset == asset][cols].dropna()
        indices = test_df[test_df.Asset == asset].index
        # display(df_train)
        df_merge = pd.concat([df_train, df_test])
        df_merge = series_to_supervised(df_merge, n_in=3, target= [target])
        test_size = int(df_merge.shape[0] * test_perc)
        
        print(f'Execute Training and Walk Forward Testing for ({name}-{str(asset)}) for {test_size} times..')
        start_time = time.time()
        y_test, y_pred, mse = self.walk_forward_validation(df_merge, test_size, algorithm)
        print("---"*10, "%s seconds |"%(time.time() - start_time), 'MAE: %.3f'%mse, "---"*10)

        return y_test, y_pred, mse, test_size


    def go_iterate_assets(self, train_df, test_df, cap = False):

        plot_export = self.plot_export
        features = self.features
        sample = self.sample
        algorithms = self.algorithms
        
        assets = self.asset_selections(sample)
        cols = self.features_selections(features = features)
        print(cols)
        coverage_df = self.get_asset_name()

        mresults = pd.DataFrame()

        for r, asset in enumerate(assets): 
            
            name = coverage_df[coverage_df.Asset == asset].iloc[0, 1]

            if algorithms == 'HAR':
                test_size = test_df.shape[0]
                y_test, y_pred = self.run_ols(train_df, test_df, asset, cols)
                mse = mean_squared_error(y_test,y_pred)*10**3

            elif algorithms == 'GARCH':
                # Notice Garch did not take any cols as an input parameter.
                y_test, y_pred, test_size = self.run_garch(train_df, test_df, asset)
                mse = mean_squared_error(y_test,y_pred)*10**3

            elif algorithms == 'EN':
                y_test, y_pred, mse, test_size = self.run_supervised(train_df, test_df, asset, cols, name, algorithms)

            elif algorithms == 'RF':
                y_test, y_pred, mse, test_size = self.run_supervised(train_df, test_df, asset, cols, name, algorithms)

            mresult = pd.DataFrame({
                'Asset': asset,
                'Name': name,
                'Model': algorithms,
                'Features': features.upper(),
                'Test Size': test_size,
                'MSE^3':mse
                        }
                , index=[r]
            )

            mresults = pd.concat([mresults, mresult])

            if plot_export: 
                self.vis_line_plot_results(y_pred, y_test, algorithms, name, r, features)

        return mresults

    def compile_train_test(self):
        '''
        '''

        variables = {'GARCH': 1
                     ,'HAR': 2
                     ,'EN': 3
                     ,'RF': 4}

        train_df = self.train_df
        test_df = self.test_df
        features = self.features
        algorithms = self.algorithms
        no_algo = variables[algorithms]
        res_export = self.res_export

        train_df.Asset = train_df.Asset.astype(int)
        test_df.Asset = test_df.Asset.astype(int)

        mresults = self.go_iterate_assets(train_df, test_df)
        if res_export:
            mresults.to_csv(f'../results/{str(no_algo).zfill(2)}-{algorithms}-{features}.csv', index=None)

        return mresults