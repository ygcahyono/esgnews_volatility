from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler

from arch import arch_model
from math import ceil
from numpy import asarray, log1p, expm1
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import warnings
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

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


class RunModels():
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

    def __init__(self, 
                 train_df, test_df, predict_days = 200, algorithms = 'HAR', vif_factor = 5, 
                 sample = True, features = 'm1', cap = False, plot_export = False, res_export = False
                 ):

        self.train_df = train_df
        self.test_df = test_df
        self.predict_days = predict_days        
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
    def elasticnet_forecast(self, train, testX, l1_ratio = 0.5):

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
        model = RandomForestRegressor(n_estimators=100, n_jobs = 40)
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]
    
    # walk-forward validation for univariate data
    def walk_forward_validation(self, data, n_test, algorithm, verbose = True):
        predictions = list()
        # split dataset
        train, test = spv_train_test_split(data, n_test)
        indicies = self.indicies

        # seed history with training dataset
        history = [x for x in train]
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

        testy = pd.Series(test[:, -1], index = indicies)
        predy = pd.Series(predictions, index = indicies)
        
        # return test[:, -1], predictions, error
        return testy, predy, error
        
    
    def sequential_feature_selection(self, cols):
        
        # Call variable from self.
        # print('before the seq processing:', cols)
        train_df = self.train_df
        train_df = train_df.select_dtypes(include=[np.number])
        train_df = train_df.dropna()

        # Initialize the linear regression model
        lr = LinearRegression()

        # Create the Sequential Feature Selector object
        sfs = SFS(lr, 
                forward=False, 
                k_features=7,
                n_jobs=-1
                )

        # Fit the object to the data.
        sfs = sfs.fit(train_df[cols].iloc[:, :-1], train_df[cols].iloc[:, -1]) 
        features = train_df[cols].columns[list(sfs.k_feature_idx_)].tolist()
        features.append('V^YZ') 
        
        return features

    def vif_feature_selection(self, feature_version = 2):

        train_df = self.train_df
        vif_factor = self.vif_factor
        
        train_df = train_df.dropna()
        cols = self.features_selections(features = 'm3', feature_version = feature_version)

        train_df = train_df[cols]
        train_df = add_constant(train_df)

        vif = pd.DataFrame()
        vif['VIF Factor'] = [variance_inflation_factor(train_df.values, i) for i in range(train_df.shape[1])]
        vif['features'] = train_df.columns
        
        features = vif[vif['VIF Factor'] <= vif_factor]['features'].tolist()

        # Check if '4' is in the list
        if 'V^YZ' not in features:
            features.append('V^YZ')
            print("adding variables")
        
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
        # clean_df = self.merge_df.drop(['ResourceUse', 'HumanRights', 'CSRStrategy', 'Emissions'], axis=1)
            if feature_version == 2:
                # version 2 is after the dropping some columns that have too many missing row values
                cols = ['vol_series_daily', 'vol_series_weekly', 'vol_series_monthly',
                    'buzz','ESG','ESGCombined','ESGControversies','EnvironmentalPillar','GovernancePillar','SocialPillar','Community',
                        'EnvironmentalInnovation','Management','ProductResponsibility','Shareholders','Workforce', 'V^YZ']
        
        else:            
            # deactivate the vif feature selection and aligning with the sequential feature selection
            cols = self.vif_feature_selection(feature_version = feature_version)

            # sequential feature selection with basis version 2
            # cols = self.features_selections(features = 'm3', feature_version = 2)
            # cols = self.sequential_feature_selection(cols)
            # print('inside the feature selections:', cols)

        self.cols = cols
        return cols

    # def vis_line_plot_results(self, y_pred, y_test, name, r):

    #     dictionaries = {
    #         'EN': 'Elastic Net',
    #         'RF': 'Random Forest',
    #         'LSTM': 'Long Short-Term Memory',
    #         'HAR': 'Heterogeneous AutoRegressive',
    #         'GARCH': 'Generalised AutoRegressive Conditional Heteroskedasticity'
    #     }

    #     algorithms = self.algorithms
    #     features = self.features

    #     plt.figure(figsize=(10,4))
    #     true, = plt.plot(y_test, alpha = 0.7, color = 'black')
    #     preds, = plt.plot(y_pred, marker='.')

    #     plt.title(f'{dictionaries[algorithms]} Prediction on "{name}" |Data:{features}|', fontsize=12)
    #     plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=9)

    #     # Add horizontal grid lines
    #     plt.grid(axis='y', alpha=0.5)

    #     # Add labels to the axes
    #     plt.xlabel('Date Key', fontsize=9)
    #     plt.ylabel('Volatility', fontsize=9)
    #     plt.xticks(rotation=0)

    #     plt.savefig(f'../outputs/{algorithms}-{features}/{str(r+1).zfill(3)}-{algorithms}-{name}.png')
    #     plt.close()

    def vis_line_plot_results(self, y_pred, y_test, name, r):

        dictionaries = {
            'EN': 'Elastic Net',
            'RF': 'Random Forest',
            'LSTM': 'Long Short-Term Memory',
            'HAR': 'Heterogeneous AutoRegressive',
            'GARCH': 'Generalised AutoRegressive Conditional Heteroskedasticity'
        }

        algorithms = self.algorithms
        features = self.features

        # Calculate absolute differences between actual and predicted values
        diff = np.abs(y_test - y_pred)

        fig, ax1 = plt.subplots(figsize=(10,5))

        # Plot actual and predicted values
        ax1.plot(y_test, alpha = 0.7, color = 'black')
        ax1.plot(y_pred, marker='.')
        ax1.legend(['True Volatility', 'Predicted Volatility'], fontsize=7.5, loc='upper left')
        ax1.grid(axis='y', alpha=0.5)
        ax1.set_ylabel('Volatility', fontsize=9)
        # print(np.min(y_test))
        ax1.set_ylim([np.min(y_test)-np.min(y_test)*.5, np.max(y_test)+np.max(y_test)*.05]) 

        # Create a second y-axis
        ax2 = ax1.twinx()

        # Plot differences on the secondary y-axis as a bar chart
        ax2.bar(y_test.index, diff, color='gray', alpha=0.8, width=1.5)
        ax2.legend(['Absolute Difference'], fontsize=7.5, loc='upper right')
        ax2.set_ylabel('Absolute Difference', fontsize=9)

        # Setting y-limits for the second axis to prevent overlap with line plots
        ax2.set_ylim([0, np.max(diff)*3]) 

        # Set main title
        plt.title(f'{dictionaries[algorithms]} Prediction on "{name}" [Data:{features}]', fontsize=12)

        plt.xticks(rotation=0)

        plt.savefig(f'../outputs/{algorithms}-{features}/{str(r+1).zfill(3)}-{algorithms}-{name}.png')
        plt.close()

    def run_ols(self, train_df, test_df, asset, cols, cap = False, target = 'V^YZ'):

        # (df_train, df_test, asset, target = 'V^YZ')
        df_train = train_df[train_df.Asset == asset][cols].dropna()
        df_test = test_df[test_df.Asset == asset][cols].dropna()
        test_size = df_test.shape[0]
        
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

        return y_test, y_pred, test_size
    
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

        predict_days = self.predict_days
        n_in = 3

        df_train = train_df[train_df.Asset == asset][cols].dropna()
        df_test = test_df[test_df.Asset == asset][cols].dropna()

        # normalisation
        # scaler = StandardScaler()
        # scaleXtrain = scaler.fit_transform(df_train.iloc[:, :-1])
        # scaleXtest = scaler.transform(df_test.iloc[:, :-1])
        # df_train = pd.concat([pd.DataFrame(scaleXtrain, index = train_indicies, columns= cols[:-1]), df_train.iloc[:, -1]], axis=1)
        # df_test = pd.concat([pd.DataFrame(scaleXtest, index = test_indicies, columns= cols[:-1]), df_test.iloc[:, -1]], axis=1)

        # indices = test_df[test_df.Asset == asset].index
        # display(df_train)
        df_merge = pd.concat([df_train, df_test])
        self.indicies = df_merge.iloc[-(predict_days+n_in):-n_in].index

        df_merge = series_to_supervised(df_merge, n_in=n_in, target= [target])
        print(f'Execute Training and Walk Forward Testing for ({name}-{str(asset)}) for {predict_days} times..')

        start_time = time.time()
        y_test, y_pred, mse = self.walk_forward_validation(df_merge, predict_days, algorithm)
        print("---"*10, "%s seconds |"%(time.time() - start_time), 'MAE: %.3f'%mse, "---"*10)

        return y_test, y_pred, mse, predict_days


    def go_iterate_assets(self, train_df, test_df, cap = False):

        plot_export = self.plot_export
        features = self.features
        sample = self.sample
        algorithms = self.algorithms
        
        assets = self.asset_selections(sample)
        cols = self.features_selections(features = features)
        print('after feature selections', cols)
        coverage_df = self.get_asset_name()

        mresults = pd.DataFrame()

        for r, asset in enumerate(assets): 
            
            name = coverage_df[coverage_df.Asset == asset].iloc[0, 1]

            if algorithms == 'HAR':

                y_test, y_pred, test_size = self.run_ols(train_df, test_df, asset, cols)
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
                self.vis_line_plot_results(y_pred, y_test, name, r)

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
            nw = str(int(time.time()))
            mresults.to_csv(f'../results/{str(no_algo).zfill(2)}-{algorithms}-{features}-{nw}.csv', index=None)

        return mresults