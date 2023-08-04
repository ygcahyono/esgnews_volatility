import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from tensorflow.keras.regularizers import l1
from keras.initializers import glorot_normal
from sklearn.metrics import mean_squared_error

from tensorflow.keras.optimizers import Adam

def cross_corr(a,b):
    '''
    Compute the cross-correlation between
    :param a: Time-series data of first stock
    :param b: Time-series data of second stock
    :return: Cross-correlation of the two stocks that are input
    '''
    return (a*b).sum()/((a**2).sum()*(b**2).sum())**0.5

def get_tick_values(tickerSymbol, start, end):
    '''
    Function to extract the time series data
    :param tickerSymbol: String of stock ticker
    :param start: String of starting date of the time-series data
    :param end: String of ending date of the time-series data
    :return: type(list): Time series data
    '''
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = yf.download(tickerSymbol, start=start, end=end)
    tickerDf = tickerDf['Adj Close']
    data = tickerDf
    return data.predict_days

def get_control_vector(val):
    '''
    Returns the mask of day instances where stock purchase/sell decisions are to be made
    :param val: Input array of stock predict_days
    :return: np.array of decisions maks labels (-2/0/2)
    '''
    return np.diff(np.sign(np.diff(val)))

class LSTM_Model_MS_II():
    '''
    '''

    def __init__(self, train_df, test_df, start, end, sector = 'Consumer cyclicals', features = 'm1',
                 train_test_split = 0.6, validation_split = .1,
                 past_history = 60, forward_look = 1,  batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = 0, infer_train = True,
                 depth = 1, naive = False, predict_days = 100, plot_values = True, plot_bot = True,
                 tickerSymbolList = None, sameTickerTestTrain = True, 
                 algorithms = 'LSTM', dropout = [False, 0,3], hu = 64, early_stop = None):
        
        '''
        Initialize parameters for the class
        :param tickerSymbol: String of Ticker symbol to train on
        :param start: String of start date of time-series data
        :param end: String of end date of time-series data
        :param past_history: Int of past number of days to look at
        :param forward_look: Int of future days to predict at a time
        :param train_test_split: Float of fraction train-test split
        :param batch_size: Int of mini-batch size
        :param epochs: Int of total number of epochs in training
        :param steps_per_epoch: Int for total number of mini-batches to run over per epoch
        :param validation_steps: Int of total number of steps to use while validating with the dev set
        :param verbose: Int to decide to print training stage results
        :param infer_train: Flag to carry out prediction on training set
        :param depth: Int to decide depth of stacked LSTM
        :param naive: Flag for deciding if we need a Vanila model
        :param predict_days: Int for number of days to predict for by iteratively updating the time-series histroy # adjusting the test set to 100
        :param plot_values: Flag to plot
        :param plot_bot: Flag to plot the investment growth by the decision making bot
        :param tickerSymbolList: List of tickers to train the model on
        :param sameTickerTestTrain: Falg, for model containing the ticker on which predictions are made
        '''

        self.train_df = train_df
        self.test_df = test_df
        self.sector = sector
        self.features = features
        self.validation_split = validation_split
        self.dropout = dropout
        self.hu = hu
        self.early_stop = early_stop

        self.start = start
        self.end = end
        self.past_history = past_history
        self.forward_look = forward_look
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.predict_days = predict_days
        self.depth = depth
        self.naive = naive
        self.custom_loss = False
        self.plot_values = plot_values
        self.plot_bot = plot_bot
        self.infer_train = infer_train
        self.sameTickerTestTrain = sameTickerTestTrain
        self.algorithms = algorithms
        tf.random.set_seed(1728)

    def unique_sector(self):

        select_cols = ['PermID', 'Name', 'TRBCEconomicSector']
        rename_cols = ['Assets', 'Firm Name', 'Economic Sector']

        sector = self.sector
        train_df = self.train_df
        cover_df = pd.read_csv('../data/coverage_dataframe.csv')[select_cols]
        cover_df['PermID'] = cover_df.PermID.astype(int)

        info_df = pd.DataFrame({
            'Assets': train_df.Asset.unique()
        })

        info_df = pd.merge(info_df, cover_df, how = 'left', left_on = 'Assets', right_on= 'PermID')
        info_df = info_df.iloc[:, 1:]
        info_df.columns = rename_cols

        ### experiment only including ftse100 not PIT

        # top100 = np.load('../data/asset_FTSE_PIT_END2022.npy')
        copy_df = info_df.copy()
        # copy_df = copy_df[copy_df['Assets'].isin(top100)]
        copy_df['Economic Sector'] = 'All'
        ### experiment

        info_df = pd.concat([copy_df, info_df])

        list_asset = info_df[info_df['Economic Sector'] == sector]['Assets'].tolist()
        self.ts = info_df[info_df['Economic Sector'] == sector]['Firm Name'].unique().tolist()

        self.info_df = info_df
        self.list_asset = list_asset

    def sequential_feature_selection(self):
        
        # Call variable from self.
        # print('before the seq processing:', cols)
        cols = self.cols
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
            # cols = self.vif_feature_selection(feature_version = feature_version)

            # sequential feature selection with basis version 2
            cols = self.features_selections(features = 'm3', feature_version = 2)
            cols = self.sequential_feature_selection()

        self.col_lengths = len(cols)
        self.cols = cols

    def train_test_prep(self):

        self.unique_sector()
        self.features_selections(features = self.features, feature_version = 2)
        print(self.cols)

        train_df = self.train_df
        test_df = self.test_df
        list_asset = self.list_asset
        train_test_split = self.train_test_split

        merge_df = pd.concat([train_df, test_df])
        merge_df = merge_df[merge_df.Asset.isin(list_asset)].dropna()

        y_all = []
        for id in list_asset:
            data = merge_df[merge_df.Asset == id][self.cols]
            y_all.append(data.values)
            maxTestValues = len(data.values) - int(len(data.values) * train_test_split)

        print('train_test_prep() function is done')
        self.y_all = y_all
    
    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        '''
        Preprocess the data to make either the test set or the train set
        :param dataset: np.array of time-series data
        :param iStart: int of index start
        :param iEnd: int of index end
        :param sHistory: int number of days in history that we need to look at
        :param forward_look: int of number of days in the future that needs to predicted
        :return: returns a list of test/train data

        training=(y, 0, validation_size, ...), 
        test=(y, validation_size, None, ...) means 
        it will start from the middle until the last row of observation.

        '''
        data = []
        target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look + 1
        # print(iEnd)
        for i in range(iStart, iEnd): 
            indices = range(i - sHistory, i)  # set the order
            if forward_look > 1:
                fwd_ind = range(i, i + forward_look)
                fwd_entity = np.asarray([])
                fwd_entity = np.append(fwd_entity, dataset[fwd_ind])
            reshape_entity = np.asarray([])
            reshape_entity = np.append(reshape_entity, dataset[
                indices][:, :-1])  # Comment this out if there are multiple identifiers in the feature vector
            
            # col length - 1 reduce total column from 
            data.append(np.reshape(reshape_entity, (sHistory, self.col_lengths - 1))) 
            if forward_look > 1:
                target.append(np.reshape(fwd_entity, (forward_look, self.col_lengths - 1)))
            else:
                target.append(dataset[i][-1])

        data = np.array(data)
        target = np.array(target)
        return data, target

    def prepare_test_train(self):
        '''
        Create the dataset from the extracted time-series data
        '''

        self.xtrain, self.xtest, self.xt = [], [], []
        self.ytrain, self.ytest, self.yt = [], [], []
        self.y_size = 0
        count = 1

        for y in self.y_all:
            # - 1 # adding a new rule - 1 to make sure not make it out of index.
            training_size = int(len(y) * self.train_test_split) 
            validation_size = int(len(y) * self.validation_split)

            # training preprocess
            ytrain = y[:training_size]
            training_mean = ytrain.mean(axis=0)  # get the average
            training_std = ytrain.std(axis=0)  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.

            self.training_mean = training_mean
            self.training_std = training_std

            ytrain = (ytrain - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios

            data, target = self.data_preprocess(ytrain, 0, training_size, self.past_history, forward_look=self.forward_look)

            self.xtrain.append(data)
            self.ytrain.append(target)
            self.y_size += len(y)
            # print(f'{count} length of training data: {len(data)}')

            # validation preprocess
            y_valid = y[training_size:]
            y_valid = (y_valid - training_mean) / training_std
            data, target = self.data_preprocess(y_valid, 0, validation_size, self.past_history, forward_look=self.forward_look)

            self.xtest.append(data)
            self.ytest.append(target)

            # print(f'{count} length of valid data: {len(data)}')

            # test prep
            data, target = self.data_preprocess(y_valid, validation_size, None, self.past_history, forward_look=self.forward_look)

            self.xt.append(data)
            self.yt.append(target)

            # print(f'{count} length of test data: {len(data)}')
            count+=1

        self.xtrain = np.concatenate(self.xtrain)
        self.ytrain = np.concatenate(self.ytrain)
        
        self.xtest = np.concatenate(self.xtest)
        self.ytest = np.concatenate(self.ytest)

        print('prepare_test_train() is done')
        # return y[:training_size] * training_std + training_mean

    def create_p_test_train(self):
        '''
        Prepare shuffled train and test data
        '''
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y_size
        p_train = tf.data.Dataset.from_tensor_slices((self.xtrain, self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices((self.xtest, self.ytest))
        self.p_test = p_test.batch(BATCH_SIZE).repeat()

    
    def create_rnn_model(self, 
                         hu=32, lags= 60, layer='LSTM', depth = 0
                         , features=3, output_size = 1, algorithm='estimation'
                         , activation = 'linear', seed = 100
                         ):
        # Create an Adam optimizer with a learning rate of 0.001
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        regulariser = l1(0.01)
        init_norm = glorot_normal(seed=seed)

        self.model = Sequential()
        if layer == 'LSTM':
            self.model.add(LSTM(hu, return_sequences = True, kernel_regularizer=regulariser, input_shape=(lags, features)
                                , kernel_initializer=init_norm
                                ))
            
        if self.dropout[0]:
            self.model.add(Dropout(self.dropout[1], seed=seed))

        if algorithm == 'estimation':
            
            for _ in range(depth):
                self.model.add(LSTM(hu, kernel_regularizer=regulariser, return_sequences = True, kernel_initializer=init_norm))
                if self.dropout[0]:
                    self.model.add(Dropout(self.dropout[1], seed=seed))

            self.model.add(LSTM(hu, kernel_regularizer=regulariser, kernel_initializer=init_norm))
            self.model.add(Dense(output_size, activation = activation))
            self.model.compile(optimizer=opt, loss='mse', metrics=['mse'])

            
            print(self.model.summary())

        return self.model

    def model_LSTM(self):
        '''
        Create the stacked LSTM model and train it using the shuffled train set
        '''

        
        early_stop = EarlyStopping(monitor='val_loss'
                                   , patience=10
                                   # Restore model weights from the epoch with the best value
                                   , restore_best_weights=True
                                   , verbose=1)
        
        self.model = self.create_rnn_model(hu = self.hu, lags = self.past_history, depth = self.depth,
                                           layer='LSTM', features = len(self.cols) - 1, output_size=self.forward_look)
        self.create_p_test_train()

        if self.early_stop:
            self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                    validation_data = self.p_test, validation_steps = self.validation_steps,
                    callbacks=[early_stop], verbose = self.verbose)

        else:    
            self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                    validation_data = self.p_test, validation_steps = self.validation_steps,
                    verbose = self.verbose)
        
        hist = self.hist
        model = self.model

        print('Running model_LSTM() is done')

        return hist, model
    
    def infer_values(self, xtest, ytest, ts = None):
        '''
        Infer predict_days by using the test set
        :param xtest: test dataset
        :param ytest: actual value dataset
        :param ts: tikcer symbol
        :return: model variables that store predicted data
        '''
        self.pred = []
        if self.infer_train:
            self.pred_train = []
            # self.pred_update_train = []
            self.usetest_train = self.xtrain.copy()
        
        predict_days = xtest.shape[0]
        if xtest.shape[0] > self.predict_days:
            predict_days = self.predict_days

        self.indicies = self.test_df.index[-(predict_days - 1):]

        for i in range(predict_days):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]), verbose = 0)[0][:]
            self.pred.append(self.y_pred)

        self.pred = np.array(self.pred)

        # self.pred_update = np.array(self.pred_update)
        self.RMS_error = self.hist.history['val_mse'][-1]
        self.RMS_error_train = self.hist.history['mse'][-1]

        pred = self.pred
        RMS_error = self.RMS_error

        # return pred, pred_update, RMS_error_update, RMS_error
        return pred, RMS_error, predict_days
                
    def plot_test_values(self, yt, pred, ts, RMS_error):
        '''
        Plot predicted predict_days against actual predict_days
        '''
        print(f'length of actual value {len(yt[:self.predict_days-1])}, length of predicted value {len(pred[1:])}')
        
        yt = yt * self.training_std[-1] + self.training_mean[-1]
        pred = pred * self.training_std[-1] + self.training_mean[-1]

        mse = np.mean((yt-pred)**2)

        plt.figure()
        plt.plot(yt[:self.predict_days-1],label='actual (%s)'%ts)
        plt.plot(pred[1:],label='predicted (%s)'%ts)
        # plt.plot(pred_update[1:],label='predicted (update)')
        plt.xlabel("Days")
        plt.ylabel("Normalized stock price")
        plt.title('The Mean Squared Error is %f' % mse)
        plt.legend()
        plt.savefig('../outputs/LSTM/MultiStock_prediction_%d_%d_%d_%d_%s.png'%(
        self.depth,int(self.naive), self.past_history, self.forward_look, ts))
        plt.clf()

    def vis_line_plot_results(self, y_test, y_pred, name, r, predict_days):

        dictionaries = {
            'EN': 'Elastic Net',
            'RF': 'Random Forest',
            'LSTM': 'Long Short-Term Memory',
            'HAR': 'Heterogeneous AutoRegressive',
            'GARCH': 'Generalised AutoRegressive Conditional Heteroskedasticity'
        }

        algorithms = self.algorithms
        features = self.features

        if algorithms == 'LSTM':

            print(f'length of actual value {len(y_test[:predict_days-1])}, length of predicted value {len(y_pred[1:])}')
            y_test = y_test * self.training_std[-1] + self.training_mean[-1]
            y_pred = y_pred * self.training_std[-1] + self.training_mean[-1]
            y_test = y_test[:predict_days-1]
            y_pred = y_pred[1:]

            y_pred = [i for i in y_pred[:, 0]]

            y_test = pd.Series(y_test, index = self.indicies)
            y_pred = pd.Series(y_pred, index = self.indicies)

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
        ax1.set_ylim([np.min(y_test)-np.min(y_test)*.5, np.max(y_test)+np.max(y_test)*.15]) 

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
    
    def full_workflow(self, model = None):
        self.train_test_prep()
        self.prepare_test_train()
        self.model_LSTM()

    def full_workflow_and_plot(self, model = None):
        '''
        Workflow to carry out the entire process end-to-end
        :param model: Choose which model to use to predict inferred predict_days
        :return:
        '''
        self.full_workflow(model = model)
        mresults = pd.DataFrame()

        for i, name  in enumerate(self.ts):
            pred, RMS_error, predict_days = self.infer_values(self.xt[i], self.yt[i], self.ts[i])
            self.vis_line_plot_results(self.yt[i], pred, self.ts[i], i, predict_days) 
            # self.plot_test_values(self.yt[i], pred, self.ts[i], RMS_error)

            mresult = pd.DataFrame({
                # 'MSE': round(np.mean(((self.yt[i][:self.predict_days-1] * self.training_std[-1] + self.training_mean[-1])
                #                 - (pred[1:]* self.training_std[-1] + self.training_mean[-1])
                #                 )**2), 10),
                'MSE': mean_squared_error( 
                        (self.yt[i][:(len(pred[1:]))] * self.training_std[-1] + self.training_mean[-1]),
                            (pred[1:]* self.training_std[-1] + self.training_mean[-1])
                ),
                'RMSE': round(np.sqrt(np.mean(((self.yt[i][:self.predict_days-1]* self.training_std[-1] + self.training_mean[-1]) 
                                - (pred[1:]* self.training_std[-1] + self.training_mean[-1])
                                )**2)), 10),
                # 'RMS_error': round(RMS_error,4),
                'NumPred': len(pred[1:]),
                'Firm': self.ts[i]
                }, index = [i])
            
            mresults = pd.concat([mresults, mresult])

        mresults.to_csv(f'../results/LSTM-{self.features}-{self.sector}-{i}.csv')