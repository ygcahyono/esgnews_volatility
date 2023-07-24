import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
import gtab

def get_categorical_tickers():
    '''
    This Function returns a dictionary of tickers for different industry types
    :return:
    ticker_dict: Dictionary of 9 different industry types with over 8 tickers each
    tickerSymbols: Set of three tickers
    '''
    ticker_dict = {}
    all_tickers = []
    ticker_dict['energy'] = ['XOM', 'CVX', 'SHEL', 'PTR.L', 'TTE', 'BP', 'PBR', '^GSPC', 'SLB', 'VLO']
    ticker_dict['materials'] = ['BHP', 'LIN', 'RIO', 'DD', 'SHW', 'CTA-PB', 'APD']
    ticker_dict['industrials'] = ['UPS', 'HON', 'LMT', 'BA', 'GE', 'MMM', 'RTX', 'CAT', 'WM', 'ABB', 'ETN', 'EMR',
                                  'FDX', 'TRI']
    ticker_dict['utilities'] = ['NEE', 'DUK', 'NGG', 'AEP', 'XEL','AWK' ,'ETR', 'PCG']
    ticker_dict['healthcare'] = ['UNH', 'JNJ', 'PFE', 'NVO', 'TMO', 'MRK', 'AZN', 'NVS', 'DHR', 'AMGN', 'CVS', 'GSK',
                                 'ZTS', 'GILD']
    ticker_dict['financials'] = ['BRK-A', 'V', 'JPM', 'BAC', 'MA', 'WFC', 'C-PJ', 'MS', 'RY', 'AXP']
    ticker_dict['discretionary'] = ['AMZN', 'TSLA', 'HD', 'BABA', 'TM', 'NKE', 'MCD', 'SBUX', 'F', 'MAR', 'GM', 'ORLY',
                                    'LILI', 'HMC', 'CMG', 'HLT']
    ticker_dict['staples'] = ['WMT', 'PG', 'KO', 'COST', 'PEP', 'BUD', 'UL', 'TGT', 'MDLZ', 'CL', 'DG', 'KHC', 'KDP',
                              'HSY']
    ticker_dict['IT'] = ['AAPL', 'MSFT', 'TSM', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'ACN', 'ADBE', 'INTC', 'CRM', 'TXN',
                         'QCOM', 'AMD', 'IBM', 'SONY', 'AMAT', 'INFY', 'ADI', 'MU', 'LRCX']
    ticker_dict['communication'] = ['GOOG', 'FB', 'DIS', 'VZ', 'CMCSA', 'TMUS', 'T', 'NFLX', 'SNAP', 'VOD', 'BAIDU',
                                    'TWTR', 'EA']
    ticker_dict['estate'] = ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'DLR', 'WELL', 'EQR', 'AVB', 'WY', 'INVH', 'MAA']
    ticker_keys = []
    for key in ticker_dict.keys():
        ticker_keys.append(key)
        all_tickers.append(ticker_dict[key])
    ticker_dict['all'] = all_tickers
    tickerSymbols = ['BRK-A', 'GOOG', 'MSFT']
    return ticker_dict, tickerSymbols

def get_company_names():
    '''
    Get a dictionary of search strings corresponding to different ticker labels
    :return:
    ticker_dict: Dictionary of search strings given a stock ticker
    '''
    ticker_dict = {}
    all_tickers = []
    ticker_dict['energy'] = {'XOM': 'Exxon Mobil', 'CVX': 'Chevron', 'SHEL': 'Shell', 'PTR': 'PetroChina',
                             'TTE': 'TotalEnergies', 'BP': 'BP', 'PBR': 'Petroleo Brasileiro',
                             'SNP': 'China Petroleum', 'SLB': 'Schlumberger', 'VLO': 'Valero'}
    '''
    ticker_dict['materials'] = ['BHP', 'LIN', 'RIO', 'DD', 'SHW', 'CTA-PB', 'APD']
    ticker_dict['industrials'] = ['UPS', 'HON', 'LMT', 'BA', 'GE', 'MMM', 'RTX', 'CAT', 'WM', 'ABB', 'ETN', 'EMR',
                                  'FDX', 'TRI']
    ticker_dict['utilities'] = ['NEE', 'DUK', 'NGG', 'AEP', 'XEL','AWK' ,'ETR', 'PCG']
    ticker_dict['healthcare'] = ['UNH', 'JNJ', 'PFE', 'NVO', 'TMO', 'MRK', 'AZN', 'NVS', 'DHR', 'AMGN', 'CVS', 'GSK',
                                 'ZTS', 'GILD']
    ticker_dict['financials'] = ['BRK-A', 'V', 'JPM', 'BAC', 'MA', 'WFC', 'C-PJ', 'MS', 'RY', 'AXP']
    ticker_dict['discretionary'] = ['AMZN', 'TSLA', 'HD', 'BABA', 'TM', 'NKE', 'MCD', 'SBUX', 'F', 'MAR', 'GM', 'ORLY',
                                    'LILI', 'HMC', 'CMG', 'HLT']
    ticker_dict['staples'] = ['WMT', 'PG', 'KO', 'COST', 'PEP', 'BUD', 'UL', 'TGT', 'MDLZ', 'CL', 'DG', 'KHC', 'KDP',
                              'HSY']
    ticker_dict['IT'] = ['AAPL', 'MSFT', 'TSM', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'ACN', 'ADBE', 'INTC', 'CRM', 'TXN',
                         'QCOM', 'AMD', 'IBM', 'SONY', 'AMAT', 'INFY', 'ADI', 'MU', 'LRCX']
    ticker_dict['communication'] = ['GOOG', 'FB', 'DIS', 'VZ', 'CMCSA', 'TMUS', 'T', 'NFLX', 'SNAP', 'VOD', 'BAIDU',
                                    'TWTR', 'EA']
    ticker_dict['estate'] = ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'DLR', 'WELL', 'EQR', 'AVB', 'WY', 'INVH', 'MAA']
    ticker_keys = []
    for key in ticker_dict.keys():
        ticker_keys.append(key)
        all_tickers.append(ticker_dict[key])
    ticker_dict['all'] = all_tickers
    '''
    return ticker_dict

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
    return data.values

def get_control_vector(val):
    '''
    Returns the mask of day instances where stock purchase/sell decisions are to be made
    :param val: Input array of stock values
    :return: np.array of decisions maks labels (-2/0/2)
    '''
    return np.diff(np.sign(np.diff(val)))

def buy_and_sell_bot(val,controls):
    '''
    Returns the growth of investment over time as function of the input decision mask and the stock values
    :param val: np.array of the actual stock value over time
    :param controls: np.array of the control mask to make purchase/sell decisions
    :return: np.array of percentage growth value of the invested stock
    '''
    inv = []
    curr_val = 100
    inds = np.where(controls)[0]
    buy_inds = np.where(controls>0)[0]
    sell_inds = np.where(controls<0)[0]
    max_limit = sell_inds[-1] if sell_inds[-1]>buy_inds[-1] else buy_inds[-1]
    for i in range(buy_inds[0]+2):
        inv.append(curr_val)
    for i in range(buy_inds[0],max_limit+1):
        if controls[i]>0:
            buy_val = val[i+1]
        elif controls[i]<0:
            sell_val = val[i+1]
            curr_val = curr_val*sell_val/buy_val
        inv.append(curr_val)
    if max_limit+1!=len(controls):
        for i in range(len(controls)-max_limit-1):
            inv.append(curr_val)
    return inv

class LSTM_Model_MS():
    '''
    Class to train and infer stock price for a model trained on multiple stocks of a given industry. The
    list of tickers can be separately supplied to train beyond tickers from one industry.
    '''
    def __init__(self,tickerSymbol, start, end,
                 past_history = 60, forward_look = 1, train_test_split = 0.8, batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = 0, infer_train = True,
                 depth = 1, naive = False, values = 200, plot_values = True, plot_bot = True,
                 tickerSymbolList = None, sameTickerTestTrain = True):
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
        :param values: Int for number of days to predict for by iteratively updating the time-series histroy
        :param plot_values: Flag to plot
        :param plot_bot: Flag to plot the investment growth by the decision making bot
        :param tickerSymbolList: List of tickers to train the model on
        :param sameTickerTestTrain: Falg, for model containing the ticker on which predictions are made
        '''
        self.tickerSymbol = tickerSymbol
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
        self.values = values
        self.depth = depth
        self.naive = naive
        self.custom_loss = False
        self.plot_values = plot_values
        self.plot_bot = plot_bot
        self.infer_train = infer_train
        self.sameTickerTestTrain = sameTickerTestTrain
        if tickerSymbolList == None:
            self.tickerSymbolList = [tickerSymbol]
        else:
            self.tickerSymbolList = tickerSymbolList
        tf.random.set_seed(1728)

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        '''
        Preprocess the data to make either the test set or the train set
        :param dataset: np.array of time-series data
        :param iStart: int of index start
        :param iEnd: int of index end
        :param sHistory: int number of days in history that we need to look at
        :param forward_look: int of number of days in the future that needs to predicted
        :return: returns a list of test/train data
        '''
        data = []
        target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look + 1
        for i in range(iStart, iEnd):
            indices = range(i - sHistory, i)  # set the order
            if forward_look > 1:
                fwd_ind = range(i, i + forward_look)
                fwd_entity = np.asarray([])
                fwd_entity = np.append(fwd_entity, dataset[fwd_ind])
            reshape_entity = np.asarray([])
            reshape_entity = np.append(reshape_entity, dataset[
                indices])  # Comment this out if there are multiple identifiers in the feature vector
            data.append(np.reshape(reshape_entity, (sHistory, 1)))  #
            if forward_look > 1:
                target.append(np.reshape(fwd_entity, (forward_look, 1)))
            else:
                target.append(dataset[i])
        data = np.array(data)
        target = np.array(target)
        return data, target

    def plot_history_values(self):
        '''
        Plots time-series data of the chosen ticker
        '''
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        y = data
        y.index = data.index
        y.plot()
        plt.title(f"{self.tickerSymbol}")
        plt.ylabel("price")
        plt.show()

    def get_ticker_values(self):
        '''
        Get ticker values in a list
        '''
        self.y_all = []
        for tickerSymbol in self.tickerSymbolList:
            tickerData = yf.Ticker(tickerSymbol)
            print(tickerSymbol, 'downloading...') # only the size of training set # the remaining data is not used
            tickerDf = yf.download(tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.y_all.append(data.values)
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)
        if self.sameTickerTestTrain == False: # This indicates self.tickerSymbol is the test ticker and self.tickerSymbolList is the training set
            print(self.tickerSymbol, 'downloading...')
            tickerData = yf.Ticker(self.tickerSymbol)
            tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.ytestSet = data.values # testset "ticker" become validation set and test set
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)


    def prepare_test_train(self):
        '''
        Create the dataset from the extracted time-series data
        '''
        self.y_size = 0
        if self.sameTickerTestTrain == True: # For each ticker, split data into train and test set. Test and validation are the same
            self.xtrain = []
            self.ytrain = []
            self.xtest = []
            self.ytest = []
            for y in self.y_all:
                training_size = int(y.size * self.train_test_split)
                training_mean = y[:training_size].mean()  # get the average
                training_std = y[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
                y = (y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look = self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                data, target = self.data_preprocess(y, training_size, None, self.past_history, forward_look = self.forward_look)
                self.xtest.append(data)
                self.ytest.append(target)
                self.y_size += y.size

            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)
            self.xtest = np.concatenate(self.xtest)
            self.ytest = np.concatenate(self.ytest)
            self.xt = self.xtest.copy()
            self.yt = self.ytest.copy()
        else: # For each ticker, data into train set only. Split test ticker data into validation and test sets
            self.xtrain = []
            self.ytrain = []
            for y in self.y_all:
                training_size = int(y.size) # only the size of training set # the remaining data is not used
                training_mean = y[:training_size].mean()  # get the average
                training_std = y[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
                y = (y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look=self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                self.y_size += y.size

            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)

            y = self.ytestSet # testset "ticker" become validation set and test set
            validation_size = int(y.size * self.train_test_split)
            validation_mean = y[:validation_size].mean()  # get the average
            validation_std = y[:validation_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
            y = (y - validation_mean) / validation_std
            data, target = self.data_preprocess(y, 0, validation_size, self.past_history, forward_look=self.forward_look)
            self.xtest = data
            self.ytest = target
            data, target = self.data_preprocess(y, validation_size, None, self.past_history, forward_look=self.forward_look)
            self.xt = data
            self.yt = target


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

    def model_LSTM(self):
        '''
        Create the stacked LSTM model and train it using the shuffled train set
        '''
        self.model = tf.keras.models.Sequential()
        if self.naive:
            self.model.add(tf.keras.layers.LSTM(20, input_shape = self.xtrain.shape[-2:]))
        else:
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True, input_shape = self.xtrain.shape[-2:]))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True))
        if self.naive is False:
            self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(self.forward_look))

        self.model.compile(optimizer='Adam',
                      loss='mse', metrics=['mse'])
        self.create_p_test_train()
        self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                  validation_data = self.p_test, validation_steps = self.validation_steps,
                  verbose = self.verbose)

    def infer_values(self, xtest, ytest, ts = None):
        '''
        Infer values by using the test set
        :param xtest: test dataset
        :param ytest: actual value dataset
        :param ts: tikcer symbol
        :return: model variables that store predicted data
        '''
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()
        if self.infer_train:
            self.pred_train = []
            self.pred_update_train = []
            self.usetest_train = self.xtrain.copy()
        for i in range(self.values):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.y_pred_update = self.model.predict(self.usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.pred.append(self.y_pred)
            self.pred_update.append(self.y_pred_update)
            self.usetest[np.linspace(i+1,i+self.past_history-1,self.past_history-1,dtype=int),np.linspace(self.past_history-2,0,self.past_history-1,dtype=int),:] =  self.y_pred_update[0]
            if self.infer_train:
                self.y_pred_train = self.model.predict(self.xtrain[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.y_pred_update_train = \
                self.model.predict(self.usetest_train[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.pred_train.append(self.y_pred_train)
                self.pred_update_train.append(self.y_pred_update_train)
                self.usetest_train[np.linspace(i + 1, i + self.past_history - 1, self.past_history - 1, dtype=int),
                np.linspace(self.past_history - 2, 0, self.past_history - 1, dtype=int), :] = self.y_pred_update_train[0]
        self.pred = np.array(self.pred)
        self.pred_update = np.array(self.pred_update)
        self.RMS_error = self.hist.history['val_mse'][-1]
        self.RMS_error_train = self.hist.history['mse'][-1]
        if self.infer_train:
            self.pred = np.array(self.pred)
            self.pred_update = np.array(self.pred_update)
        if self.forward_look > 1:
            self.RMS_error_update = (np.mean(((self.ytest[:self.values - 1, 0, 0] - self.pred_update[1:, 0]) / (
                self.ytest[:self.values - 1, 0, 0])) ** 2)) ** 0.5 / self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(
                    ((self.ytrain[:self.values - 1, 0, 0] - self.pred_update_train[1:, 0]) / (
                        self.ytrain[:self.values - 1, 0, 0])) ** 2)) ** 0.5 / self.batch_size
        else:
            self.RMS_error_update = (np.mean(
                ((self.ytest[:self.values - 1] - self.pred_update[1:]) / (
                self.ytest[:self.values - 1])) ** 2)) ** 0.5 / self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(((self.ytrain[:self.values - 1] - self.pred_update_train[1:]) / (
                    self.ytrain[:self.values - 1])) ** 2)) ** 0.5 / self.batch_size

    def plot_test_values(self):
        '''
        Plot predicted values against actual values
        '''
        plt.figure()
        if self.forward_look>1:
            plt.plot(self.yt[:self.values-1,0,0],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:,0],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:,0],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../outputs/LSTM/MultiStock_prediction_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.figure()
            plt.plot(self.pred[1:, 0]-self.pred_update[1:,0], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../outputs/LSTM/MSDifference_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.clf()
            np.savez('../outputs/LSTM-save_mat/MSstore_%d_%d_%d_%d_%s_%s.png' % (
                self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)),
                     y=self.yt[:self.values - 1, 0, 0], pred=self.pred[1:, 0], pred_up=self.pred_update[1:, 0])
        else:
            plt.plot(self.yt[:self.values-1],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../outputs/LSTM/MultiStock_prediction_%d_%d_%d_%d_%s.png'%(
            self.depth,int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.figure()
            plt.plot(self.pred[1:] - self.pred_update[1:], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../outputs/LSTM/MSDifference_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.clf()
            np.savez('../outputs/LSTM-save_mat/MSstore_%d_%d_%d_%d_%s_%s.png' % (
                self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)),
                     y=self.yt[:self.values - 1], pred=self.pred[1:], pred_up=self.pred_update[1:])
        print('The relative test RMS error is %f'%self.RMS_error)
        print('The relative test RMS error for the updated dataset is %f' % self.RMS_error_update)
        if self.infer_train:
            print('The relative train RMS error is %f' % self.RMS_error_train)
            print('The relative train RMS error for the updated dataset is %f' % self.RMS_error_update_train)

    def full_workflow(self, model = None):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()
        if model is None:
            self.ts = self.tickerSymbol
        else:
            self.xt = model.xtest
            self.yt = model.ytest
            self.ts = model.tickerSymbol
        if self.sameTickerTestTrain == True:
            self.ts = 'Ensemble'

        self.infer_values(self.xt, self.yt, self.ts)

    def model_workflow(self):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()

    def prepare_test(self):
        '''
        Create the dataset from the extracted time-series data
        '''
        training_size = int(self.ytemp.size * self.train_test_split)
        training_mean = self.ytemp[:training_size].mean()  # get the average
        training_std = self.ytemp[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
        self.ytemp = (self.ytemp - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
        data, target = self.data_preprocess(self.ytemp, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest, self.ytest = data, target

    def get_tick_values(self):
        '''
        Get ticker values in a list
        '''
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        self.ytemp = data.values

    def prepare_workflow(self):
        self.get_tick_values()
        self.prepare_test()

    def full_workflow_and_plot(self, model = None):
        '''
        Workflow to carry out the entire process end-to-end
        :param model: Choose which model to use to predict inferred values
        :return:
        '''
        self.full_workflow(model = model)
        self.plot_test_values()

    def plot_bot_decision(self):
        '''
        calculate investment growth from the inferred prediction value and plot the resulting growth
        '''
        if self.forward_look > 1:
            ideal = self.yt[:self.values - 1, 0, 0]
            pred = np.asarray(self.pred[1:, 0]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:, 0]).reshape(-1,)
        else:
            ideal = self.yt[:self.values - 1]
            pred = np.asarray(self.pred[1:]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:]).reshape(-1,)
        control_ideal = get_control_vector(ideal)
        control_pred = get_control_vector(pred)
        control_pred_update = get_control_vector(pred_update)
        bot_ideal = buy_and_sell_bot(ideal, control_ideal)
        bot_pred = buy_and_sell_bot(ideal, control_pred)
        bot_pred_update = buy_and_sell_bot(ideal, control_pred_update)
        plt.figure()
        plt.plot(bot_ideal, label='Ideal case (%.2f)'%bot_ideal[-1])
        plt.plot(bot_pred, label='From prediction (%.2f)'%bot_pred[-1])
        plt.plot(bot_pred_update, label='From prediction (updated) (%.2f)'%bot_pred_update[-1])
        plt.plot(ideal / ideal[0] * 100.0, label='Stock value(%s)' % self.ts)
        plt.xlabel("Days")
        plt.ylabel("Percentage growth")
        plt.legend()
        plt.savefig('../outputs/LSTM/MSBot_prediction_%d_%d_%d_%d_%s.png' % (self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
        np.savez('../outputs/LSTM-save_mat/MSbot_%d_%d_%d_%d_%s_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)),
                 ideal=bot_ideal, pred=bot_pred, pred_up=bot_pred_update)
        plt.clf()
