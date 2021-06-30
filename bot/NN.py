import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
import yfinance as yf
import pandas as pd
import numpy as np


class NN(keras.layers.Layer):

    tickers_list = ['MMM', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'XOM', 'JPM', 'GS', 'HD', 'INTC', 'IBM', 'JNJ',
                    'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'DIS']
    window = 10

    prediction = None

    def __init__(self):
        super(NN, self).__init__()

        self.model = Sequential()
        self.model.add(SimpleRNN(50, return_sequences=True, input_shape=(self.window, len(self.tickers_list))))

        for i in range(100):
            self.model.add(SimpleRNN(5, return_sequences=True))
            self.model.add(Activation('relu'))

        self.model.add(SimpleRNN(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(len(self.tickers_list)))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.load_weights("weights_deep_model_sru_relu_dema.h5")

    def fit(self):
        df = pd.DataFrame(columns=self.tickers_list)
        for ticker in self.tickers_list:
            df[ticker] = yf.download(ticker, period=(self.window + 1).__str__() + 'd')['Adj Close']

        Y = np.array([df[-1:].values/df[-2:-1].values])

        X = np.array([df[:-1].values])

        self.model.fit(X, Y)

    def predict(self):
        df = pd.DataFrame(columns=self.tickers_list)
        for ticker in self.tickers_list:
            df[ticker] = yf.download(ticker, period=self.window.__str__()+'d')['Adj Close']

        X = np.array([df.values])

        self.prediction = pd.DataFrame(self.model.predict([X]), columns=self.tickers_list).sort_values(by=0, axis=1,
                                                                                                       ascending=False)

    def get_prediction(self):
        return self.prediction.T.to_string(header=False, justify='center')

    def get_best(self):
        return self.prediction.T[:1].to_string(header=False, justify='center')

    def get_worst(self):
        return self.prediction.T[-1:].to_string(header=False, justify='center')

    def get_stocks(self):
        return ', '.join(self.tickers_list)
