# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.tseries.offsets import DateOffset

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


#from modules import database

def createSimpleModelLSTM(n_input_layers, n_input, n_features, dropout=False):
    
    model = Sequential()
    model.add(LSTM(n_input_layers, activation='relu', input_shape = (n_input, n_features)))
    
    if(dropout == True):
        model.add(Dropout(0.15))
        
    model.add(Dense(1))
    
    return model
    
def trainSimpleLSTM(model, data, n_input):
    
    print(data)
    
    train = data
    
    scaler = MinMaxScaler()
    scaler.fit(train)
    
    train = scaler.transform(train)
    
    generator = TimeseriesGenerator(train, train, length = n_input, batch_size = 2)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss = 'mse')
    
    history = model.fit_generator(generator, epochs = 20, verbose = 1)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.scatter(x=hist['epoch'],y=hist['loss'])
    plt.show()
    
    return [train, scaler]   
    
def makePrediction(model, df, train, scaler, n_input, n_features):
    
    pred_list = []
    
    batch = train[-n_input:].reshape((1, n_input, n_features))
    
    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:,1:,:], [[pred_list[i]]], axis=1)
    
    ### Create a list with the next week of dates from our last date entry
    add_dates = [pd.to_datetime(df.index[-1]) + DateOffset(days=x) for x in range(0, n_input + 1)]
    
    ### Filter out the time from the newly added dates
    add_dates = [str(add_dates[d]).split(' ')[0] for d in range(0, len(add_dates))]
    
    future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)
    
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=future_dates[-n_input:].index, columns=['Prediction'])
    df_predict = df_predict.round(0)
    
    df_proj = pd.concat([df, df_predict], axis=1)
    
    ### Print the prediction for the next week
    print(df_proj.tail(n_input))
    
    ### Graph the predictions
    plt.scatter(x=df_proj.index, y=df_proj['Deaths'])
    plt.scatter(x=df_proj.index, y=df_proj['Prediction'])
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100], rotation=45)
    plt.show()
    
    return df_proj