# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:47:12 2020

@author: Brayden, Morgan

Based Off:
    @author: Andrejus Baranovskis
    @github: https://github.com/abaranovskis-redsamurai/automation-repo/tree/master/forecast-lstm
    
"""

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

def createSimpleLinearModel(n_input_layers, n_features, history_window):
    
    model = Sequential()
    model.add(LSTM(n_input_layers, activation='relu', input_shape = (history_window, n_features)))
    model.add(Dense(1))
    
    return model

def createSimpleRateModel(n_input_layers, n_features, history_window):
    
    model = Sequential()
    model.add(LSTM(n_input_layers, activation='tanh', input_shape = (history_window, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    return model
    
def trainSimpleLSTM(model, data, history_window, n_batch, n_epochs):
    
    print(data)
    
    trainingData = data
    
    #Calculate this once
    scaler = MinMaxScaler()
    scaler.fit(trainingData)
    
    trainingData = scaler.transform(trainingData)
    
    generator = TimeseriesGenerator(
        trainingData,
        trainingData,
        length = history_window,
        batch_size = n_batch
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss = 'mse')
    
    history = model.fit_generator(generator, epochs = n_epochs, verbose = 1)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.scatter(x=hist['epoch'],y=hist['loss'])
    plt.show()
    
    return trainingData, scaler
    
def makePrediction(statistic, model, df, train, scaler,
                   n_features, prediction_window, history_window):
    
    pred_list = []
    
    batch = train[-history_window:].reshape((1, history_window, n_features))
    
    for i in range(prediction_window):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:,1:,:], [[pred_list[i]]], axis=1)
    
    ### Append the days we want to predict
    add_dates = [pd.to_datetime(df.index[-1]) + DateOffset(days=x) for x in range(0, prediction_window + 1)]
    
    ### Filter out the time from the newly added dates
    add_dates = [str(add_dates[d]).split(' ')[0] for d in range(0, len(add_dates))]
    
    future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)
    
    df_predict = pd.DataFrame(
        scaler.inverse_transform(pred_list),
        index=future_dates[-prediction_window:].index,
        columns=['Prediction']
    )
    
    df_predict = df_predict.round(0)
    
    df_proj = pd.concat([df, df_predict], axis=1)
    
    ### Print the prediction for the next week
    print(df_proj.tail(prediction_window))
    
    ### Graph the predictions
    plt.scatter(x=df_proj.index, y=df_proj[statistic])
    plt.scatter(x=df_proj.index, y=df_proj['Prediction'])
    plt.xticks([0,10,20,30,40,50,60,70,80,90,100], rotation=45)
    plt.show()
    
    return df_proj