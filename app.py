# -- coding: utf-8 --
"""
Created on Sat Apr 11 11:47:12 2020

@author: Brayden, Morgan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.machinelearning import data_request

from pandas.tseries.offsets import DateOffset

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

import warnings
warnings.filterwarnings("ignore")

df = data_request.getData();

### Show the original data set
print(df);

train = df

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)

n_input = 7
n_features = 1
generator = TimeseriesGenerator(train, train, length = n_input, batch_size = 2)

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape = (n_input, n_features)))
model.add(Dense(1))

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss = 'mse')

history = model.fit_generator(generator, epochs = 20, verbose = 1)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.scatter(x=hist['epoch'],y=hist['loss'])
plt.show()

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

