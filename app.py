# -- coding: utf-8 --
"""
Created on Sat Apr 11 11:47:12 2020

@author: Brayden, Morgan
    
"""

from modules.machinelearning import data_request
from modules.machinelearning import model
from modules import database

import warnings
warnings.filterwarnings("ignore")

prediction_window = 7
history_window = 14

n_epochs = 5
n_input_layers = 200
batch_size = 2
n_features = 1

n_loops = 5

#statistics = ['Deaths','Confirmed','Recovered','Concurrent','Death_Rate','Confirmed_Rate','Recovered_Rate']
statistics = ['Deaths','Confirmed','Recovered','Concurrent','Death_Rate','Confirmed_Rate','Recovered_Rate']

simpleLinearModelLSTM = model.createSimpleModelLSTM(n_input_layers, n_features, history_window, False, 'relu')
simpleRateModelLSTM = model.createSimpleModelLSTM(n_input_layers, n_features, history_window, True, 'tanh')

currentModel = simpleLinearModelLSTM

for statistic in statistics:
    
    data = data_request.getData(statistic)
    
    if(statistic.find('_Rate') is not -1):
        currentModel = simpleRateModelLSTM
    
    trainData = model.trainSimpleLSTM(
        currentModel,
        data,
        history_window,
        batch_size,
        n_epochs
    )

    predictions = model.makePrediction(
        statistic, currentModel, data,
        trainData[0], trainData[1],
        n_features, prediction_window, history_window
    )
    
    val = predictions.head(len(predictions) - prediction_window)
    pred = predictions.tail(prediction_window)
    
    database.addGlobal(statistic, val, pred)