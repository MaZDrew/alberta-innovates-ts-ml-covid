# -- coding: utf-8 --
"""
Created on Sat Apr 11 11:47:12 2020

@author: Brayden, Morgan
    
"""

from modules.machinelearning import data_request
from modules.machinelearning import model
from modules import database
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

prediction_window = 7
history_window = 7

n_epochs = 5
n_input_layers = 100
batch_size = 2
n_features = 1

n_loops = 50

#statistics = ['Deaths','Confirmed','Recovered','Concurrent','Death_Rate','Confirmed_Rate','Recovered_Rate']
statistics = ['Death_Rate']

for statistic in statistics:
    
    currentModel = None
    all_predictions = pd.DataFrame();
    data = data_request.getData(statistic)
    
    if(statistic.find('_Rate') is not -1):
        currentModel = model.createSimpleRateModel(n_input_layers, n_features, history_window)
    else:
        currentModel = model.createSimpleLinearModel(n_input_layers, n_features, history_window)
    
    for i in range(0, n_loops):
        
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
        
        all_predictions[i] = pred['Prediction']
        
        print('{} cycle {}'.format(statistic, i))
    
    print(all_predictions)
    
    dfm_predict = pd.DataFrame(
        all_predictions.mean(axis=1),
        index=all_predictions.index,
        columns=['Prediction']
    )
    
    dfm_predict = dfm_predict.round(0)
    
    print(dfm_predict);
    
    #database.addGlobal(statistic, val, dfm_predict)