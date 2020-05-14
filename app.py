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

#How many days in the future we want to make our prediction
prediction_window = 7
#How many days are we looking at in the past to make our prediction
history_window = 21
#The number of iterations to train with to create an averaged out the prediction
n_iters = 100

n_epochs = 5
batch_size = 2
n_features = 1
n_input_layers = 100

statistics = ['Deaths','Confirmed','Recovered','Concurrent','Death_Rate','Confirmed_Rate','Recovered_Rate', 'Concurrent_Rate']
#scopes = ['global','CAN', 'USA']

#statistics = ['Death_Rate']
scopes = ['CAN', 'USA']

for scope in scopes:
    
    #Loop through all of our statistics
    for statistic in statistics:
        
        currentModel = None
        
        #create a dataframe we can store a reference of all the predictions
        all_predictions = pd.DataFrame()
        
        #Grab the data from the Covid API
        data = data_request.getData(statistic, scope)
        
        #if we are using rate statistic, use a rate model otherwise linear
        if(statistic.find('_Rate') is not -1):
            currentModel = model.createSimpleRateModel(n_input_layers, n_features, history_window)
        else:
            currentModel = model.createSimpleLinearModel(n_input_layers, n_features, history_window)
        
        #how many times we are iterating to average the out the prediction values
        for i in range(0, n_iters):
            
            #Train the model
            trainingData, scaler = model.trainSimpleLSTM(
                currentModel,
                data,
                history_window,
                batch_size,
                n_epochs
            )
        
            #Make a prediction
            results = model.makePrediction(
                statistic, currentModel, data,
                trainingData, scaler,
                n_features, prediction_window, history_window
            )
            
            #Grab the values from our prediction results
            values = results.head(len(results) - prediction_window)
            predictions = results.tail(prediction_window)
            
            #Add the newest set of predictions
            all_predictions[i] = predictions['Prediction']
            print('[{}] {} cycle {}'.format(scope, statistic, i))
        
        #Make a new dataframe that is the mean of all our predictions
        df_predict_mean = pd.DataFrame(
            all_predictions.mean(axis=1),
            index=all_predictions.index,
            columns=['Prediction']
        )
        
        #Round all of the mean's of the predicted values
        df_predict_mean = df_predict_mean.round(0)
        print(df_predict_mean)
        
        #Store the results in the database
        database.addGlobal(statistic, scope, values, df_predict_mean)