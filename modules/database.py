# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:50:30 2020

@author: Morgan
"""

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

if not firebase_admin._apps:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL':'https://ml-covid.firebaseio.com/'
    })
    
def getCartesianPredictionsAsDict(predictions):
    
    predictionData = {}
    
    for index, row in predictions.iterrows():
        predictionData[index] = {
            'x':index,
            'y':row['Prediction']
        }
        
    return predictionData

def getCartesianValuesAsDict(statistic, values):
    
    valueData = {}
    
    for index, row in values.iterrows():
        valueData[index] = {
            'x':index,
            'y':row[statistic]
        }
        
    return valueData
        
def addGlobal(statistic, df, n_input):
    
    globalRef = db.reference(path='statistics/global/{}/'.format(str(statistic).lower()))
    
    valuesRef = globalRef.child('values')
    predictionsRef = globalRef.child('predictions')
    
    values = df.head(len(df) - n_input)
    predictions = df.tail(n_input)
    
    valueData = getCartesianValuesAsDict(statistic, values)
    predictionData = getCartesianPredictionsAsDict(predictions)
        
    predictionsRef.set(predictionData)
    valuesRef.set(valueData)