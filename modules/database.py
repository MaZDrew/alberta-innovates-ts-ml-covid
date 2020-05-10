# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:50:30 2020

@author: Brayden, Morgan
"""

import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

if not firebase_admin._apps:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL':'https://ml-covid.firebaseio.com/'
    })
    
def getCartesianPredictionsAsDict(predictions):
    
    data = {}
    
    maxRange = -np.inf
    
    for index, row in predictions.iterrows():
        
        prediction = row['Prediction']
        
        if(prediction > maxRange):
            maxRange = prediction
        
        data[index] = {
            'x' : index,
            'y' : prediction
        }
        
    return [data, maxRange]

def getCartesianValuesAsDict(statistic, values):
    
    data = {}
    
    maxRange = -np.inf
    
    for index, row in values.iterrows():
        
        value = row[statistic]
        
        if(value > maxRange):
            maxRange = value
            
        data[index] = {
            'x' : index,
            'y' : value
        }
        
    return [data, maxRange]
        
def addGlobal(statistic, values, predictions):
    
    globalRef = db.reference(path='statistics/global/{}/'.format(str(statistic).lower()))
    
    maxDomain = predictions.index[-1]
    minDomain = values.index[0]
    
    value = getCartesianValuesAsDict(statistic, values)
    prediction = getCartesianPredictionsAsDict(predictions)
    
    valueData = value[0]
    predictionData = prediction[0]
    
    valueMaxRange = value[1]
    predictionMaxRange = prediction[1]
    
    maxRange = valueMaxRange
    
    if(predictionMaxRange > maxRange):
        maxRange = predictionMaxRange
        
    globalRef.child('values').set(valueData)
    globalRef.child('predictions').set(predictionData)
    globalRef.child('maxRange').set(maxRange)
    globalRef.child('maxDomain').set(maxDomain)
    globalRef.child('minDomain').set(minDomain)
    