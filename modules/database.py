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
    
def getCartesianValuesAsDict(statistic, values):
    
    data = {}
    
    #Start the min / max at positive and negative infinity
    maxRange = -np.inf
    minRange = np.inf
    
    #iterate the dataframes rows by index
    for index, row in values.iterrows():
        
        value = row[statistic]
        
        #check for the maximum value
        if(value > maxRange):
            maxRange = value
        
        #check for the minimum value
        if(value < minRange):
            minRange = value
            
        #store the time and value as cartesian coords in a dictionary
        data[index] = {
            'x' : index,
            'y' : value
        }
        
    return data, minRange, maxRange
        
def addGlobal(statistic, scope, values, predictions):
    
    globalRef = db.reference(path='statistics/{}/{}/'.format(scope, str(statistic).lower()))
    
    #get the max domain as the last preditions timestamp
    maxDomain = predictions.index[-1]
    #get the minimum domain from the first values timestamp
    minDomain = values.index[0]
    
    lastCurrentValueDate = values.index[-1]
    
    valueData, valueMinRange, valueMaxRange = getCartesianValuesAsDict(statistic, values)
    predictionData, predictionMinRange, predictionMaxRange = getCartesianValuesAsDict('Prediction', predictions)
    
    maxRange = valueMaxRange
    minRange = valueMinRange
    
    if(predictionMaxRange > maxRange):
        maxRange = predictionMaxRange
    
    if(predictionMinRange < minRange):
        minRange = predictionMinRange
        
    rangeData = {
        'minRange': minRange,
        'maxRange': maxRange
    }
    
    domainData = {
        'minDomain': minDomain,
        'maxDomain': maxDomain
    }
    
    lastPredictions = globalRef.child('predictions').order_by_key().get()
    
    """
    historyNode = globalRef.child('history').get()
    if(historyNode != None):
        print('History Node cotains data')
    """
    
    #Store the old predictions
    
    lastValues = globalRef.child('values').order_by_key().get()
    lastHistoricValueDate = lastValues.popitem()[0]
    
    globalRef.child('history').set({
        '{}'.format(lastHistoricValueDate) : lastPredictions
    })
    
    globalRef.child('lastUpdated').set(lastCurrentValueDate)
    
    #Add the values and predictions to the database
    globalRef.child('values').set(valueData)
    globalRef.child('predictions').set(predictionData)
    
    #Add the range and domain data to the database
    globalRef.child('range').set(rangeData)
    globalRef.child('domain').set(domainData)
    
    