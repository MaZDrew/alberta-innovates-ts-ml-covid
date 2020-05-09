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

n_input = 7
n_features = 1
n_input_layers = 200

data = data_request.getData();

modelLSTM = model.createSimpleModelLSTM(n_input_layers, n_input, n_features)

trainData = model.trainSimpleLSTM(modelLSTM, data, n_input)

predictions = model.makePrediction(
    modelLSTM, data, trainData[0], trainData[1],
    n_input, n_features
)

database.addGlobal('Deaths', predictions, n_input)