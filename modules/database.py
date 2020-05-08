# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:50:30 2020

@author: Morgan
"""

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL':'https://ml-covid.firebaseio.com/'
})


def addGlobal(statistic, x_value, y_value):
    globalref = db.reference(path='/global/{}/'.format(statistic))
    return ''
