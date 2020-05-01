# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:42:34 2020

@author: Brayden
"""

import requests
import pandas as pd

def getData():
    
    response = requests.get("https://covidapi.info/api/v1/global/count")
    json = response.json()['result']
    
    
    
    df = pd.DataFrame.from_dict(json)
    
    
    
    #for key, value in data.iteritems() :
        #cleaned_data.append({'timestamp': key, 'deaths': value['deaths']})