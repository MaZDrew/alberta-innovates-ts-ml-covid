# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:42:34 2020

@author: Brayden
"""

import requests
import pandas as pd

def getData(univariate = 'Deaths'):
    
    response = requests.get("https://covidapi.info/api/v1/global/count")
    json = response.json()['result']
      
    response_data = pd.DataFrame.from_dict(json)
    
    df = pd.DataFrame(data=response_data).T
    df = df.reset_index(level=None, drop=False, col_level=0)
    df.columns = ['Date Time', 'Confirmed', 'Deaths', 'Recovered']
    
    uni_data = df[univariate]
    uni_data.index = df['Date Time']
    
    return uni_data



