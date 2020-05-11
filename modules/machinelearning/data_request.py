# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:42:34 2020

@author: Brayden, Morgan
"""

import requests
import pandas as pd

def getData(statistic = 'Deaths'):

    #Get the JSON data from the Covid API
    response = requests.get("https://covidapi.info/api/v1/global/count")
    json = response.json()['result']

    #Convert the JSON data to a pandas dataframe
    response_data = pd.DataFrame.from_dict(json)

    #Transpose the dataframe and reset the index so timestamp is now index
    df = pd.DataFrame(data=response_data).T
    df = df.reset_index(level=None, drop=False, col_level=0)
    
    #Set the new column names
    df.columns = ['Date Time', 'Confirmed', 'Deaths', 'Recovered']
    
    #calculate the concurrent number of infections
    df['Concurrent'] = df.apply(lambda x: (x['Confirmed'] - x['Recovered'] - x['Deaths']), axis=1)
    
    #calculate the rates
    df['Death_Rate'] = df['Deaths'].diff();
    df['Confirmed_Rate'] = df['Confirmed'].diff();
    df['Recovered_Rate'] = df['Recovered'].diff();
    df['Concurrent_Rate'] = df['Concurrent'].diff();
    
    #set the first value of these rates to 0, as they are NaN
    df['Death_Rate'][0] = 0;
    df['Confirmed_Rate'][0] = 0;
    df['Recovered_Rate'][0] = 0;
    df['Concurrent_Rate'][0] = 0;

    #Make a new data frame with the inputted statistic
    uni_data = pd.DataFrame();

    uni_data[statistic] = df[statistic]
    uni_data.index = df['Date Time']

    return uni_data