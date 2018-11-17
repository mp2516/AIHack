# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:21:28 2018

@author: arabi
"""
import csv
import os
import pandas as pd


path = r'D:\Hackathon\Challenges\California\california\train' #change it to your local path 
directory = os.listdir(path)




def get_code(feature=None):
    a=0
    with open(path + '\\' + 'BG_METADATA_2016.csv') as csvfile:
        info = csv.DictReader(csvfile)
        for row in info:
            if row['Full_Name'] == feature: #identifies the short code for the feature we want
                a = row['Short_Name'] 
    csvfile.close()             
    return (a)

def get_data(subset = None):
    actual_data =[]
    info = pd.read_csv(path + '\\' + subset)
    df = pd.DataFrame(info)
    ide = get_code('SEX BY AGE: Total: Total population -- (Estimate)')
    for i in df.columns:
        if i == ide:
            actual_data = df[i].values
            #print (actual_data)        
            #print (len(actual_data))
    return (actual_data)

get_code('SEX BY AGE: Total: Total population -- (Estimate)')
get_data('X01_AGE_AND_SEX.csv')