# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:21:28 2018

@author: arabi
"""
import csv
import os


path = r'D:\Hackathon\Challenges\California\california\train' #change it to your local path 
directory = os.listdir(path)

def open_file():
    with open(path + '\\' + 'BG_METADATA_2016.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['Full_Name'])
            print (row['Short_Name'])
    return (reader)  

open_file()
