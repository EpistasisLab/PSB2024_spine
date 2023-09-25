# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:03:58 2017
@author: thy1995
MODIFED PATHS FOR DEBUGGING ON 8/29/2019 BY ZJR
Nodiefied By aorlenko on 7/21/2023
"""

import os
from os.path import isfile, join
from os import listdir
import numpy as np
import csv
from fileOP import writeRows
import internal_validation
from sklearn.metrics import silhouette_score
from resultOP import table_result
import pandas as pd

#file path to the file
cwd = os.getcwd()
result_path = "./"
# Specify the location of input data file with the features.
dataFolder =  result_path + "/set/"
# specify the location of clustering results csv files
labelFolder = result_path + "/clustering_results/"  # (pimaSPectral.csv)	 (mixture_model)   (add data to the name)
# This is the location of the output of the raw cluster validation result
internalFolder = result_path + "/validation/"

import os.path

dataFiles = [(dataFolder + f) for f in listdir(dataFolder) if isfile(join(dataFolder, f))]
labelFiles = [(labelFolder + f) for f in listdir(labelFolder) if isfile(join(labelFolder, f))]


counter = 0
for data_file_name in dataFiles:
    data_peak = pd.read_csv(data_file_name, header=None)  
    num_cols = len(data_peak.columns)
    num_rows = len(data_peak)    
    data = np.zeros([num_rows, num_cols])  
    with open(data_file_name) as csvfile:
        row_index = 0
        reader = csv.reader(csvfile)
        r2= pd.read_csv(data_file_name)
        new_row=r2[r2.columns[0]]
        for row in reader:
            for cols_index in range(num_cols):
                data[row_index][cols_index] = row[cols_index]
            row_index += 1
    targets=[]
    for i in range(len(labelFiles)):
        lF = labelFiles[i]
        targets.append(lF)
    target = data_file_name.split("/")[-1].split(".csv")[0]
    scatL = []
    distL = []
    comL = []
    sepL = []

    for label_file_name in targets:
        print("current label", label_file_name)
        name = label_file_name.split(".")[0].split("/")[-1]
        exist = os.path.exists(internalFolder + name + "_internal1.csv")
        if exist:
            continue
        data_peak = pd.read_csv(label_file_name, header=None)
        num_cols = len(data_peak.columns)
        num_rows = len(data_peak)
        label = np.zeros([num_rows, num_cols])  
        with open(label_file_name) as csvfile:
            row_index = 0
            reader= csv.reader(csvfile)
            r2= pd.read_csv(label_file_name)
            new_row=r2[r2.columns[0]]
            for row in reader:
                for cols_index in range(num_cols):
                    label[row_index][cols_index]= row[cols_index]

                row_index+=1
        
        label = label.T        
        for d_column in label:
            num_k = np.unique(d_column)
            inter_index = internal_validation.internalIndex(len(num_k))
            scat , dis = inter_index.SD_valid(data, d_column)
            com , sep = inter_index.CVNN(data, d_column)
            scatL.append(scat)
            distL.append(dis)
            comL.append(com)
            sepL.append(sep)
        result_over_k = []
        for i in range(len(label)):
            d_column = label[i]
            num_k = np.unique(d_column)
            result = []
            inter_index = internal_validation.internalIndex(len(num_k))
            result.append(silhouette_score(data, d_column, metric = 'euclidean'))
            result.append(inter_index.dbi(data, d_column))
            result.append(inter_index.xie_benie(data, d_column))
            result.append(inter_index.dunn(data, d_column))
            result.append(inter_index.CH(data, d_column))
            result.append(inter_index.I(data, d_column))
            result.append(inter_index.SD_valid_n(scatL, distL, i))
            result.append(inter_index.SDbw(data, d_column))
            result.append(inter_index.CVNN_n(comL, sepL, i))           
            result_over_k.append(result)
        to_export = np.array(result_over_k).T
        name = label_file_name.split(".")[0].split("/")[-1]
        to_export = table_result(to_export,[[name]] ,[['','Sil', 'Db', 'Xb', 'Dunn', 'CH', "I", "SD", "SDb_w", "CVNN"]])
        writeRows(internalFolder + name + "internal.csv" , to_export)
    counter = counter + 1
