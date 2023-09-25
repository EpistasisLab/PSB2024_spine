#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:42:10 2023

@author: aorlenko
"""
import pandas as pd 
import operator
import numpy as np
import collections
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict

result_path = '/result_path/'
r=15
#this specify the location of raw result from batch internal
result_path = "/clustering_results/"
dataFiles = [(result_path + f) for f in listdir(result_path) if isfile(join(result_path, f))]
methods_to_remove=[]
dd=defaultdict(list)
for i in dataFiles:
    df= pd.read_csv(i, header=None)
    xx=df[df.columns[0]].value_counts().tolist()
    dd[i.split("/")[-1].split(".")[0]].append(xx)  
    thr =0.01*len(df[df.columns[0]])
    if min(xx)<thr:
        mtr=i.split("/")[-1].split(".")[0]
        methods_to_remove.append(mtr)
pdd = pd.DataFrame.from_dict(dd).T
pdd=pdd.sort_index()
pdd.to_csv('./cluster_size_dist.csv', header=False)
validation_result_path = "/validation/"
validation_files = [os.path.join(validation_result_path, file) for file in os.listdir(validation_result_path) if os.path.isfile(os.path.join(validation_result_path,file))]
# Please check the validation indeices results and enter the column number of the indices list found at the end
df_all=pd.DataFrame()
row_counter = 0
for i in validation_files:
    for j in methods_to_remove:
        if str(j) in (i):
            validation_files.remove(i)            

for num, path in enumerate(validation_files):
        name = path.split("internal")[0].split("/")[-1]
        df= pd.read_csv(path)
        df=df.set_index(df.columns[1])  
        df_all[name]=df[name]
df_all = df_all.reindex(sorted(df_all.columns), axis=1)
df_all.drop('CVNN', inplace=True)
df_all.drop('Dunn', inplace=True)
df_all.drop('SD', inplace=True)
df_all.drop('SDb_w', inplace=True)
weights = {
			"Sil": 1,
			"Db": -1,
			"Xb": -1,
			"CH": 1,
			"I": 1,
            }

#sort merged dataframe by index alphabetically
df_all = df_all.sort_index()
# sort weight dictionary alphabetically
od = collections.OrderedDict(sorted(weights.items()))
wl=list(od.values()) 	
#apply weight to sorted dataframe
df_all2=df_all.apply(lambda x: np.asarray(x) * np.asarray(wl)) 
# list of r scores
rank=sorted(list(range(1,r+1)),reverse=True)
# the rest will be zeroes
z_list = [0] * (len(df_all.columns)-r)
rank =rank+z_list
dff = df_all2.T
cols_ori=dff.columns
rank_cols=[]
for i in df_all.index:
    
    dff = dff.sort_values(by=[i], ascending=False)
    dff[i+'_rank']=dff[i].rank(method='max')  
    ll = len(dff[i+'_rank'].unique())
    #ranks for metrics with duplicates
    if ll < len(dff.index):
            #check if  number of unique scores is less than rank range
            if ll< r:
                rank_1 = sorted(list(range(r-ll+1,r+1)),reverse=True)
                dff[i+'_rank'].replace(dff[i+'_rank'].unique(), rank_1, inplace=True)

            else:
                rank_2 = sorted(list(range(1,r+1)),reverse=True)
                z_list2 = [0] * (ll-r)
                rank_2 =rank_2+z_list2
                dff[i+'_rank'].replace(dff[i+'_rank'].unique(), rank_2, inplace=True)        
    else:
        dff[i+'_rank']=rank
    rank_cols.append(i+'_rank')
dff['sum'] =dff[rank_cols].sum(axis=1)
dff = dff.sort_values(by=['sum'], ascending=False)
dff[df_all.T.columns]=df_all.T[df_all.T.columns]
dff.to_csv('/out/')


    