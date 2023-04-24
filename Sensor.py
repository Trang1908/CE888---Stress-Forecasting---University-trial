#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Stress_Forecasting:
    
    def __init__(self):
        self.accuracy = None
        self.recall = None
        self._data = None
        self._features = None
        self._target = None
        self._start_val = None
        self._start_test = None
        

    # Define a function to load and process raw data of volunteers and then combine all together
    def wrangle(self, filepath, s):
    # filepath = "https://raw.githubusercontent.com/italha-d/Stress-Predict-Dataset/main/Raw_data/"
    # s: the number from 1 to 35 represents 35 volunteers

        df = pd.DataFrame()      
        # Pre-process ACC
        # Load data from Github
        ACC_path = str(filepath + "S"+ str(s).zfill(2)+"/ACC.csv")
        ACC = pd.read_csv(ACC_path, names=["ACC(x)","ACC(y)","ACC(z)"])
        # Select the initial time (from the first row) and sample rate (from the second row) of ACC
        initial_time_ACC = ACC.iloc[0][0]
        sample_rate_ACC = ACC.iloc[1][0]
        # Drop the "index" column and the first two rows (initial time and sample rate). Then reset all index to the default
        ACC = ACC.iloc[2:].reset_index().drop(columns="index")
        # Compute mean of each axis by the sample rate
        ACC[["ACC(x)_mean", "ACC(y)_mean", "ACC(z)_mean"]] = ACC[["ACC(x)", "ACC(y)", "ACC(z)"]].groupby(np.arange(len(ACC))//sample_rate_ACC).mean()
        # Compute standard deviation of each axis by the sample rate
        ACC[["ACC(x)_std", "ACC(y)_std", "ACC(z)_std"]] = ACC[["ACC(x)", "ACC(y)", "ACC(z)"]].groupby(np.arange(len(ACC))//sample_rate_ACC).std()
        # Compute sum of mean of all axis
        ACC["Sum_ACC_mean"] = ACC["ACC(x)_mean"]+ACC["ACC(y)_mean"]+ACC["ACC(z)_mean"]
        # Compute sum of standard deviation of all axis
        ACC["Sum_ACC_std"] = ACC["ACC(x)_std"]+ACC["ACC(y)_std"]+ACC["ACC(z)_std"]
        # Compute the time corresponding with mean and standard deviation
        ACC["Time (sec)"] = initial_time_ACC + 1 + np.arange(0, ACC.shape[0])
        # Drop columns
        ACC.drop(columns=["ACC(x)","ACC(y)","ACC(z)"], inplace=True)
        # Set "Time (sec)" column as index
        ACC = ACC.set_index("Time (sec)")

        # Pre-process BVP (the explanation for steps is similar to ACC)
        BVP_path = str(filepath + "S"+ str(s).zfill(2)+"/BVP.csv")
        BVP = pd.read_csv(BVP_path, names=["BVP"])
        initial_time_BVP = BVP.iloc[0][0]
        sample_rate_BVP = BVP.iloc[1][0]
        BVP = BVP.iloc[2:].reset_index().drop(columns="index")
        BVP["BVP_mean"] = BVP["BVP"].groupby(np.arange(len(BVP))//sample_rate_BVP).mean()
        BVP["BVP_std"] = BVP["BVP"].groupby(np.arange(len(BVP))//sample_rate_BVP).std()
        BVP["Time (sec)"] = initial_time_BVP + 1 + np.arange(0, BVP.shape[0])
        BVP.drop(columns=["BVP"], inplace=True)
        BVP = BVP.set_index("Time (sec)")

        # Pre-process EDA (the explanation for steps is similar to ACC)
        EDA_path = str(filepath + "S"+ str(s).zfill(2)+"/EDA.csv")
        EDA = pd.read_csv(EDA_path, names=["EDA"])
        initial_time_EDA = EDA.iloc[0][0]
        sample_rate_EDA = EDA.iloc[1][0]
        EDA = EDA.iloc[2:].reset_index().drop(columns="index")
        EDA["EDA_mean"] = EDA["EDA"].groupby(np.arange(len(EDA))//sample_rate_EDA).mean()
        EDA["EDA_std"] = EDA["EDA"].groupby(np.arange(len(EDA))//sample_rate_EDA).std()
        EDA["EDA_max"] = EDA["EDA"].groupby(np.arange(len(EDA))//sample_rate_EDA).max()
        EDA["EDA_min"] = EDA["EDA"].groupby(np.arange(len(EDA))//sample_rate_EDA).min()
        EDA["Time (sec)"] = initial_time_EDA + 1 + np.arange(0, EDA.shape[0])
        EDA.drop(columns=["EDA"], inplace=True)
        EDA = EDA.set_index("Time (sec)")

        # Pre-process HR (the explanation for steps is similar to ACC)
        HR_path = str(filepath + "S"+ str(s).zfill(2)+"/HR.csv")
        HR = pd.read_csv(HR_path, names=["HR"])
        initial_time_HR = HR.iloc[0][0]
        sample_rate_HR = HR.iloc[1][0]
        HR = HR.iloc[2:].reset_index().drop(columns="index")
        HR["Time (sec)"] = initial_time_HR + np.arange(0, HR.shape[0])
        HR = HR.set_index("Time (sec)")


        # Pre-process TEMP (the explanation for steps is similar to ACC)
        TEMP_path = str(filepath + "S"+ str(s).zfill(2)+"/TEMP.csv")
        TEMP = pd.read_csv(TEMP_path, names=["TEMP"])
        initial_time_TEMP = TEMP.iloc[0][0]
        sample_rate_TEMP = TEMP.iloc[1][0]
        TEMP = TEMP.iloc[2:].reset_index().drop(columns="index")
        TEMP["TEMP_mean"] = TEMP["TEMP"].groupby(np.arange(len(TEMP))//sample_rate_TEMP).mean()
        TEMP["Time (sec)"] = initial_time_TEMP + 1 + np.arange(0, TEMP.shape[0])
        TEMP.drop(columns=["TEMP"], inplace=True)
        TEMP = TEMP.set_index("Time (sec)")

        # Combine all above data which have the same "Time (sec)" together 
        combine_data = pd.concat([ACC, BVP, EDA, HR, TEMP], axis=1)
        combine_data.dropna(axis=0, inplace=True) 
        #Note: Dropna here is the quickest way we trim the processed data at the beginning and the end.
        #It doesn't mean we are destroying the serial correlation of our time series.

        # Pre-process LABEL
        time_stamp_path = str(filepath + "S"+ str(s).zfill(2)+"/tags_S"+ str(s).zfill(2)+".csv")
        time_stamp = pd.read_csv(time_stamp_path, names=["Time_stamp"])
        # Return timestamp to the local date and time
        list_date=[]
        for i in time_stamp["Time_stamp"]:
            list_date.append(datetime.datetime.fromtimestamp(i))
        time_stamp["Date"] = list_date

        # Label the whole dataset (1 for stress and 0 for no stress)
        if s==1:
            mask1 = (combine_data.index >= time_stamp["Time_stamp"][0]) & (combine_data.index <= time_stamp["Time_stamp"][1])
            mask2= (combine_data.index >= time_stamp["Time_stamp"][3]) & (combine_data.index <= time_stamp["Time_stamp"][4])
            mask3= (combine_data.index >= time_stamp["Time_stamp"][5]) & (combine_data.index <= time_stamp["Time_stamp"][6])
        else:
            mask1 = (combine_data.index >= time_stamp["Time_stamp"][0]) & (combine_data.index <= time_stamp["Time_stamp"][1])
            mask2= (combine_data.index >= time_stamp["Time_stamp"][2]) & (combine_data.index <= time_stamp["Time_stamp"][3])
            mask3= (combine_data.index >= time_stamp["Time_stamp"][4]) & (combine_data.index <= time_stamp["Time_stamp"][5])
        combine_data["Label"] = mask1 | mask2 | mask3

        combine_data["Label"] = combine_data["Label"].astype(int)
        combine_data["Participant"] = s
        combine_data.reset_index()
        df = pd.concat([df, combine_data], axis=0)
        self._data = df
        self._start_val = df.index.tolist().index(int(time_stamp["Time_stamp"][1]))
        if s==1:
            self._start_test = df.index.tolist().index(int(time_stamp["Time_stamp"][4]))
        else:
            self._start_test = df.index.tolist().index(int(time_stamp["Time_stamp"][3]))
        return df
    

    # Define a function to create a lag feature  
    def create_data(self, lag_length=1):
    # lag_length: the fixed time period

        X = self._data.drop(columns=["Label","Participant"])
        l_X = X.shift(1)
        for i in range(2,lag_length+1):
            l_X = pd.concat([X.shift(i),l_X],axis=1) #Create a lag of X
        l_X.dropna(axis=0, inplace=True)

        y = self._data["Label"]
        y = y.iloc[lag_length:] #y
        self._features = l_X.to_numpy()
        self._target = y
        

    # Define a function for feature scaling with standardization
    def _standardize(self):
        scaler = StandardScaler()
        scaler.fit(self._features)
        standard_features = scaler.transform(self._features)
        return standard_features
        

    # Define a function for reducing dimensionality of dataset
    def _reduce_dimension(self, variance):
        pca = PCA(n_components=variance, svd_solver='full')
        pca.fit(self._features)
        print("Total explained variance: %.2f"%(sum(pca.explained_variance_ratio_)))
        reduced_X = pca.transform(self._features)
        return reduced_X



    # Define a function for prediction by each model
    def get_results(self, step, types="NB", preprocessing="default", variance=0.8):
        
        model = None
        true_positive = 0
        false_negative = 0
        true_negative = 0
        
        #PREPROCESSING:
        if preprocessing == "default":
            None
        elif preprocessing == "standardize":
            self._features = self._standardize()
        elif preprocessing == "reduce_dim":
            self._features = self._standardize()
            self._features = self._reduce_dimension(variance)
        else:
            print("This preprocessing is not supported!")
            return
        
        #MODEL:
        if types == "NB":#Naive Bayes
            model = GaussianNB()
        elif types == "LM":
            model = LogisticRegression(max_iter = 5000)
        elif types == "SVM":
            model = svm.LinearSVC(max_iter = 1000)
        else:
            print("This type is not supported!")
            return
        
        start = 0
        end = 0
        if step == "validation":
            start = self._start_val
            end = len(self._target)-self._start_test
        elif step == "test":
            start = self._start_test+1
            end = len(self._target)-1
        else:
            print("This step is not supported!")
            return
        for i in range(start, end):
            y_pred = model.fit(self._features[:i,:], self._target.iloc[:i]).predict(self._features[i+1,:].reshape(1,len(self._features[i+1,:])))
            true_positive += (int(y_pred)==self._target.iloc[i+1]) & (int(y_pred)==1)
            true_negative += (int(y_pred)==self._target.iloc[i+1]) & (int(y_pred)==0)
            false_negative += (int(y_pred)<self._target.iloc[i+1])
        self.accuracy = (true_positive+true_negative)/(end-start)
        self.recall = true_positive/(true_positive+false_negative)
        

    def __repr__(self):
        return "This is a model for forecasting stress."
