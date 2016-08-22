#/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, linear_model
import tensorflow as tf

classifier_type=2;
Emo_Dict={0:"Neutral", 1:"Anger", 2:"Boredom", 3:"Disgust", 4:"Fear", 5:"Happiness", 6:"Sadness"};
Feature_Dict=dict();
Emo_Num=7;

def Judge_Label(file_name):
    if file_name[5]=='N':
        return 0;
    elif file_name[5]=='W':
        return 1;
    elif file_name[5]=='L':
        return 2;
    elif file_name[5]=='E':
        return 3; 
    elif file_name[5]=='A':
        return 4;
    elif file_name[5]=='F':
        return 5;
    elif file_name[5]=='T':
        return 6; 
        
def Load_Feature(feature_dir_path):
    f_dir=os.listdir(feature_dir_path);
    #feature_matrix=np.array(0);
    #label_vector=np.array(0);
    X_train=np.array(0);
    Y_train=np.array(0);
    X_test=np.array(0);
    Y_test=np.array(0);
    flag=0;
    
    for file_name in f_dir:
        file_path=os.path.join(feature_dir_path, file_name);
        if os.path.isfile(file_path) and re.match(".*.csv", file_name):
            fp=open(file_path, "r");
            line_list=fp.readlines();
            line=line_list[-1];
            fp.close();
            line=line.strip();
            line_split=re.split(",", line);
            line_split_len=len(line_split);
            vector=np.zeros((line_split_len-3));
            n=0;
            
            for value in line_split[2:-1]:
                vector[n]=np.float64(value);
                n+=1;
                
            if file_name[0:2]=="03" or file_name[0:2]=="08":
                if X_test.ndim==0:
                    X_test=vector.copy();
                    Y_test=Judge_Label(file_name);
                else:
                    X_test=np.vstack([X_test, vector]);
                    Y_test=np.hstack([Y_test, Judge_Label(file_name)]);
                    
            else:
                if X_train.ndim==0:
                    X_train=vector.copy();
                    Y_train=Judge_Label(file_name);
                else:
                    X_train=np.vstack([X_train, vector]);
                    Y_train=np.hstack([Y_train, Judge_Label(file_name)]);
                
            if flag==0:
                i=0;
                for temp in line_list[4:-5]:
                    temp=temp.strip();
                    temp_split=re.split(" ", temp);
                    Feature_Dict[temp_split[1]]=i;
                    i+=1;
                flag=1;
    #print(Feature_Dict);

    print("Feature Number: {}".format(len(Feature_Dict)));
    print("Train Sample Number: {}".format(Y_train.size));
    print("Test Sample Number: {}".format(Y_test.size));
    return X_train, Y_train, X_test, Y_test;
   
def Result_Analysis(target_list, predict_list):
    print("***  Target Label  ***");
    print(target_list);
    print("***  Predict Label  ***");
    print(predict_list);

    confuse_mat=np.zeros([Emo_Num, Emo_Num]);
    for i in range(Emo_Num):
        for j in range(Emo_Num):
            for k in range(len(target_list)):
                if target_list[k]==i and predict_list[k]==j:
                    confuse_mat[i, j]+=1;
    for i in range(Emo_Num):
        sum_num=confuse_mat[i, :].sum()
        confuse_mat[i, :]/=sum_num;
    print("Confusion Matrix:");
    print(confuse_mat);

    right_num=0;
    for target, predict in zip(target_list, predict_list):
        if predict==target:
            right_num+=1;
    print("Accuracy: {}".format(float(right_num)/len(target_list)));
 
def Neural_Network_Single(X_train, Y_train, X_test, Y_test):
    classifier=tf.contrib.learn.DNNClassifier(hidden_units=[50], n_classes=7);
    classifier.fit(x=X_train, y=Y_train, steps=1000);
     
    result_list=classifier.predict(X_test);
    print("***  Single  ***");
    Result_Analysis(Y_test, result_list);
    
def Train_Classifier(X, Y, clf_list_list):
    for i in range(Emo_Num):
        clf_list=list();
        for j in range(i+1, Emo_Num):
            sample_list=list();
            for k in range(len(Y)):
                if Y[k]==i or Y[k]==j:
                    sample_list.append(k);
            clf_base=tf.contrib.learn.DNNClassifier(hidden_units=[50], n_classes=7);
            clf_base.fit(x=X[sample_list], y=Y[sample_list], steps=1000);
            clf_list.append(clf_base);
            print("***  "+Emo_Dict[i]+"_"+Emo_Dict[j]+" Finished  ***");
        clf_list_list.append(clf_list);

def Judge_Winner(reco_matrix, reco_vector):
    best_result=0;
    max_vote=0;
    for i in range(Emo_Num):
        if reco_vector[i]>max_vote:
            best_result=i;
            max_vote=reco_vector[i];
        elif reco_vector[i]==max_vote:
            best_result=reco_matrix[i][best_result];
    return best_result;
    
def Predict_Result(X, Y, clf_list_list):
    result_list=list();
    result_matrix=np.array(0, dtype=int);
    indice_list=list();
    for i in range(Emo_Num):
        for j in range(i+1, Emo_Num):
            result=clf_list_list[i][j-i-1].predict(X);
            indice_list.append((i, j));
            if result_matrix.ndim==0:
                result_matrix=result.copy();
            else:
                result_matrix=np.vstack([result_matrix, result]);
                
    result_matrix=np.transpose(result_matrix);
    for i in range(Y.size):
        reco_matrix=np.zeros([Emo_Num, Emo_Num], dtype=int);
        reco_vector=np.zeros(Emo_Num);
        for j in range(len(indice_list)):
            reco_vector[result_matrix[i, j]]+=1;
            reco_matrix[indice_list[j][0], indice_list[j][1]]=result_matrix[i, j];
            reco_matrix[indice_list[j][1], indice_list[j][0]]=result_matrix[i, j];
        result_list.append(Judge_Winner(reco_matrix, reco_vector));
   
    print("***  Double  ***"); 
    Result_Analysis(Y, result_list);    
    
if __name__=="__main__":
    if len(sys.argv)!=2:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_dir\n");
        sys.exit(2);

    start=time.time();
    feature_dir_path=sys.argv[1];

    X_train, Y_train, X_test, Y_test=Load_Feature(feature_dir_path);
    
    #clf_list_list=list();
    #Train_Classifier(X_train, Y_train, clf_list_list);
    #Predict_Result(X_test, Y_test, clf_list_list);
    
    Neural_Network_Single(X_train, Y_train, X_test, Y_test);
    
    end=time.time();
    print("Total Time {}s".format(end-start));
