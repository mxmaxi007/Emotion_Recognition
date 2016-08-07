#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, linear_model

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
    
def Train_Classifier(feature_file_path, X, Y, clf, feature_list):
    fp=open(feature_file_path, 'r');
    line_list=fp.readlines();
    fp.close();
    
    feature_set=set();
    for line in line_list[4:]:
        line=line.strip();
        line_split=re.split(" +|\t+", line);
        feature_set=feature_set | set(line_split[1:]);
                
    for feature_name in list(feature_set):
        feature_list.append(Feature_Dict[feature_name]);
    feature_list.sort();

    clf.fit(X[:, feature_list], Y);
            
    #print("***  {}_{}  ***".format(i, j));
    #print("Feature Numbers: {}".format(len(clf_feature_list_list[i][j-i-1])));
    #print("Sample Numbers: {}".format(len(sample_list)));
            
    
def Predict_Result(X, Y, clf, feature_list):
    result_list=clf.predict(X[:, feature_list]);
    
    print("***  Target Label  ***");
    print(Y);
    print("***  Predict Label  ***");
    print(result_list);
    right_num=0;
    for predict, target in zip(result_list, Y):
        if predict==target:
            right_num+=1;
    print("Accuracy: {}".format(float(right_num)/len(result_list)));
    
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
    
if __name__=="__main__":
    if len(sys.argv)!=3:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_dir feature_sel_file\n");
        sys.exit(2);

    start=time.time();
    feature_dir_path=sys.argv[1];
    feature_sel_file_path=sys.argv[2];
    
    X_train, Y_train, X_test, Y_test=Load_Feature(feature_dir_path);

    feature_list=list();
    if classifier_type==1:
        clf=linear_model.LogisticRegression(random_state=1);
    elif classifier_type==2:
        clf=svm.LinearSVC(random_state=1);
    Train_Classifier(feature_sel_file_path, X_train, Y_train, clf, feature_list);
    Predict_Result(X_test, Y_test, clf, feature_list);
    
     
    
    
