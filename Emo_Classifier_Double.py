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
    
def Train_Classifier(feature_sel_dir_path, X, Y, clf_list_list, clf_feature_list_list):
    for i in range(Emo_Num):
        clf_list=list();
        clf_feature_list=list();
        for j in range(i+1, Emo_Num):
            file_name=Emo_Dict[i]+"_"+Emo_Dict[j]+".txt";
            file_path=os.path.join(feature_sel_dir_path, file_name);
            fp=open(file_path, 'r');
            line_list=fp.readlines();
            fp.close();
    
            feature_set=set();
            for line in line_list[4:]:
                line=line.strip();
                line_split=re.split(" +|\t+", line);
                feature_set=feature_set | set(line_split[1:]);
                
            feature_list=list();
            for feature_name in list(feature_set):
                feature_list.append(Feature_Dict[feature_name]);
            feature_list.sort();

            sample_list=list();
            for k in range(len(Y)):
                if Y[k]==i or Y[k]==j:
                    sample_list.append(k);
    
            clf_feature_list.append(feature_list);
            if classifier_type==1:
                clf_base=linear_model.LogisticRegression(random_state=1);
            elif classifier_type==2:
                clf_base=svm.LinearSVC(random_state=1);
            clf_base.fit(X[np.ix_(sample_list, feature_list)], Y[sample_list]);
            clf_list.append(clf_base);
            
            #print("***  {}_{}  ***".format(i, j));
            #print("Feature Numbers: {}".format(len(clf_feature_list_list[i][j-i-1])));
            #print("Sample Numbers: {}".format(len(sample_list)));
            
        clf_feature_list_list.append(clf_feature_list);
        clf_list_list.append(clf_list);

def Judge_Winner(result_matrix, reco_vector):
    best_result=0;
    max_vote=0;
    for i in range(Emo_Num):
        if reco_vector[i]>max_vote:
            best_result=i;
            max_vote=reco_vector[i];
        elif reco_vector[i]==max_vote:
            best_result=result_matrix[i][best_result];
    return best_result;
    
def Predict_Result(X, Y, clf_list_list, clf_feature_list_list):
    result_list=list();
    for x, y in zip(X, Y):
        result_matrix=np.zeros([7, 7], dtype=int);
        reco_vector=np.zeros(7);
        for i in range(Emo_Num):
            for j in range(i+1, Emo_Num):
                result=clf_list_list[i][j-i-1].predict(x[clf_feature_list_list[i][j-i-1]].reshape(1, -1));
                reco_vector[result]+=1;
                result_matrix[i][j]=result;
                result_matrix[j][i]=result;
        result_list.append(Judge_Winner(result_matrix, reco_vector));
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
        print("Usage: python " + sys.argv[0] + " feature_dir feature_sel_dir\n");
        sys.exit(2);

    start=time.time();
    feature_dir_path=sys.argv[1];
    feature_sel_dir_path=sys.argv[2];
    
    X_train, Y_train, X_test, Y_test=Load_Feature(feature_dir_path);
    
    clf_list_list=list();
    clf_feature_list_list=list();
    Train_Classifier(feature_sel_dir_path, X_train, Y_train, clf_list_list, clf_feature_list_list);
    Predict_Result(X_test, Y_test, clf_list_list, clf_feature_list_list);
    
    
   
    
    

