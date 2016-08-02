#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, linear_model

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

def Classifier(clf, X, Y):
    clf.fit(X, Y);
    #print(clf.predict(X[-1:]));
    #k_fold=cross_validation.KFold(len(X), n_folds=3);
    #X_folds=np.array_split(X, 3);
    #Y_folds=np.array_split(Y, 3);
    #score=list();
    #for train_indices, test_indices in k_fold:
    #    score.append(clf.fit(X[train_indices], Y[train_indices]).score(X[test_indices], Y[test_indices]));
    #print(score);
    #score_list=cross_validation.cross_val_score(clf, X, Y, cv=k_fold, n_jobs=-1);
    #return sum(score_list)/len(score_list);
    
def Train_Classifier(X, Y, clf_list_list, clf_feature_list_list):
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
            #clf_base=linear_model.LogisticRegression(random_state=1);
            clf_base=svm.SVC(random_state=1);
            Classifier(clf_base, X[np.ix_(sample_list, feature_list)], Y[sample_list]);
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
                #print(i, j);
                #print(len(clf_feature_list_list[i][j-i-1]));
                result=clf_list_list[i][j-i-1].predict(x[clf_feature_list_list[i][j-i-1]].reshape(1, -1));
                reco_vector[result]+=1;
                result_matrix[i][j]=result;
                result_matrix[j][i]=result;
        result_list.append(Judge_Winner(result_matrix, reco_vector));
    print(Y);
    print(result_list);
    right_num=0;
    for predict, target in zip(result_list, Y):
        if predict==target:
            right_num+=1;
    print("Accuracy: {}".format(float(right_num)/len(result_list)));
    
if __name__=="__main__":
    if len(sys.argv)!=3:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_dir feature_sel_dir\n");
        sys.exit(2);

    start=time.time();
    feature_dir_path=sys.argv[1];
    feature_sel_dir_path=sys.argv[2];
    
    f_dir=os.listdir(feature_dir_path);
    feature_matrix=np.array(0);
    label_vector=np.array(0);
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
                
            if(feature_matrix.ndim==0):
                feature_matrix=vector.copy();
                label_vector=Judge_Label(file_name);
            else:
                feature_matrix=np.vstack([feature_matrix, vector]);
                label_vector=np.hstack([label_vector, Judge_Label(file_name)]);
                
            if flag==0:
                i=0;
                for temp in line_list[4:-5]:
                    temp=temp.strip();
                    temp_split=re.split(" ", temp);
                    Feature_Dict[temp_split[1]]=i;
                    i+=1;
                flag=1;
    #print(Feature_Dict);

    print("Feature Numbers: {}".format(len(Feature_Dict)));
    print("Sample Numbers: {}".format(label_vector.size));
    
    clf_list_list=list();
    clf_feature_list_list=list();
    Train_Classifier(feature_matrix[:400], label_vector[:400], clf_list_list, clf_feature_list_list);
    Predict_Result(feature_matrix[401:], label_vector[401:], clf_list_list, clf_feature_list_list);
     
    
    

