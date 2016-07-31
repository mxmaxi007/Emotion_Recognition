#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, linear_model
from sklearn.ensemble import VotingClassifier

Emo_Dict={0:"Neutral", 1:"Anger", 2:"Boredom", 3:"Disgust", 4:"Fear", 5:"Happiness", 6:"Sadness"};
Feature_Dict=dict();
Emo_Num=7;

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
    
def Judge_Winner(result_list_list, reco_vector):
    best_reco=0;
    max_vote=0;
    equal_list=list();
    for i in range(Emo_Num):
        if reco_vector[i]>max_vote:
            best_reco=i;
            max_vote=reco_vector[i];
        elif reco_vector[i]=max_vote:
            
        
    return Judge_Winner(result_list_list, reco_vector);
    
if __name__=="__main__":
    if len(sys.argv)!=2:
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
                    temp_split=re.split(" ", temp);
                    Feature_Dict[i]=temp_split[1];
                    i+=1;
                flag=1;
    #print(Feature_Dict);

    print("Feature Numbers: {}".format(len(Feature_Dict)));
    print("Sample Numbers: {}".format(label_vector.size));
    
    
    clf_list_list=list();
    for i in range(Emo_Num):
        clf_list=list();
        for j in range(i+1, Emo_Nums):
            file_name=Emo_Dict[i]+"_"+Emo_Dict[j]+".txt";
            file_path=os.path.join(feature_sel_dir_path, file_name);
            fp=open(feature_file1, 'r');
            line_list=f1.readlines();
            fp.close();
    
            feature_set=set();
            for line in line_list[4:]:
                line=line.strip();
                line_split=re.split(" +|\t+", line);
                feature_set=feature_set | set(line_split[1:]);
            feature_list=sorted(list(feature_set));
            
            sample_list=list();
            for k in range(len(label_vector)):
                if label_vector[k]=i or label_vector[k]==j:
                    sample_list.append(k);
                    
            clf_base=LogisticRegression(random_state=1);
            Classifier(clf_base, feature_matrix[sample_list, feature_list], label_vector[sample_list]);
            clf_list.append(clf_base);
        clf_list_list.append(clf_list);
            
    result_list_list=list();
    reco_vector=np.zeros(7);
    for clf_list in clf_list_list:
        result_list=list();
        for clf in clf_list:
            result_vector=clf.predict(feature_matrix);
            for k in result_vector:
                reco_vector[k]+=1;
            result_list.append(result_vector);
        result_list_list.append(result_list);
    
    Judge_Winner(result_list_list, reco_vector);
    
    

