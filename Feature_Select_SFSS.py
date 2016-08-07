#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, linear_model

classifier_type=1;
Emo_Dict={0:"Neutral", 1:"Anger", 2:"Boredom", 3:"Disgust", 4:"Fear", 5:"Happiness", 6:"Sadness"};
Feature_Dict=dict();
Feature_Num=50;

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
    k_fold=cross_validation.KFold(len(X), n_folds=3);
    score_list=cross_validation.cross_val_score(clf, X, Y, cv=k_fold, n_jobs=-1);
    return sum(score_list)/len(score_list);

def Evaluate(X, Y, individual):
    if len(individual)==0:
        return 0;
    if classifier_type==1:
        clf=linear_model.LogisticRegression();
    elif classifier_type==2:
        clf=svm.LinearSVC();
    X_new=X[:, individual];
    return np.asscalar(Classifier(clf, X_new, Y)), ;

def Seq_Float_Forw_Select(X, Y, features, max_k, print_steps=False):
    # Initialization
    feat_sub = []
    k = 0
 
    start=time.time();
    while True:
        # Step 1: Inclusion
        if print_steps:
            print("Inclusion from features")
            print(features)
        if len(features) > 0:
            crit_func_max = Evaluate(X, Y, feat_sub + [features[0]])
            best_feat = features[0]
            if len(features)>1:
                for x in features[1:]:
                    crit_func_eval = Evaluate(X, Y, feat_sub + [x])
                    if crit_func_eval > crit_func_max:
                        crit_func_max = crit_func_eval
                        best_feat = x
            features.remove(best_feat)
            feat_sub.append(best_feat)
            if print_steps:
                print("include: {} ->; feature_subset: {}".format(best_feat, feat_sub))
 
        # Step 2: Conditional Exclusion
            worst_feat_val = None
            if len(features) + len(feat_sub) > max_k:
                crit_func_max = Evaluate(X, Y, feat_sub) 
                for i in reversed(range(0,len(feat_sub))):
                    crit_func_eval = Evaluate(X, Y, feat_sub[:i] + feat_sub[i+1:])
                    if crit_func_eval > crit_func_max:
                        worst_feat, crit_func_max = i, crit_func_eval
                        worst_feat_val = feat_sub[worst_feat]
                if worst_feat_val:        
                    del feat_sub[worst_feat]
            if print_steps:
                print("exclude: {} ->; feature subset: {}".format(worst_feat_val, feat_sub))
        
        end=time.time();
        print("Time: {}s".format(end-start));
        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break
 
    return feat_sub
    
def Feature_Select_SFFS(X, Y):
    feature_num=len(Feature_Dict);
    feature_list=range(0, feature_num);
    return Seq_Float_Forw_Select(X, Y, feature_list, Feature_Num, True);
    
def Single_Revelance(X, Y):
    for i in range(7):
        Y_new=np.zeros(Y.size, dtype=int);
        for j in range(Y.size):
            if Y[j]==i:
                Y_new[j]=1;
                
        feature_list=Feature_Select_SFFS(X, Y_new);
        if classifier_type==1:
            file_path="Single_SFFS_Logistic/"+Emo_Dict[i]+".txt";
        elif classifier_type==2:
            file_path="Single_SFFS_SVM/"+Emo_Dict[i]+".txt";
        feature_file=open(file_path, 'w');
        
        fit=Evaluate(X, Y_new, feature_list)
        feature_file.write("Min {}\n" .format(fit));
        feature_file.write("Max {}\n" .format(fit));
        feature_file.write("Avg {}\n" .format(fit));
        feature_file.write("Std {}\n" .format(0));     
            
        feature_name_list=list();
        for feature_label in feature_list:
            feature_name_list.append(Feature_Dict[feature_label]);
        feature_file.write(str(fit)+' '+' '.join(feature_name_list)+'\n');
        feature_file.close();
        
        print("***  {} Finished  ***\n".format(Emo_Dict[i]));

def Double_Revelance(X, Y):
    for i in range(7):
        for j in range(i+1, 7):
            valid_list=list();
            for k in range(Y.size):
                if Y[k]==i or Y[k]==j:
                    valid_list.append(k);
                
            feature_list=Feature_Select_SFFS(X[valid_list], Y[valid_list]);
            if classifier_type==1:
                file_path="Double_SFFS_Logistic/"+Emo_Dict[i]+".txt";
            elif classifier_type==2:
                file_path="Double_SFFS_SVM/"+Emo_Dict[i]+'_' +Emo_Dict[j]+".txt";
            feature_file=open(file_path, 'w');
            
            fit=Evaluate(X[valid_list], Y[valid_list], feature_list);
            feature_file.write("Min {}\n" .format(fit));
            feature_file.write("Max {}\n" .format(fit));
            feature_file.write("Avg {}\n" .format(fit));
            feature_file.write("Std {}\n" .format(0));       
            
            feature_name_list=list();
            for feature_label in feature_list:
                feature_name_list.append(Feature_Dict[feature_label]);
            feature_file.write(str(fit)+' '+' '.join(feature_name_list)+'\n');
            feature_file.close();
            
            print("***  {}_{} Finished  ***\n".format(Emo_Dict[i], Emo_Dict[j]));
            
def Global_Revelance(X, Y):
    feature_list=Feature_Select_SFFS(X, Y);
    if classifier_type==1:
        file_path="Global/Global_SFFS_Logistic.txt";
    elif classifier_type==2:
        file_path="Global/Global_SFFS_SVM.txt";
    feature_file=open(file_path, 'w');
    
    fit=Evaluate(X, Y, feature_list)
    feature_file.write("Min {}\n" .format(fit));
    feature_file.write("Max {}\n" .format(fit));
    feature_file.write("Avg {}\n" .format(fit));
    feature_file.write("Std {}\n" .format(0));
    
    feature_name_list=list();
    for feature_label in feature_list:
        feature_name_list.append(Feature_Dict[feature_label]);
    feature_file.write(str(fit)+' '+' '.join(feature_name_list)+'\n');
    feature_file.close();
            
    print("***  Global Finished  ***\n");
            
def Load_Feature(dir_path):
    f_dir=os.listdir(dir_path);
    X_train=np.array(0);
    Y_train=np.array(0);
    X_test=np.array(0);
    Y_test=np.array(0);
    flag=0;
    
    for file_name in f_dir:
        file_path=os.path.join(dir_path, file_name);
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
                    Feature_Dict[i]=temp_split[1];
                    i+=1;
                flag=1;
    #print(Feature_Dict);

    print("Feature Number: {}".format(len(Feature_Dict)));
    print("Train Sample Number: {}".format(Y_train.size));
    print("Train Sample Number: {}".format(Y_test.size));
    return X_train, Y_train, X_test, Y_test;
    
if __name__=="__main__":
    if len(sys.argv)!=2:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_dir\n");
        sys.exit(2);
    
    start=time.time();
    dir_path=sys.argv[1];
    
    X_train, Y_train, X_test, Y_test=Load_Feature(dir_path);

    #Single_Revelance(X_train, Y_train);
    Double_Revelance(X_train, Y_train);
    Global_Revelance(X_train, Y_train);
    
    end=time.time();
    print("Total Time {}s".format(end-start));

