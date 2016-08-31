#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, linear_model
from sklearn.naive_bayes import GaussianNB
from deap import base
from deap import creator
from deap import tools
from scoop import futures, shared

multi_process=1;
classifier_type=1; #1:Logistic Regression; 2:SVM; 3:Naive Bayes
validation_type=0;
Validation_Dict={0:("03", "08"), 1:("09", "10"), 2:("11", "13"), 3:("12", "14"), 4:("15", "16")};
Emo_Dict={0:"Neutral", 1:"Anger", 2:"Boredom", 3:"Disgust", 4:"Fear", 5:"Happiness", 6:"Sadness"};
Feature_Dict=dict();
creator.create("FitnessMax", base.Fitness, weights=(1.0, ));
creator.create("Individual", list, fitness=creator.FitnessMax);

    
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
        
def Load_Feature(dir_path):
    f_dir=os.listdir(dir_path);
    #feature_matrix=np.array(0);
    #label_vector=np.array(0);
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
            if file_name[0:2]==Validation_Dict[validation_type][0] or file_name[0:2]==Validation_Dict[validation_type][1]:
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
    elif classifier_type==3:
        clf=GaussianNB();
    X_new=X[:, individual];
    return np.asscalar(Classifier(clf, X_new, Y)), ;

def Feature_Select_GA(X, Y):
    feature_num=len(Feature_Dict);
    #creator.create("FitnessMax", base.Fitness, weights=(1.0, ));
    #creator.create("Individual", list, fitness=creator.FitnessMax);

    IND_SIZE=50;
    POP_SIZE=100;
    NGEN=300;
    CXPB=0.8
    MUTPB=0.1

    toolbox=base.Toolbox();
    toolbox.register("attr_int", random.randint, 0, feature_num-1);
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE);
    toolbox.register("population", tools.initRepeat, list, toolbox.individual);

    toolbox.register("evaluate", Evaluate, X, Y);
    toolbox.register("mate", tools.cxTwoPoint);
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=feature_num-1, indpb=0.05);
    toolbox.register("select", tools.selTournament, tournsize=3);
    
    if multi_process==1:
        toolbox.register("map",  futures.map);
    
    start=time.time();
    pop=toolbox.population(POP_SIZE);
    
    if multi_process==1:
        fitness=list(toolbox.map(toolbox.evaluate, pop));
    else:
        fitness=list(map(toolbox.evaluate, pop));

    for ind, fit in zip(pop, fitness):
        ind.fitness.values=fit;

    pre_avg=0;
    for g in range(NGEN):
        print("-- Generation {} --".format(g+1));
        #offspring=toolbox.select(pop, POP_SIZE);
        pop=toolbox.select(pop, POP_SIZE);
        offspring=list(map(toolbox.clone, pop));
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random()<CXPB:
                toolbox.mate(child1, child2);
                del child1.fitness.values;
                del child2.fitness.values;

        for mutant in offspring:
            if random.random()<MUTPB:
                toolbox.mutate(mutant);
                del mutant.fitness.values;

        invalid_ind=[ind for ind in offspring if not ind.fitness.valid];
        if multi_process==1:
            fitness=list(toolbox.map(toolbox.evaluate, invalid_ind));
        else:
            fitness=list(map(toolbox.evaluate, invalid_ind));
         
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values=fit;

        #pop[:]=offspring;
        pop.extend(invalid_ind);
    
        # Gather all the fitnesses in one list and print the stats
        fits=[ind.fitness.values[0] for ind in pop];
        
        length = len(pop);
        mean = sum(fits) / length;
        sum2 = sum(x*x for x in fits);
        std = abs(sum2 / length - mean**2)**0.5;
        
        end=time.time();
        print("  Time {}s".format(end-start));
        print("  Min {}" .format(min(fits)));
        print("  Max {}" .format(max(fits)));
        print("  Avg {}" .format(mean));
        print("  Std {}" .format(std));
        
        if abs(mean-pre_avg)<1e-5 and std<1e-5:
            break;
        pre_avg=mean;
        
    pop=toolbox.select(pop, POP_SIZE);
    return pop;

def Single_Revelance(X, Y):
    for i in range(7):
        Y_new=np.zeros(Y.size, dtype=int);
        for j in range(Y.size):
            if Y[j]==i:
                Y_new[j]=1;
                
        pop=Feature_Select_GA(X, Y_new);
        # Gather all the fitnesses in one list and print the stats
        fits=[ind.fitness.values[0] for ind in pop];
        length = len(pop);
        mean = sum(fits) / length;
        sum2 = sum(x*x for x in fits);
        std = abs(sum2 / length - mean**2)**0.5;
        
        if classifier_type==1:
            file_path="Single_GA_Logistic/"+str(validation_type)+Emo_Dict[i]+".txt";
        elif classifier_type==2:
            file_path="Single_GA_SVM/"+str(validation_type)+Emo_Dict[i]+".txt";
        elif classifier_type==3:
            file_path="Single_GA_NB/"+str(validation_type)+Emo_Dict[i]+".txt";
        feature_file=open(file_path, 'w');
        
        feature_file.write("Min {}\n" .format(min(fits)));
        feature_file.write("Max {}\n" .format(max(fits)));
        feature_file.write("Avg {}\n" .format(mean));
        feature_file.write("Std {}\n" .format(std));
        
        for ind in pop:
            feature_ind=list();
            for feature_label in ind:
                feature_ind.append(Feature_Dict[feature_label]);
            feature_file.write(str(ind.fitness.values[0])+' '+' '.join(feature_ind)+'\n');
        feature_file.close();
        print("***  {} Finished  ***\n".format(Emo_Dict[i]));

def Double_Revelance(X, Y):
    for i in range(7):
        for j in range(i+1, 7):
            valid_list=list();
            for k in range(Y.size):
                if Y[k]==i or Y[k]==j:
                    valid_list.append(k);
                
            pop=Feature_Select_GA(X[valid_list], Y[valid_list]);
            # Gather all the fitnesses in one list and print the stats
            fits=[ind.fitness.values[0] for ind in pop];
            length = len(pop);
            mean = sum(fits) / length;
            sum2 = sum(x*x for x in fits);
            std = abs(sum2 / length - mean**2)**0.5;
        
            if classifier_type==1:
                file_path="Double_GA_Logistic_"+str(validation_type)+"/"+Emo_Dict[i]+'_' +Emo_Dict[j]+".txt";
            elif classifier_type==2:
                file_path="Double_GA_SVM_"+str(validation_type)+"/"+Emo_Dict[i]+'_' +Emo_Dict[j]+".txt";
            elif classifier_type==3:
                file_path="Double_GA_NB_"+str(validation_type)+"/"+Emo_Dict[i]+'_' +Emo_Dict[j]+".txt";
            feature_file=open(file_path, 'w');
        
            feature_file.write("Min {}\n" .format(min(fits)));
            feature_file.write("Max {}\n" .format(max(fits)));
            feature_file.write("Avg {}\n" .format(mean));
            feature_file.write("Std {}\n" .format(std));
        
            for ind in pop:
                feature_ind=list();
                for feature_label in ind:
                    feature_ind.append(Feature_Dict[feature_label]);
                feature_file.write(str(ind.fitness.values[0]) + ' ' + ' '.join(feature_ind) + '\n');
            feature_file.close();
            
            print("***  {}_{} Finished  ***\n".format(Emo_Dict[i], Emo_Dict[j]));
 
def Global_Revelance(X, Y):
    pop=Feature_Select_GA(X, Y);
    # Gather all the fitnesses in one list and print the stats
    fits=[ind.fitness.values[0] for ind in pop];
    length = len(pop);
    mean = sum(fits) / length;
    sum2 = sum(x*x for x in fits);
    std = abs(sum2 / length - mean**2)**0.5;
     
    if classifier_type==1:
        file_path="Global/Global_GA_Logistic_"+str(validation_type)+".txt";
    elif classifier_type==2:
        file_path="Global/Global_GA_SVM_"+str(validation_type)+".txt";
    elif classifier_type==3:
        file_path="Global/Global_GA_NB_"+str(validation_type)+".txt";
    feature_file=open(file_path, 'w');
        
    feature_file.write("Min {}\n" .format(min(fits)));
    feature_file.write("Max {}\n" .format(max(fits)));
    feature_file.write("Avg {}\n" .format(mean));
    feature_file.write("Std {}\n" .format(std));
        
    for ind in pop:
        feature_ind=list();
        for feature_label in ind:
            feature_ind.append(Feature_Dict[feature_label]);
        feature_file.write(str(ind.fitness.values[0]) + ' ' + ' '.join(feature_ind) + '\n');
    feature_file.close();
            
    print("*** Global Finished  ***\n");
    
    
if __name__=="__main__":
    if len(sys.argv)!=4:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_dir classifier_type(1:Logistic Regression, 2:SVM, 3:Naive Bayes) validation_type(0-4)\n");
        sys.exit(2);
    
    start=time.time();
    dir_path=sys.argv[1];
    classifier_type=int(sys.argv[2]);
    validation_type=int(sys.argv[3]);
    
    X_train, Y_train, X_test, Y_test=Load_Feature(dir_path);

    #Single_Revelance(X_train, Y_train);
    Double_Revelance(X_train, Y_train);
    Global_Revelance(X_train, Y_train);
    
    end=time.time();
    print("Total Time {}s".format(end-start));

    
    
