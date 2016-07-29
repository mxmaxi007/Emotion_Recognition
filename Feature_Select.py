#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import random

import numpy as np
from sklearn import svm, cross_validation, cluster, linear_model
from deap import base
from deap import creator
from deap import tools
from scoop import futures,  shared

multi_process=1;
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
        
def Classifier(clf, X, Y):
    #print(clf.fit(X[:-100], Y[:-100]).score(X[-100:], Y[-100:]));
    #print(clf.predict(X[-1:]));
    k_fold=cross_validation.KFold(len(X), n_folds=3);
    #X_folds=np.array_split(X, 3);
    #Y_folds=np.array_split(Y, 3);
    #score=list()
    #for train_indices, test_indices in k_fold:
    #    score.append(clf.fit(X[train_indices], Y[train_indices]).score(X[test_indices], Y[test_indices]));
    #print(score);
    score_list=cross_validation.cross_val_score(clf, X, Y, cv=k_fold, n_jobs=-1);
    return sum(score_list)/len(score_list);

def Cluster(clt, X, Y):
    clt.fit(X);
    print(clt.labels_[::10]);
    print(Y[::10]);    

def Evaluate(X, Y, individual):
    #clf=svm.SVC(gamma=0.001, C=100);
    clf=linear_model.LogisticRegression();
    X_new=X[:, individual];
    return np.asscalar(Classifier(clf, X_new, Y)), ;

def Feature_Select(X, Y):
    feature_num=len(Feature_Dict);
    #creator.create("FitnessMax", base.Fitness, weights=(1.0, ));
    #creator.create("Individual", list, fitness=creator.FitnessMax);

    IND_SIZE=100;
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
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05);
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
        #print("  Length {}".format(length));
        print("  Min {}" .format(min(fits)));
        print("  Max {}" .format(max(fits)));
        print("  Avg {}" .format(mean));
        print("  Std {}" .format(std));
        
    pop=toolbox.select(pop, POP_SIZE);
    return pop;

def Single_Revelance(X, Y):
    for i in range(7):
        Y_new=np.zeros(Y.size, dtype=int);
        for j in range(Y.size):
            if Y[j]==i:
                Y_new[j]=1;
                
        pop=Feature_Select(X, Y_new);
        # Gather all the fitnesses in one list and print the stats
        fits=[ind.fitness.values[0] for ind in pop];
        length = len(pop);
        mean = sum(fits) / length;
        sum2 = sum(x*x for x in fits);
        std = abs(sum2 / length - mean**2)**0.5;
        
        feature_file=open("Single/"+Emo_Dict[i]+".txt", 'w');
        
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
                
            pop=Feature_Select(X[valid_list], Y[valid_list]);
            # Gather all the fitnesses in one list and print the stats
            fits=[ind.fitness.values[0] for ind in pop];
            length = len(pop);
            mean = sum(fits) / length;
            sum2 = sum(x*x for x in fits);
            std = abs(sum2 / length - mean**2)**0.5;
        
            feature_file=open("Double/"+Emo_Dict[i] + '_' + Emo_Dict[j] + ".txt", 'w');
        
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

if __name__=="__main__":
    if len(sys.argv)!=2:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_dir\n");
        sys.exit(2);
    dir_path=sys.argv[1];
    f_dir=os.listdir(dir_path);
    feature_matrix=np.array(0);
    label_vector=np.array(0);
    flag=0;
    start=time.time();
    
    for file_name in f_dir:
        file_path=os.path.join(dir_path, file_name);
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

    Single_Revelance(feature_matrix, label_vector);
    Double_Revelance(feature_matrix, label_vector);
    
    #Feature_Select(feature_matrix, label_vector);
    
    #clf=svm.SVC(gamma=0.001, C=100);
    #print(Classifier(clf, feature_matrix, label_vector));
    
    #clt=cluster.KMeans(n_clusters=7);
    #Cluster(clt, feature_matrix, label_vector);
    
    end=time.time();
    print("Total Time {}s".format(end-start));

    
    
