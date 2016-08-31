#! /bin/bash

clf=2
val=$1
test_type=$2

#python Emo_Classifier_Double.py ../../data/Features/ Double_GA_Logistic_0/ Global/Global_GA_Logistic_0.txt $clf $val
#python Emo_Classifier_Global.py ../../data/Features/ Global/Global_GA_Logistic_0.txt $clf $val
python Emo_Classifier_NN.py ../../data/Features/ $val $test_type
