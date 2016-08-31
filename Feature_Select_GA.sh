#! /bin/bash

data_type=3

python -m scoop Feature_Select_GA.py ../../data/Features/ 1 ${data_type} > Output_Info/result_GA_Logistic.txt &
python -m scoop Feature_Select_GA.py ../../data/Features/ 2 ${data_type} > Output_Info/result_GA_SVM.txt &
#python -m scoop Feature_Select_GA.py ../../data/Features/ 3 ${data_type} > Output_Info/result_GA_NB.txt &
wait
echo ${date_type} > result.txt
