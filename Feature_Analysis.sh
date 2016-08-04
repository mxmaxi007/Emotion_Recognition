#! /bin/bash

for file1 in `ls Double_Logistic/`
do
	for file2 in `ls Double_Logistic/`
	do
		echo $file1 $file2;
		python Feature_Analysis.py Double_Logistic/${file1} Double_Logistic/${file2}
	done
	
done
