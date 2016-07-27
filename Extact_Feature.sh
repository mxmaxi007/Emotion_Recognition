#!/bin/bash

for file in /home/maxi/Emotion/data/wav/*
do
	name=${file##*/}
	name=${name%.*}
	SMILExtract -C /home/maxi/Downloads/openSMILE-2.1.0/config/emobase.conf -I ${file} -O /home/maxi/Emotion/data/Features/${name}.csv
done


