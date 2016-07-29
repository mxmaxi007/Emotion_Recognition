#/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import time
import re
import os
import copy

if __name__=="__main__":
    if len(sys.argv)!=3:
        print(sys.argv);
        print("Usage: python " + sys.argv[0] + " feature_file1 feature_file2\n");
        sys.exit(2);
    
    feature_file1=sys.argv[1];
    feature_file2=sys.argv[2];
    
    f1=open(feature_file1, 'r');
    line_list=f1.readlines();
    f1.close();
    
    all_f1_set=set();
    for line in line_list[4:]:
        line=line.strip();
        line_split=re.split(" +|\t+", line);
        all_f1_set=all_f1_set | set(line_split[1:]);

    f2=open(feature_file2, 'r');
    line_list=f2.readlines();
    f2.close();
    
    all_f2_set=set();
    for line in line_list[4:]:
        line=line.strip();
        line_split=re.split(" +|\t+", line);
        all_f2_set=all_f2_set | set(line_split[1:]);
    
    same_f_set=all_f1_set & all_f2_set;
    
    all_f1_set_len=len(all_f1_set);
    all_f2_set_len=len(all_f2_set);
    same_f_set_len=len(same_f_set);
    
    print("Number of Same Features: {}".format(same_f_set_len));
    print("Number of Feature 1: {}".format(all_f1_set_len));
    print("Similar Ratio(Feature 1): {}".format(float( same_f_set_len)/all_f1_set_len));
    print("Number of Feature 2: {}".format(all_f2_set_len));
    print("Similar Ratio(Feature 2): {}".format(float(same_f_set_len)/all_f2_set_len));
