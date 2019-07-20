# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 13:56:49 2019

@author: Bokkin Wang
"""
import os
import json
import re
import sys
import pickle as pkl
import difflib
from copy import deepcopy
import pandas as pd
from itertools import islice   #导入迭代器
import time  
import numpy as np  
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath('__file__'))))
  

def random_lab(num,kind):
    mex = np.random.rand(num, kind)
    index = np.argmax(mex, axis=1)
    mex = np.zeros((num, kind), np.float32)
    for i in list(range(num)):
        mex[i][index[i]] = 1.0
    return mex
 
# label propagation  
def labelPropagation( df_prob,num_classes=10, max_iter = 2, tol = 20):
    # initialize  
    affinity_matrix = df_prob.values
    
    num_samples = affinity_matrix.shape[0]                                   #样本数量

    label_function = random_lab(num_samples,num_classes)              #随机
            
    # start to propagation  
    iter = 0
    pre_label_function = np.zeros((num_samples, num_classes), np.float32)  #生成原始矩阵
    
    changed = np.abs(pre_label_function - label_function).sum()            #记录标签改变，记录收敛
    while iter < max_iter and changed > tol:  
        if iter % 1 == 0:  
            print ("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))  
        pre_label_function = label_function  
        iter += 1  
          
        # propagation  
        label_function = np.dot(affinity_matrix, label_function)    #更新标签矩阵
          
        # check converge  
        changed = np.abs(pre_label_function - label_function).sum()  #计算改变
      
    # get terminate label of unlabeled data  
    label_data_labels = np.zeros(num_samples)  
    for i in range(num_samples):  
        label_data_labels[i] = np.argmax(label_function[i])  #返回标签
      
    return label_data_labels  


#def labelPropagation1(Mat_Label, Mat_Unlabel, affinity_matrix, labels, kernel_type = 'rbf', rbf_sigma = 1.5,\
#                      knn_num_neighbors = 10, max_iter = 500, tol = 1e-3):  
#    # initialize  
#    num_label_samples = Mat_Label.shape[0]  
#    num_unlabel_samples = Mat_Unlabel.shape[0]  
#    num_samples = num_label_samples + num_unlabel_samples  
#    labels_list = np.unique(labels)  
#    num_classes = len(labels_list)  
#      
#    MatX = np.vstack((Mat_Label, Mat_Unlabel))  
#    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)  
#    for i in range(num_label_samples):  
#        clamp_data_label[i][labels[i]] = 1.0  
#      
#    label_function = np.zeros((num_samples, num_classes), np.float32)  
#    label_function[0 : num_label_samples] = clamp_data_label  
#    label_function[num_label_samples : num_samples] = random_lab(num_unlabel_samples,num_classes) 
#      
#    # graph construction  
#    
#      
#    # start to propagation  
#    iter = 0; pre_label_function = np.zeros((num_samples, num_classes), np.float32)  
#    changed = np.abs(pre_label_function - label_function).sum()  
#    while iter < max_iter and changed > tol:  
#        if iter % 1 == 0:  
#            print ("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))  
#        pre_label_function = label_function  
#        iter += 1  
#          
#        # propagation  
#        label_function = np.dot(affinity_matrix, label_function)  
#          
#          
#        # check converge  
#        changed = np.abs(pre_label_function - label_function).sum()  
#      
#    # get terminate label of unlabeled data  
#    label_data_labels = np.zeros(num_samples)  
#    for i in range(num_samples):  
#        label_data_labels[i] = np.argmax(label_function[i])  
#      
#    return label_data_labels  
#
## label propagation  
#def labelPropagation2(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 1.5,\
#                      knn_num_neighbors = 10, max_iter = 500, tol = 1e-3):  
#    # initialize  
#    num_label_samples = Mat_Label.shape[0]  
#    num_unlabel_samples = Mat_Unlabel.shape[0]  
#    num_samples = num_label_samples + num_unlabel_samples  
#    labels_list = np.unique(labels)  
#    num_classes = len(labels_list)  
#      
#    MatX = np.vstack((Mat_Label, Mat_Unlabel))  
#    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)  
#    for i in range(num_label_samples):  
#        clamp_data_label[i][labels[i]] = 1.0  
#      
#    label_function = np.zeros((num_samples, num_classes), np.float32)  
#    label_function[0 : num_label_samples] = clamp_data_label  
#    label_function[num_label_samples : num_samples] = random_lab(num_unlabel_samples,num_classes)   
#      
#    # graph construction  
#    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)  
#      
#    # start to propagation  
#    iter = 0; pre_label_function = np.zeros((num_samples, num_classes), np.float32)  
#    changed = np.abs(pre_label_function - label_function).sum()  
#    while iter < max_iter and changed > tol:  
#        if iter % 1 == 0:  
#            print ("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))  
#        pre_label_function = label_function  
#        iter += 1  
#          
#        # propagation  
#        label_function = np.dot(affinity_matrix, label_function)  
#          
#        # clamp  
#        label_function[0 : num_label_samples] = clamp_data_label  
#          
#        # check converge  
#        changed = np.abs(pre_label_function - label_function).sum()  
#      
#    # get terminate label of unlabeled data  
#    unlabel_data_labels = np.zeros(num_unlabel_samples)  
#    for i in range(num_unlabel_samples):  
#        unlabel_data_labels[i] = np.argmax(label_function[i+num_label_samples])  
#      
#    return unlabel_data_labels