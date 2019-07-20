# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:19:10 2019

@author: Bokkin Wang
"""

import sys
sys.path.append("D:/bigdatahw/pan_paper/code")
from Spectral_method import caleigen 
import numpy as np
from math import log
from sklearn.cluster import KMeans
from copy import deepcopy
import itertools
from collections import Counter
import pickle as pkl
import re
import networkx as nx
from sklearn import datasets
from sklearn import metrics
import warnings
from matplotlib import pyplot as plt
from itertools import cycle, islice
warnings.filterwarnings("ignore")

##D—Score##########################################################################

# for the weakly connected gaint component
def D_Score(A, K_num):
    '''
    input:the adjacency matrix of largest component of network
    output:a label array for each sample
    '''
    
    #SVD for Adjacent matrix
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    u1 = U[:,0]
    v1 = V[:,0]
    
    #set parameter
    n = len(A)
    T_n = log(n) #threshold
    
    #search the support of u1 and v1 
    ob_supp_u1 = [i[0] for i in np.argwhere(u1 == 0)]  #u1
    ob_supp_v1 = [i[0] for i in np.argwhere(v1 == 0)]  #v1
    N1 = list(set(range(n)).difference(set(ob_supp_u1))) 
    N2 = list(set(range(n)).difference(set(ob_supp_v1)))
    
    #calculate R_star_l
    R_star_l = np.zeros((len(A), K_num-1))
    for i in range(K_num-1):
        R_star_l[:,i] = U[:,i+1]/u1
    R_star_l[R_star_l > T_n] = T_n    
    R_star_l[R_star_l < -T_n] = -T_n
    for i in ob_supp_u1:
        R_star_l[i,:] = np.zeros(K_num-1)
        
    #calculate R_star_r
    R_star_r = np.zeros((len(A), K_num-1))
    for i in range(K_num-1):
        R_star_r[:,i] = V[:,i+1]/v1
    R_star_r[R_star_r > T_n] = T_n    
    R_star_r[R_star_r < -T_n] = -T_n
    
    for i in ob_supp_v1:
        R_star_r[i,:] = np.zeros(K_num-1)
    
    #N1+N2
    #Restrict the rows of R(l) and R(r) to the set N1+N2
    N1_N2 = list(set(N1).intersection(set(N2)))
    R_res_l = R_star_l[N1_N2,:]
    R_res_r = R_star_r[N1_N2,:]
    R_res = np.hstack((R_res_l, R_res_r)) 
    N1_N2_kmeans = KMeans(n_clusters = K_num).fit(R_res)
    
    #construct a dictionary to store the result of detection 
    community_dict = {}
    community_label = list(set(N1_N2_kmeans.labels_))
    dict_attribute = {'center_l':[], 'center_r':[], 'cluster_l':[], 'cluster_r':[], 'index':[]}
    community_dict = community_dict.fromkeys(community_label)
    for i in community_dict.keys():
        community_dict[i] = deepcopy(dict_attribute)
    
    #find the central node of each community
    cluster_R_res_l = list(zip(R_res_l, N1_N2_kmeans.labels_, N1_N2))
    cluster_R_res_r = list(zip(R_res_r, N1_N2_kmeans.labels_, N1_N2))
    for item in cluster_R_res_l:
        community_dict[item[1]]['cluster_l'].append(item[0])
        community_dict[item[1]]['index'].append(item[2])
    for item in cluster_R_res_r:
        community_dict[item[1]]['cluster_r'].append(item[0])        
    for i in community_dict.keys():
        cluster_l = np.array(community_dict[i]['cluster_l'])
        community_dict[i]['center_l'] = np.sum(cluster_l, axis = 0)/len(community_dict[i]['cluster_l'])
    for i in community_dict.keys():
        cluster_r = np.array(community_dict[i]['cluster_r'])
        community_dict[i]['center_r'] = np.sum(cluster_r, axis = 0)/len(community_dict[i]['cluster_r'])
    
    #N1/N2
    #find the nodes contained by N1 but not by N2
    N1eN2 = list(set(N1).difference(set(N1_N2)))
    #A matrix including all of the central vector in different communities
    center_matrix = []
    for value in community_dict.values():
        center_matrix.append(value['center_l'])
    center_matrix = np.array(center_matrix)
    #measure the distance between certain node and center vector of each community by euclidDistance 
    R_res_l_N1eN2 = R_star_l[N1eN2,:]
    for item in zip(R_res_l_N1eN2, N1eN2):
        distance = np.apply_along_axis(lambda a: np.sqrt(np.sum(a**2)),1,(center_matrix - item[0]))
        one_label = distance.argmax()
        community_dict[one_label]['index'].append(item[1])
    
    #N2/N1
    #find the nodes contained by N2 but not by N1
    N2eN1 = list(set(N2).difference(set(N1_N2)))
    #A matrix including all of the central vector in different communities
    center_matrix = []
    for value in community_dict.values():
        center_matrix.append(value['center_r'])
    center_matrix = np.array(center_matrix)
    #measure the distance between certain node and center vector of each community by euclidDistance 
    R_res_r_N2eN1 = R_star_r[N2eN1,:]
    for item in zip(R_res_r_N2eN1, N2eN1):
        distance = np.apply_along_axis(lambda a: np.sqrt(np.sum(a**2)),1,(center_matrix - item[0]))
        one_label = distance.argmax()
        community_dict[one_label]['index'].append(item[1])
        
    #/N1/N2
    #find the nodes contained by N1 and by N2
    eN2eN1 = list(set(range(n)).difference(set(list(set(N1).union(set(N2))))))
    R_res_eN2eN1_out= A[eN2eN1,:]
    R_res_eN2eN1_in = A[:,eN2eN1]
    for item in zip(R_res_eN2eN1_out, R_res_eN2eN1_in.T,eN2eN1):
        connect_node = list(set(np.r_[np.argwhere(item[0] == 1),np.argwhere(item[1] == 1)][0]))
        common_node_num = []
        for i in community_dict.keys():
            common_node_num.append(len(set(community_dict[i]['index']) & set(connect_node)))
        one_label = np.array(common_node_num).argmax()
        community_dict[one_label]['index'].append(item[2])
        
    #generate label vector for output
    label_vec = []
    for i in community_dict.keys():
        n = len(community_dict[i]['index'])        
        label_vec.extend(list(zip([i]*n, community_dict[i]['index'])))
    order_label_vec = sorted(label_vec, key = lambda label_vec: label_vec[1] )
    label = [item[0] for item in order_label_vec]
    return label

##prepartion for algorithm#########################################################
  
def networkx2A_directed(G):
    #extract connected weakly giant c
    
    
    
    
    omponent in directed network
    largest = max(nx.weakly_connected_components(G), key=len)
    A = nx.to_numpy_matrix(G.subgraph(largest)).getA()
    return A

##evaluation########################################################################    
def Hamming_error_rate(est_label, tr_label):
    #deepcopy
    estimated_label = deepcopy(est_label)
    true_label = deepcopy(tr_label)
    
    #stat num of label
    n = len(estimated_label)
    n_1 = len(list(set(true_label)))
    n_2 = len(list(set(estimated_label)))
        
    #update arrangement of estimated label
    ori_label_est = list(set(estimated_label))
    if n_1 > n_2 :
        arr = list(itertools.permutations(range(n_1),n_2))
    else:
        arr = list(itertools.permutations(range(n_2)))
    arr_rates = []
    
    #change the labels in estimated label
    new_label = list(islice([111, 222, 333, 444, 555, 666, 777, 888, 999], n_2))
    label_pair = sorted(zip(ori_label_est, new_label))
    for pair in label_pair:
        estimated_label[estimated_label == pair[0]] = pair[1]
    vir_label = list(set(estimated_label))
    
    #change the labels in true_label 
    ori_label_true = list(set(true_label))
    new_label_true = list(range(len(ori_label_true)))
    new_label_true_m = list(islice([111, 222, 333, 444, 555, 666, 777, 888, 999], n_1))
    label_pair = sorted(zip(ori_label_true, new_label_true_m))
    for pair in label_pair:
        true_label[true_label == pair[0]] = pair[1]    
    label_pair = sorted(zip(new_label_true_m, new_label_true))
    for pair in label_pair:
        true_label[true_label == pair[0]] = pair[1] 
        
    #test error rates under different labels    
    for one_arr in arr:
        re_label= sorted(zip(vir_label, one_arr))
        tri_label = deepcopy(estimated_label)
        for one_pair in re_label:
            tri_label[tri_label == one_pair[0]] = one_pair[1]
        err_rate = 1-Counter(tri_label - true_label)[0]/n
        arr_rates.append((tri_label,err_rate))
    
    #order label by error rates 
    arr_rates = sorted(arr_rates, key = lambda arr_rates: arr_rates[1])
    return  true_label, arr_rates[0][0], arr_rates[0][1]
    
def NMIaAMI(label_A, label_B):
#    label_A = [0, 0, 0, 1, 1, 1]
#    label_B = [0, 0, 1, 1, 2, 2]
    MI = {}    
    MI['AMI'] = metrics.adjusted_mutual_info_score(label_A, label_B) 
    MI['NMI'] = metrics.normalized_mutual_info_score(label_A,label_B)
    return MI
    
def ARI(label_A, label_B):
#    label_A = [0, 0, 0, 1, 1, 1]
#    label_B = [0, 0, 1, 1, 2, 2]    
    return metrics.adjusted_rand_score(label_A, label_B) 

## Ancillary functions ############################################################

# the function for read pickel document
def read_pkl(path_pkl):
    x = open(path_pkl, 'rb')
    journals_dict = pkl.load(x,encoding='iso-8859-1')
    x.close()
    return journals_dict

#remove the useless char
def rm_char(text):
    text = re.sub('\x01', '', text)                        #全角的空白符
    text = re.sub('\u3000', '', text) 
    text = re.sub('\n+', " ", text)
#    text = re.sub(' +', "E S", text)
    text = re.sub(r"[\)(↓%·▲ ……\s+】&【]", " ", text) 
    text = re.sub(r"[\d（）《》–>*!<`‘’:“”──"".￥%&*﹐,～-]", " ", text,flags=re.I)
    text = re.sub('\n+', " ", text)
    text = re.sub('[，、：@。_;」※\\\\☆=／|―「！"●#★\'■//◆－~？?；——]', " ", text)
    text = re.sub(' +', " ", text)
    text = re.sub('\[', " ", text)
    text = re.sub('\]', " ", text)
    return text
    

# build directed network with dict 
def build_Di_network(certain_dict):
    #搭建点
    G = nx.DiGraph()
    G.add_nodes_from(list(certain_dict.keys()))
    #搭建边(非自连)
    paper_edge = combine_tuple(certain_dict)
    
    G.add_edges_from(paper_edge)
    #搭建边（自连）
    #paper_edge = combine_tuple(certain_dict)
    #G.add_edges_from(paper_edge)
    return G

#generate the edges in certain network with dict
def combine_tuple(paper_dict):
    edge_list = []
    stat_result = statistics_paper_dict(paper_dict)
    all_papers = stat_result[2]             #all of the unisolated nodes
    for cite_paper, paper_content in paper_dict.items():
        if paper_content['cite_paper'] and cite_paper != 'untitled':
            for cited_paper in paper_content['cite_paper'].keys(): 
                if cited_paper in all_papers and cited_paper != cite_paper and cited_paper != 'untitled':
                    edge_list.append((cite_paper, cited_paper))      
    return edge_list

# statistics 
def statistics_paper_dict(paper_dict):
    paper_name = list(paper_dict.keys())
    paper = [''.join(rm_char(item).split())for item in paper_name]#remove all of the blocks in the titles
    cite_all = []                                                 #all the citations
    cite_paper = []
    cited_paper = []
    all_paper = []
    paper_dict_name = deepcopy(list(paper_dict.keys()))
    for one_paper in paper_dict_name:
        cite_all.extend(list(paper_dict[one_paper]['cite_paper'].keys()))
        cited_paper_dict_name = deepcopy(list(paper_dict[one_paper]['cite_paper'].keys()))
        for one_cited_paper in cited_paper_dict_name:
            one_cited_paper_abb = ''.join(rm_char(one_cited_paper).split())
            if one_cited_paper_abb in paper:
                orignal_name = paper_name[paper.index(one_cited_paper_abb)]
                cited_paper.append(orignal_name)
                cite_paper.append(one_paper)
                if one_cited_paper != orignal_name:
                    print([one_cited_paper,orignal_name])
                    paper_dict[one_paper]['cite_paper'][orignal_name] = paper_dict[one_paper]['cite_paper'][one_cited_paper]
                    paper_dict[one_paper]['cite_paper'].pop(one_cited_paper)                    
    cite_all = list(set(cite_all))            #the amount of citations
    cite_paper = list(set(cite_paper))        
    cited_paper = list(set(cited_paper))      
    all_paper.extend(cite_paper)              
    all_paper.extend(cited_paper)
    all_paper = list(set(all_paper))
    return[cite_paper,cited_paper,all_paper,cite_all] 

####main function##################################################################
if __name__ == "__main__": 
    path_subnet_cluster_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/subnet_cluster.pkl'
    subnet_cluster = read_pkl(path_subnet_cluster_pkl)
    G = build_Di_network(subnet_cluster['subnet_0'])
    A = networkx2A_directed(G)
    estimated_label = np.array(D_Score(A, 4))
    
    #evaluation 
    true_label = np.array(list(islice(cycle([0, 1, 2, 3]), 620))) # just for test the evaluation functions
    true_label, estimated_label, rate = Hamming_error_rate(estimated_label, true_label)
    NMIaAMI(estimated_label, true_label)
    ARI(estimated_label, true_label)
    


  
    
    