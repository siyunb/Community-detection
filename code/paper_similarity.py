# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:24:38 2019

@author: Bokkin Wang
"""

import os
import json
from math import log, sqrt
import re
import csv
import sys 
sys.path.append("D:/bigdatahw/dissertation/pre_defence/code")
import difflib
import numpy as np
from copy import deepcopy
from scipy import sparse
import pandas as pd
from itertools import islice   #导入迭代器
from collections import Counter
import networkx as nx
import pickle as pkl
from LDA_for_label import convert_doc_to_wordlist
from LDA_for_label import *
from gensim import corpora
from scipy.sparse import bsr_matrix, dok_matrix
from gensim.models import word2vec
from network_bulit import combine_tuple, trans_title_to_num
import warnings
warnings.filterwarnings("ignore")

def read_pkl(path_pkl):
    x = open(path_pkl, 'rb')
    journals_dict = pkl.load(x,encoding='iso-8859-1')
    x.close()
    return journals_dict

def single_list(arr, target):
    """获取单个元素的出现次数，使用list中的count方法"""
    return arr.count(target)

##拓扑相似度
def Topology_similarity(graph,subnet_name):
    ##距离相似度
    print(subnet_name+'\'s topological distance similarity starts to be constructed')
    short_distance = nx.all_pairs_shortest_path(graph)                            #自身的节点距离为1
    df_empty = pd.DataFrame(index=graph.nodes(), columns=graph.nodes()).fillna(0)  
    for index, row in df_empty.iterrows():
        for link_nod, route in short_distance[index].items():
            df_empty[index][link_nod]  = 1/(2**(len(route)-1))
    df_empty_T = deepcopy(df_empty.T) 
    df_dis_sim = df_empty_T + df_empty                                            #没有连接的就为零
    print(subnet_name+'\'s topological distance similarity has been finished')
    ##共同引用与共同被引用相似度
    print(subnet_name+'\'s topological citation similarity starts to be constructed')
    paper_group_matrix = nx.adjacency_matrix(graph).todense()
    paper_group_dataframe = pd.DataFrame(paper_group_matrix, index=graph.nodes(), columns=graph.nodes())
    df_com_sim = pd.DataFrame(index=graph.nodes(), columns=graph.nodes()).fillna(0)
    paper_name_list = paper_group_dataframe.columns.values.tolist()
    paper_name_list_c = deepcopy(paper_name_list)
    for index1 in paper_name_list:
        for index2 in paper_name_list_c:
            ##共同引用
            sum_cite = single_list(list(paper_group_dataframe.loc[index1]),1)+single_list(list(paper_group_dataframe.loc[index2]),1)
            com_cite = single_list(list(paper_group_dataframe.loc[index1]+paper_group_dataframe.loc[index2]),2)
            if sum_cite ==0:
                copq = 0
            else:
                copq = com_cite/(sum_cite-com_cite)
            ##共同被引用
            sum_cited = single_list(list(paper_group_dataframe.loc[:,index1]),1)+single_list(list(paper_group_dataframe.loc[:,index2]),1)
            com_cited = single_list(list(paper_group_dataframe.loc[:,index1]+paper_group_dataframe.loc[:,index2]),2)
            if sum_cited == 0:
                capq = 0
            else:
                capq = com_cited/(sum_cited-com_cited)
            df_com_sim.loc[index1,index2] = copq+capq
            df_com_sim.loc[index2,index1] = copq+capq
        paper_name_list_c.remove(index1)
        print(index1)
    df_com_sim = df_com_sim.apply(lambda x: x/x.sum(),axis=1)
    df_dis_sim = df_dis_sim.apply(lambda x: x/x.sum(),axis=1)
    print(subnet_name+'\'s topological citation similarity has been finished')
    print('Addition of distance similarity to citation similarity is in the progress')
    similarity_matrix = df_com_sim/2 +  df_dis_sim/2              
    return similarity_matrix

#属性相似度
def Attribute_similarity(paper_group, graph, path_dictionary, lda_model_path, w2v_model_path, num_topics, subnet_name):
    
    #发行商
    def publisher(paper_group, graph, subnet_name):
        print(subnet_name+'\'s attribute publisher similarity starts to be constructed')        
        df_publisher_sim = pd.DataFrame(index = graph.nodes(), columns = graph.nodes()).fillna(0) 
        paper_name_list = list(paper_group.keys())
        paper_name_list_c = deepcopy(paper_name_list)
        for index1 in paper_name_list:
            for index2 in paper_name_list_c:
                if paper_group[index1]['ego_attribute']['publisher'] == paper_group[index2]['ego_attribute']['publisher']:
                    df_publisher_sim.loc[index1,index2] = 1
                    df_publisher_sim.loc[index2,index1] = 1
            paper_name_list_c.remove(index1)   
            #print(index1)
        df_publisher_sim = df_publisher_sim.apply(lambda x: x/x.sum(),axis=1)       #归一化
        print(subnet_name+'\'s attribute publisher similarity has been finished')
        return df_publisher_sim
    
    def abstract(paper_group, graph, path_dictionary, lda_model_path, subnet_name):
        print(subnet_name+'\'s attribute abstract similarity starts to be constructed')                
        #载入LDA主题模型和字典
        dictionary = corpora.Dictionary.load(path_dictionary)
        ldamodel = read_pkl(lda_model_path)
        ##生成结果储存矩阵
        df_lda_score = pd.DataFrame(index = graph.nodes(), columns = ["lda"+str(i) for i in range(num_topics)]).fillna(0.00001) #防止除零无意义
        for paper, item in paper_group.items():
            bow_vector = dictionary.doc2bow(convert_doc_to_wordlist(item['ego_attribute']['abstract']))
            for index, score in sorted(ldamodel[bow_vector], key=lambda tup: -1*tup[1]):
                df_lda_score.loc[paper,"lda"+str(index)] = score+df_lda_score.loc[paper,"lda"+str(index)]
        #    print(paper)
        #Jensen-shannon散度
        df_lda_sim = pd.DataFrame(index = graph.nodes(), columns = graph.nodes()).fillna(0) 
        paper_name_list = list(paper_group.keys())
        paper_name_list_c = deepcopy(paper_name_list)
        for index1 in paper_name_list:
            for index2 in paper_name_list_c:
                M = (df_lda_score.loc[index1]+df_lda_score.loc[index2])/2
                D1 = (df_lda_score.loc[index1]*((df_lda_score.loc[index1]/M).apply(lambda x:log(x)))).sum(axis=0)
                D2 = (df_lda_score.loc[index2]*((df_lda_score.loc[index2]/M).apply(lambda x:log(x)))).sum(axis=0) 
                topic = 1- sqrt(D1 + D2)
                df_lda_sim.loc[index1,index2] = topic
                df_lda_sim.loc[index2,index1] = topic 
            paper_name_list_c.remove(index1)  
            #print(index1)
        df_lda_sim = df_lda_sim.apply(lambda x: x/x.sum(),axis=1)       #归一化  
        print(subnet_name+'\'s attribute abstract similarity has been finished')
        return df_lda_sim
        
    def keywords_and_plus(paper_group, graph,w2v_model_path, subnet_name):
        print(subnet_name+'\'s attribute keywords similarity starts to be constructed')                
        #所有标签集合
        r1 = u'[a-zA-Z]'
        label_sum = []
        for key, item in paper_group.items():
            label_sum.extend(item['ego_attribute']['keywords']+item['ego_attribute']['keyword_plus'])
        label_sum = sorted(list(set(label_sum)))
        #构建二部网络矩阵
        df_label = pd.DataFrame(index = graph.nodes(), columns = label_sum).fillna(0)         
        #判断词汇相似性利用w2v技术和字符串匹配,也就是相似度和距离
        w2v_model = read_pkl(w2v_model_path)
        for index1, item in paper_group.items():
            candidate = list(set(item['ego_attribute']['keywords']+item['ego_attribute']['keyword_plus']))
            for word in candidate:
                word_loc = label_sum.index(word)
                if word_loc<20:
                    word_loc = 20
                for index2 in label_sum[word_loc-20:word_loc+20] :
                    if difflib.SequenceMatcher(None, index2, word).quick_ratio() > 0.98:
                        df_label.loc[index1,index2] = 1
                    elif 0.80<difflib.SequenceMatcher(None, index2, word).quick_ratio()<0.98:
                        try:
                            if len(re.sub(r1, '', index2)+re.sub(r1, '', word))== 0:
                                if w2v_model.n_similarity(''.join(index2.split()),''.join(word.split())) >0.95:
                                    df_label.loc[index1,index2] = 1
                            else:
                                if w2v_model.n_similarity(''.join(convert_doc_to_wordlist(index2)),''.join(convert_doc_to_wordlist(word)) ) >0.93:
                                    df_label.loc[index1,index2] = 1
                        except:
                            pass
            #print(index1)
        #单词计算权重
        word_weigh = {}
        for keyword in label_sum:
            word_weigh[keyword] = 1/(log(df_label.loc[:,keyword].sum(axis=0))+0.01)
        #计算标签相似度
        df_key_sim = pd.DataFrame(index = graph.nodes(), columns = graph.nodes()).fillna(0) 
        paper_name_list = list(graph.nodes())
        paper_name_list_c = deepcopy(paper_name_list)     
        for paper1 in paper_name_list:
            for paper2 in paper_name_list_c:
                paper_array = df_label.loc[paper1,] + df_label.loc[paper2,]
                com_label = paper_array[paper_array==2]._stat_axis.values.tolist()
                com_label_cou = len(com_label)
                sim_sum = 0
                if com_label_cou > 0:
                    for one_label in com_label:
                        sim_sum += word_weigh[one_label]                    
                df_key_sim[paper1, paper2] = sim_sum
                df_key_sim[paper2, paper1] = sim_sum               
            paper_name_list_c.remove(paper1)  
            #print(paper1)
        df_key_sim = df_key_sim.apply(lambda x: x/x.sum(),axis=1)       #归一化 
        print(subnet_name+'\'s attribute keywords similarity has been finished')                                
        return df_key_sim
    
    return (publisher(paper_group, graph, subnet_name)+abstract(paper_group, graph, path_dictionary, lda_model_path, subnet_name)+keywords_and_plus(paper_group, graph,w2v_model_path, subnet_name))/3

        
#计算论文节点相似度
def paper_similarity(paper_cluster,path_dictionary, lda_model_path, w2v_model_path, path_df_prob_pkl, num_topics):
    subnet_name = paper_cluster[0]
    paper_group = paper_cluster[1] 
    print(subnet_name+'\'s probability matrix'+'starts to be constructed')
    graph = nx.DiGraph()
    graph.add_nodes_from(list(paper_group.keys()))    #注意这里加入了所有点
    paper_group_edge = combine_tuple(paper_group)     #不是所有点都具有连边
    graph.add_edges_from(paper_group_edge)            #加入了部分节点的连边
    print(subnet_name+'\'s topology similarity'+'starts to be constructed')
    df_topology = Topology_similarity(graph,subnet_name)          #拓扑相似度
    print(subnet_name+'\'s topology similarity'+'has been finished')
    print(subnet_name+'\'s attribute similarity'+'starts to be constructed')
    df_attribute = Attribute_similarity(paper_group, graph, path_dictionary, lda_model_path, w2v_model_path, num_topics,subnet_name) #属性相似度
    print(subnet_name+'\'s attribute similarity'+'has been finished')
    df_pro = 0.382*df_topology + 0.618*df_attribute   #加权
    #归一化
    df_prob = df_pro.apply(lambda x: x/x.sum(),axis=1)
    path_df_prob_pkl_gen = path_df_prob_pkl + subnet_name + '.pkl'
    df_prob_file = open(path_df_prob_pkl_gen, 'wb')
    pkl.dump(df_prob, df_prob_file)
    df_prob_file.close()
    print(subnet_name+'\'s probability matrix'+'has been finished')

    


if __name__ == "__main__": 

    path_journals_dict_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/journal_dict.pkl'
    path_paper_dict_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/paper_dict.pkl'
    path_subnet_cluster_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/subnet_cluster.pkl'
    journals_dict = read_pkl(path_journals_dict_pkl)
    paper_dict = read_pkl(path_paper_dict_pkl)
    subnet_paper_cluster = read_pkl(path_subnet_cluster_pkl)
    paper_num_dict, paper_num_title, paper_title_num = trans_title_to_num(paper_dict)
    
    ##制作路径
    w2v_model_path = 'D:/bigdatahw/dissertation/pre_defence/model/w2v_model.pkl'
    lda_model_path = 'D:/bigdatahw/dissertation/pre_defence/model/lda_true.pkl'
    path_dictionary = 'D:/bigdatahw/dissertation/pre_defence/model/dictionary.pkl'
    path_df_prob_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/df_paper_prob_'
    
    ##切分子网络
    paper_cluster = ('subnet_0',subnet_paper_cluster['subnet_0'])
    
    #尝试运行paper_similarity
    num_topics = 4
    paper_similarity(paper_cluster,path_dictionary, lda_model_path, w2v_model_path, path_df_prob_pkl, num_topics)
    
    
    
    
    
    