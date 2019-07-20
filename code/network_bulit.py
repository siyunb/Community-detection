# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 23:42:28 2019

@author: Bokkin Wang
"""

import os
import json
import re
import csv 
import pickle as pkl
import difflib
import numpy as np
from copy import deepcopy
import networkx as nx
import pandas as pd
from itertools import islice   #导入迭代器
import matplotlib.pyplot as plt
from scipy import sparse
from collections import Counter
from LDA_for_label import *
from gensim import corpora
from scipy.sparse import bsr_matrix, dok_matrix
from gensim.models import word2vec
import warnings
warnings.filterwarnings("ignore")

def single_list(arr, target):
    """获取单个元素的出现次数，使用list中的count方法"""
    return arr.count(target)

def read_pkl(path_pkl):
    x = open(path_pkl, 'rb')
    journals_dict = pkl.load(x,encoding='iso-8859-1')
    x.close()
    return journals_dict

#不包含自连
def combine_tuple(paper_dict):
    edge_list = []
    stat_result = statistics_paper_dict(paper_dict)
    all_papers = stat_result[2]             #所有不孤立点
    for cite_paper, paper_content in paper_dict.items():
        if paper_content['cite_paper'] and cite_paper != 'untitled':
            for cited_paper in paper_content['cite_paper'].keys(): 
                if cited_paper in all_papers and cited_paper != cite_paper and cited_paper != 'untitled':
                    edge_list.append((cite_paper, cited_paper))      
    return edge_list

#包含自连
def combine_tuple_ego(paper_dict):
    edge_list = []
    stat_result = statistics_paper_dict(paper_dict) #统计连边
    all_papers = stat_result[2]                     #
    for cite_paper, paper_content in paper_dict.items():
        edge_list.append((cite_paper,cite_paper))
        if paper_content['cite_paper'] :
            for cited_paper in paper_content['cite_paper'].keys(): 
                if cited_paper in all_papers:
                    edge_list.append((cite_paper, cited_paper))
    return edge_list

#去除无用字符
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
    
#筛选文章，输入文章字典增加连边
def statistics_paper_dict(paper_dict):
    paper_name = list(paper_dict.keys())
    paper = [''.join(rm_char(item).split())for item in paper_name]#将论文的名称去掉空格
    cite_all = []                                                 #所有的引用
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
    cite_all = list(set(cite_all))            #总引用
    cite_paper = list(set(cite_paper))        #引用者
    cited_paper = list(set(cited_paper))      #被引用者   
    all_paper.extend(cite_paper)              #网咯中不孤立的点
    all_paper.extend(cited_paper)
    all_paper = list(set(all_paper))
    return[cite_paper,cited_paper,all_paper,cite_all] #引用者,被引用者,网咯中不孤立的点,总引用

#做出一个对应字典，进入一个正常字典生成一个标号字典和一个标号字典名
def trans_title_to_num (paper_dict):
    paper_num_dict = {}
    paper_num_title = {}
    paper_title_num = {}
    i = 1
    for title, value in paper_dict.items():
        paper_num_dict[str(i)] = value 
        paper_num_title[str(i)] = title
        paper_title_num[title] = str(i)
        i= i+1
    return paper_num_dict, paper_num_title, paper_title_num
    
def change_title_num(title,paper_title_num):
    num = paper_title_num[title]
    return num

def change_num_title(num,paper_num_title):
    title = paper_num_title[num]
    return title

#阉割版引用网络，也就是去掉孤立点
def paper_dict_refine(paper_dict):
    stat_result = statistics_paper_dict(paper_dict)
    all_papers = stat_result[2]
    drop_list = set(list(paper_dict.keys())).difference(set(all_papers))
    paper_dict_drop = deepcopy(paper_dict)
    for drop_title in drop_list:
        paper_dict_drop.pop(drop_title)
    return paper_dict_drop

#构建网络
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
    return G,paper_edge

#
def build_network(certain_dict):
    G = nx.Graph()
    G.add_nodes_from(list(certain_dict.keys()))
    #搭建边(非自连)
    paper_edge = combine_tuple(certain_dict)
    G.add_edges_from(paper_edge)
    #搭建边（自连）
    #paper_edge = combine_tuple(certain_dict)
    #G.add_edges_from(paper_edge)
    return G

    
#找最大连通组件
def find_max_network(G):
    largest_components = max(nx.connected_components(G),key=len)
    print(len(largest_components))
    drop_nodes = set(list(G.nodes())).difference(set(largest_components))
    G_max = deepcopy(G)
    G_max.remove_nodes_from(drop_nodes) 
    return G_max,largest_components

#精炼网络转化为数字id
#def refine_num(paper_refine, paper_title_num):
#    paper_refine_num = {}
#    for key,content in paper_refine.items():
#        paper_refine_num[paper_title_num[key]] = content
#    return paper_refine_num

#输出点及连接关系方便gephi作图,分别
def net_output_csv(G,paper_title_num, path_csv):  
    start_spot = [edgetuple[0] for edgetuple in G.edges()]
    start_spot_id = [paper_title_num[edgetuple[0]] for edgetuple in G.edges()]
    end_spot = [edgetuple[1] for edgetuple in G.edges()]
    end_spot_id = [paper_title_num[edgetuple[1]] for edgetuple in G.edges()]
    pd.DataFrame({'start_spot':start_spot,'start_spot_id':start_spot_id,'end_spot':end_spot,'end_spot_id':end_spot_id}).to_csv(path_csv) 

if __name__ == '__main__':  
    #筛选数据
    path_read_journals_dict_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/journal_dict.pkl'
    path_read_paper_dict_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/paper_dict.pkl'
    path_subnet_cluster_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/subnet_cluster.pkl'
    path_papernet_csv = 'D:/bigdatahw/dissertation/pre_defence/csv/test.csv'
    path_papernet_refine_csv = 'D:/bigdatahw/dissertation/pre_defence/csv/refine'
    path_paper_subnet1 = 'D:/bigdatahw/dissertation/pre_defence/csv/subnet1.csv'
    path_paper_subnet2 = 'D:/bigdatahw/dissertation/pre_defence/csv/subnet2.csv'
    path_paper_subnet3 = 'D:/bigdatahw/dissertation/pre_defence/csv/subnet3.csv'
    path_paper_subnet4 = 'D:/bigdatahw/dissertation/pre_defence/csv/subnet4.csv'
    journals_dict = read_pkl(path_read_journals_dict_pkl)
    paper_dict = read_pkl(path_read_paper_dict_pkl)
    subnet_cluster = read_pkl(path_subnet_cluster_pkl)
    paper_num_dict, paper_num_title, paper_title_num = trans_title_to_num(paper_dict)
    paper_refine = paper_dict_refine(paper_dict)                                #精炼的网络
#    paper_num_refine_dict = refine_num(paper_refine, paper_title_num)
    
    G_paper_title,paper_edge = build_Di_network(paper_dict) #构建最基础的paper_network
#    G_paper_num = build_network(paper_num_dict) #所包含的是id
    G_paper_fine_title = build_network(paper_refine) #构建精炼的paper_network
#    G_paper_fine_num = build_network(paper_num_refine_dict) #精炼后的id网络
    
    #创建4个子网络
    G_paper_subnet1,paper_edge_subnet1 = build_Di_network(subnet_cluster['subnet_0'])
    G_paper_subnet2,paper_edge_subnet2 = build_Di_network(subnet_cluster['subnet_1'])
    G_paper_subnet3,paper_edge_subnet3 = build_Di_network(subnet_cluster['subnet_2'])
    G_paper_subnet4,paper_edge_subnet4 = build_Di_network(subnet_cluster['subnet_3'])
    
    #输出csv给gephi
    net_output_csv(G_paper_title,paper_title_num, path_papernet_csv)
#    net_output_csv(G_paper_fine_title,paper_title_num, path_papernet_refine_csv)
    net_output_csv(G_paper_subnet1,paper_title_num, path_paper_subnet1)
    net_output_csv(G_paper_subnet2,paper_title_num, path_paper_subnet2)
    net_output_csv(G_paper_subnet3,paper_title_num, path_paper_subnet3)
    net_output_csv(G_paper_subnet4,paper_title_num, path_paper_subnet4)
    
    G_paper_fine, largest_of_G_paper_fine = find_max_network(G_paper_title)
    largest_of_G_paper_fine_num = [change_title_num(title,paper_title_num) for title in list(largest_of_G_paper_fine)]
    
    #绘图
    nx.draw(g,pos=nx.random_layout(g),node_color = 'b',edge_color = 'r',node_size =10,style='solid',node_shape='o',font_size=20)
    plt.savefig("paper_group.png", dpi=400, bbox_inches='tight')
    plt.show()

    #描述
    len(G_paper_title)
    G_paper_title.number_of_edges()
    nx.degree_histogram(G_paper_fine)

    #转矩阵    
    nx.adjacency_matrix(G_paper_title).todense()
    dense = nx.adjacency_matrix(G_paper_title).todense()
    sparse.coo_matrix(dense)
    
    ##储存中间结果
    path_pkl = 'D:/bigdatahw/dissertation/pre_defence/nod/paper_group1_metrix.pkl'
    paper_dict_file = open(path_pkl, 'wb')
    pkl.dump(dense, paper_dict_file)
    paper_dict_file.close()
