# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:24:48 2019

@author: Bokkin Wang
"""
#我的abstract貌似都漏掉了一个s哭笑，请注意下
import os
import json
import re
import nltk
import csv 
import difflib
import numpy as np
from copy import deepcopy
import pandas as pd
from collections import Counter
import pyLDAvis
import pyLDAvis.gensim
import pickle as pkl
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer       #词形变化
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from gensim.models import LdaModel
from gensim import corpora, models
import tempfile
from pprint import pprint

##缩略语补全
class RegexpReplacer(object):
    replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')
    ]
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

##重复字符删除
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word    

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

##去除字符        
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

##去除数字、非英文字符、停用词、转换成小写、词形还原等文本整体处理函数
def rm_tokens(words):  # 去掉一些停用词和完全包含数字的字符串
    ##进行基础词汇定义
    words = words.encode("utf-8").decode("utf-8")
    words = re.sub("\d","",words)
    Reg = RegexpReplacer()                           #删除缩写
    words_seg = [re.sub(u'\W', "", Reg.replace(i)) for i in nltk.word_tokenize(words)] 
    space_len = words_seg.count(u"")
    for i in list(range(space_len)):
        words_seg.remove(u'')
    filtered = [w.lower() for w in words_seg if w.lower() not in stopwords.words('english')] 
    tagged_sent = pos_tag(filtered)     # 获取单词词性
    wnl = WordNetLemmatizer()
    lemmatized = []
    #Rep = RepeatReplacer()
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        if wordnet_pos not in ['v','a','r']:
            lemmatized.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
    #final = [Rep.replace(w) for w in stems]          #删除重复字符
    return " ".join(lemmatized)

#删除无含义的单个字符保留有含义的单个字符例如r,p,a
def remove_single(tmp_list):
#    flu_word = ['model','data','method','time','using','use','john','paper','one','used','problem','analysis','based','approach','study','result','also']
    flu_word = []
    retain_single_word = ['r','p','a'] 
    new_list = []
    for word in tmp_list:
        if len(word)>2 and word not in flu_word:
            new_list.append(word)  
        if word in retain_single_word:
            new_list.append(word)        
    return new_list

#将文档转化为单词列表
def convert_doc_to_wordlist(str_doc):
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list)  # 去掉一些字符，例如\u3000
    word_list = [rm_tokens(part) for part in sent_list]  # 分词
    word2list = [remove_single(a.split()) for a in word_list]
    return word2list[0]

#读取序化模型
def read_pkl(path_pkl):
    x = open(path_pkl, 'rb')
    journals_dict = pkl.load(x,encoding='iso-8859-1')
    x.close()
    return journals_dict

#写出序化模型
def write_pkl(path_pkl,abtract_complete):
    abtract_complete_file = open(path_pkl, 'wb')
    pkl.dump(abtract_complete, abtract_complete_file)
    abtract_complete_file.close()

##删除词频为1的词汇
def remove_flu_one(abtract_complete, topmany):
    abtract_complete_count = []
    for onelist in abtract_complete.values():
        abtract_complete_count.extend(onelist)
    word_count = (Counter(abtract_complete_count).most_common())
    word_count_top = word_count[0:topmany]                            #词频最高的20个单词
    for i in range(len(word_count)-1, -1, -1):
        if word_count[i][1] == 1:
            word_count_top.append(word_count[i])                #删除次品唯一的单词
    for word_list in abtract_complete.values():
        for minute_word in word_count_top:
            while minute_word[0] in word_list:
                word_list.remove(minute_word[0])
    return abtract_complete

##综合文本状况
def sum_paper_massage(year_list,journals_dict):
    abtract_complete = {}
    ##读取文本状况
    for journal in journals_dict.keys(): 
        bag = []
        for year in year_list:
            bag.extend(convert_doc_to_wordlist(journals_dict[journal][year]['abstract_sum']))
        abtract_complete[journal] = bag
    return abtract_complete

#LDA结果类
class LDA_result(object):
    
    def __init__(self, abtract_complete_true,num_topics = 4, chunksize = 1000, passes = 60, iterations = 600, eval_every = None):
        self.num_journal = len(abtract_complete_true)
        self.abtract_complete_true = abtract_complete_true
        self.abtract_complete = self.abtract_complete_combination()
        self.dictionary = corpora.Dictionary(self.abtract_complete)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.abtract_complete]    
        self.temp = self.dictionary[0]
        self.id2word = self.dictionary.id2token
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.eval_every = eval_every
        self.model = LdaModel(corpus = self.corpus, id2word = self.id2word, chunksize = self.chunksize, \
                       alpha='auto', eta='auto', \
                       iterations = self.iterations, num_topics = self.num_topics, \
                       passes = self.passes, eval_every = self.eval_every)           #建立模型    
    
    #组合摘要词汇
    def abtract_complete_combination(self):
        abtract_complete = []
        for journal_word_list in self.abtract_complete_true.values():
            abtract_complete.append(journal_word_list)
        return abtract_complete
                
    ##描述情况
    def description(self):
        print('Number of unique tokens: %d' % len(self.dictionary))
        print('Number of documents: %d' % len(self.corpus))
    
    #转tfidf向量    
    def word2tfidf(self):
        tfidf = models.TfidfModel(self.corpus)
        corpusTfidf = tfidf[self.corpus]
        return corpusTfidf
    
    #输出各个主题关键词
    def key_words(self):
        top_topics = self.model.top_topics(self.corpus)
        pprint(top_topics) 
        
    #每一行包含了主题词和主题词的权重
    def key_weight(self):
        print(self.model.print_topic(0,10))
        print(self.model.print_topic(1,10))  
    
    #判断第一个训练集文档属于哪个主题，没什么卵用凑个数
    def topic_belong(self):
        for index, score in sorted(self.model[self.corpus[0]], key=lambda tup: -1*tup[1]):
            print("Score: {}\n Topic: {}".format(score, self.model.print_topic(index, 10)))
    
    #LDA进行可视化
    def visible(self):
        vis_wrapper = pyLDAvis.gensim.prepare(self.model,self.corpus,self.dictionary)
        pyLDAvis.display(vis_wrapper)
        pyLDAvis.save_html(vis_wrapper,"lda%dtopics.html"%self.num_topics)
        pyLDAvis.show(vis_wrapper)
    
    #给训练集输出其属于不同主题概率  
    def community_belong(self):
        journal_community = {}
        for i,element in enumerate(abtract_complete_true):
            journal_community[element] = []
            for index, score in sorted(self.model[self.corpus[i]], key=lambda tup: -1*tup[1]):
                if score > 0.2:
                    journal_community[element].append(str(index))
                print(index, score)
        return journal_community

    #给定新的语料    
#    @staticmethod
#    def word_corpus(abtract_complete):
#        dictionary = corpora.Dictionary(abtract_complete)
#        corpus = [dictionary.doc2bow(text) for text in abtract_complete]  
#        return corpus
    
    #判断新预料的主题归属
    def identify_community(self, abtract_complete):
        corpus = self.dictionary.doc2bow(abtract_complete)
        community = []
        for index, score in sorted(self.model[corpus], key=lambda tup: -1*tup[1]):
            if score > 0.2:
                community.append(str(index)) 
        return community

#将社区信息加入字典
def add_com_dict(journal_dict,journal_community):
    for key,community in journal_community.items():
        journals_dict[key]['community'] = community
    return journal_dict

if __name__ == '__main__':  
    os.chdir("D:/bigdatahw/dissertation/pre_defence/data")
    path_csv = 'D:/bigdatahw/dissertation/pre_defence/data'
    path_read_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/journal_dict.pkl'
    path_write_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/abtract_complete.pkl'
    path_abtract_complete_true_pkl = 'D:/bigdatahw/dissertation/pre_defence/pkl/abtract_complete_true.pkl'
    path_model_pkl = 'D:/bigdatahw/dissertation/pre_defence/model/lda_true.pkl'
    path_dictionary_pkl = 'D:/bigdatahw/dissertation/pre_defence/model/dictionary.pkl'
    journals_dict = read_pkl(path_read_pkl)
    year_list = ['2018','2017','2016','2015','2014']
    abtract_complete = sum_paper_massage(year_list,journals_dict)
    ##写出文本分词的pkl        
    write_pkl(path_write_pkl,abtract_complete)
    ##去除高频词
    abtract_complete_true = remove_flu_one(abtract_complete, 20)
    write_pkl(path_abtract_complete_true_pkl,abtract_complete)
    ##读取文本分词
    abtract_complete_true = read_pkl(path_abtract_complete_true_pkl)    
    #写出标记社区的字典
    write_pkl(path_read_pkl,journals_dict)     
    #实验一下
    a = LDA_result(abtract_complete_true)
    write_pkl(path_model_pkl,a.model) #写出模型和字典
    write_pkl(path_dictionary_pkl,a.dictionary)
    journals_dict = add_com_dict(journals_dict,a.community_belong())
    write_pkl(path_read_pkl,journals_dict)

    
    

    


