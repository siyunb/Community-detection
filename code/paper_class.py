# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:01:32 2018

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
import pandas as pd
from itertools import islice   #导入迭代器

#paper类主要负责的是进入对一条paper信息进行统合整理，转化为可以建模的网络节点信息
class paper(object):
    
    def __init__(self,one_line):
        self.authors = [author.split('@')[0].replace(', ',',').replace('  ',' ').replace('. ','.').title().replace('-','') for author in one_line[10].split('::')]  #创建一个作者名录
        self.title = self.regulate_title(one_line[0].lower().strip().replace('  ',' '))     
        self.publisher = self.regulate_publisher(one_line[1].lower())
        self.doi = one_line[2].lower()
        self.publish_date = one_line[3].strip().replace('  ',' ')
        self.cited_num = one_line[4]
        self.time_cited = one_line[5]
        self.abstract = one_line[6]
        self.keyword = ','.join([item.strip().lower().replace('  ',' ') for item in one_line[7].split(',')])
        self.keyword_plus = ','.join([item.strip().lower().replace('  ',' ') for item in one_line[8].split(',')])
        self.university = list(set([thing for item in one_line[10].split('::') for thing in item.split('@')[1:]]))
        self.cite_papers = one_line[12].split('::')
    
    #进行文本匹配抽出论文题目    
    def regulate_title(self,title):
        pattern = re.compile(r'\(\d{4}\),\s"(.*)\W".*')   # 查找数字
        result = pattern.findall(title)
        if result:
            return result[0]
        else:
            return title
    
    #抽象出出版商    
    def regulate_publisher(self, publisher):
        pattern = re.compile(r'(.*),vol\s+\d+')   # 查找数字
        result = pattern.findall(publisher)
        if result:
            return result[0]
        else:
            return publisher
    
    #    
    def time_transform(self):
        def months_match(var):
            return {'JAN': '1','JANUARY':'1','FEB': '2','FEBRUARY':'2','MAR': '3','MARCH':'3',
                    'APR':'4','APRIL':'4','MAY':'5','JUN':'6','JUNE':'6','JUL':'7','JULY':'7',
                    'AUG':'8','AUGUST':'8','SEP':'9','SEPT':'9','SEPTEMBER':'9',
                    'OCT':'10','OCTOBER':'10','NOV':'11','NOVEMBER':'11','DEC':'12','DECEMBER':'12'}.get(var,'error')  
        re_pattern1 = re.compile(r"[A-Za-z]{3,9}\s+?\d{1,2}\s+?\d{4}")
        re_pattern2 = re.compile(r"[A-Za-z]{3,9}\s+?\d{4}")
        re_pattern3 = re.compile(r"\d{4}/\d{1,2}/\d{1,2}")
        re_pattern4 = re.compile(r"[A-Za-z]{3}-[A-Za-z]{3}\s+?\d{4}")
        re_pattern5 = re.compile(r"\d{4}")
        re_pattern6 = re.compile(r"[A-Za-z]{3}-\d{2}")
        re_pattern7 = re.compile(r"[A-Za-z]{3,9}\s*\d{1,2}\s*\d{4}")
        re_pattern8 = re.compile(r"\d{1,2}-[A-Za-z]{3}")
        re_pattern9 = re.compile(r"\d{4}-[A-Za-z]{3,9}")
        re_pattern10 = re.compile(r"\d{2}\s+?\d{4}")
        self.publish_date = self.publish_date.replace('.',' ')
        if re_pattern1.fullmatch(self.publish_date):
            date_time = self.publish_date.split()
            date = date_time[2]+'-' + months_match(date_time[0].upper())
        if re_pattern2.fullmatch(self.publish_date):
            date_time = self.publish_date.split()
            date = date_time[1]+'-' + months_match(date_time[0].upper())
        if re_pattern3.fullmatch(self.publish_date):
            date = self.publish_date.split('/')[0]+'-'+self.publish_date.split('/')[1]  
        if re_pattern4.fullmatch(self.publish_date):
            date_time = [item.split('-')[0] for item in self.publish_date.split()]
            date = date_time[1]+'-'+ months_match(date_time[0].upper())
        if re_pattern5.fullmatch(self.publish_date) and len(self.publish_date) == 4:
            date_time = self.publish_date
            date = self.publish_date + '-6'    
        if re_pattern6.fullmatch(self.publish_date):
            if int(self.publish_date.split('-')[1]) > 80:
                year = '19'+self.publish_date.split('-')[1]
            else:
                year = '20'+self.publish_date.split('-')[1]
            date = year + '-'+months_match(self.publish_date.split('-')[0].upper()) 
        if re_pattern7.fullmatch(self.publish_date):
            date = self.publish_date.split()[2]+'-'+months_match(self.publish_date.split()[0])  
        if re_pattern8.fullmatch(self.publish_date):
            if int(self.publish_date.split('-')[0]) > 80:
                year = '19'+self.publish_date.split('-')[0]
            else:
                if len(self.publish_date.split('-')[0]) == 1:
                    year = '200'+self.publish_date.split('-')[0]
                    date = year + '-'+months_match(self.publish_date.split('-')[1].upper())                     
                else:    
                    year = '20'+self.publish_date.split('-')[0]
                    date = year + '-'+months_match(self.publish_date.split('-')[1].upper())  
        if re_pattern9.fullmatch(self.publish_date):
            date = self.publish_date.split('-')[0]+'-'+months_match(self.publish_date.split('-')[1].upper())
        if re_pattern10.fullmatch(self.publish_date):
            date = self.publish_date.split()[1]+'-'+self.publish_date.split()[0].replace('0','')        
        if self.publish_date == '':
            date = '1905-6'
        return date 
    
    def time_transform1(self,publish_date):
        def months_match(var):
            return {'JAN': '1','JANUARY':'1','FEB': '2','FEBRUARY':'2','MAR': '3','MARCH':'3',
                    'APR':'4','APRIL':'4','A':'4','MAY':'5','JUN':'6','JUNE':'6','JUL':'7','JULY':'7',
                    'AUG':'8','AUGUST':'8','SEP':'9','SEPT':'9','SEPTEMBER':'9',
                    'OCT':'10','OCTOBER':'10','NOV':'11','NOVEMBER':'11','DEC':'12','DECEMBER':'12'}.get(var,'error')  
        re_pattern1 = re.compile(r"[A-Za-z]{3,9}\s+?\d{1,2}\s+?\d{4}")
        re_pattern2 = re.compile(r"[A-Za-z]{1,9}\s+?\d{4}")
        re_pattern3 = re.compile(r"\d{4}/\d{1,2}/\d{1,2}")
        re_pattern4 = re.compile(r"[A-Za-z]{3,9}\s*-[A-Za-z]{3,9}\s+?\d{4}")
        re_pattern5 = re.compile(r"\d{4}")
        re_pattern6 = re.compile(r"[A-Za-z]{3}.\d{2}")
        re_pattern7 = re.compile(r"[A-Za-z]{3,9}\s*\d{1,2}\s*\d{4}")
        re_pattern8 = re.compile(r"\d{1,2}-[A-Za-z]{3}")
        re_pattern9 = re.compile(r"\d{4}-[A-Za-z]{3,9}")
        re_pattern10 = re.compile(r"\d{1,2}\s+?\d{4}")
        re_pattern11 = re.compile(r"\d{1,2}[A-Za-z]{2}\s*[A-Za-z]{3,9}\s*\d{4}")
        re_pattern12 = re.compile(r"[A-Za-z]{3,9}\s*/[A-Za-z]{3,9}\s+?\d{4}")
        re_pattern13 = re.compile(r"\d{1,2}\s*[A-Za-z]{3,9}\s*\d{4}")
        re_pattern14 = re.compile(r"\d{1,2}-\d{1,2}\s*[A-Za-z]{3,9}\s*\d{4}")
        re_pattern15 = re.compile(r"\d{4}-[A-Za-z]{3,9}-\d{1,2}")
        re_pattern16 = re.compile(r"\d{1,2}\s+?\d{1,2}\s+?\d{4}")
        re_pattern17 = re.compile(r"\d{1,2}\s*[A-Za-z]{3,9}\s*-\d{1,2}\s*[A-Za-z]{3,9}\s*\d{4}")
        re_pattern18 = re.compile(r"\d{4}\s*[A-Za-z]{3,9}\s*\d{1,2}\s*-\d{1,2}\s*")
        re_pattern19 = re.compile(r"\d{4}\s*[A-Za-z]{3,9}\s*\d{1,2}")
        re_pattern20 = re.compile(r"\d{4}-\d{1,2}")
        re_pattern21 = re.compile(r"\d{4}\s*[A-Za-z]{3,9}-[A-Za-z]{3,9}")
        re_pattern22 = re.compile(r"\d{1,2}/\d{1,2}/\s*\d{4}")
        re_pattern23 = re.compile(r"[A-Za-z]{3,9}\s*\d{1,2}-\d{1,2}\s*\d{4}")
        re_pattern24 = re.compile(r"\d{1,2}\s*[A-Za-z]{3,9}\s*-\s*\d{1,2}\s*[A-Za-z]{3,9}\s*\d{4}")
        re_pattern25 = re.compile(r"\d{4}-\d{4}")
        re_pattern26 = re.compile(r"[A-Za-z]{3,9}\s*\d{4}\(\d{4}\)")
        re_pattern27 = re.compile(r"-\d{4}\s*\d{4}")
        re_pattern28 = re.compile(r"[A-Za-z]{3,9}\s*\d{2}")
        re_pattern29 = re.compile(r"[A-Za-z]{3,9}\s*\d{1,2}-[A-Za-z]{3,9}\s*\d{1,2}\s*\d{4}")
        re_pattern30 = re.compile(r"[A-Za-z]{3,9}-[A-Za-z]{3,9}\s*\d{4}\(\d{4}\)")
        re_pattern31 = re.compile(r"/\d{1,2}\s*\d{4}")
        publish_date = publish_date.replace('.',' ').replace(',',' ')
        if re_pattern1.fullmatch(publish_date):
            date_time = publish_date.split()
            date = date_time[2]+'-' + months_match(date_time[0].upper())
        if re_pattern2.fullmatch(publish_date):
            date_time = publish_date.split()
            date = date_time[1]+'-' + months_match(date_time[0].upper())
        if re_pattern3.fullmatch(publish_date):
            date = publish_date.split('/')[0]+'-'+publish_date.split('/')[1]  
        if re_pattern4.fullmatch(publish_date):
            date_time = [item.split('-')[0] for item in publish_date.split()]
            date = date_time[1]+'-'+ months_match(date_time[0].upper())
        if re_pattern5.fullmatch(publish_date) and len(publish_date) == 4:
            date_time = publish_date
            date = publish_date + '-6'    
        if re_pattern6.fullmatch(publish_date):
            if int(publish_date.split('-')[1]) > 70:
                year = '19'+publish_date.split('-')[1]
            else:
                year = '20'+publish_date.split('-')[1]
            date = year + '-'+months_match(publish_date.split('-')[0].upper()) 
        if re_pattern7.fullmatch(publish_date):
            date = publish_date.split()[2]+'-'+months_match(publish_date.split()[0])  
        if re_pattern8.fullmatch(publish_date):
            if int(publish_date.split('-')[0]) > 70:
                year = '19'+publish_date.split('-')[0]
                date = year + '-'+months_match(publish_date.split('-')[1].upper())
            else:
                if len(publish_date.split('-')[0]) == 1:
                    year = '200'+publish_date.split('-')[0]
                    date = year + '-'+months_match(publish_date.split('-')[1].upper())                     
                else:    
                    year = '20'+publish_date.split('-')[0]
                    date = year + '-'+months_match(publish_date.split('-')[1].upper()) 
        if re_pattern9.fullmatch(publish_date):
            date = publish_date.split('-')[0]+'-'+months_match(publish_date.split('-')[1].upper())
        if re_pattern10.fullmatch(publish_date):
            date = publish_date.split()[1]+'-'+publish_date.split()[0].replace('0','')
        if re_pattern11.fullmatch(publish_date):
            date = publish_date.split()[2]+'-'+months_match(publish_date.split()[1].upper())
        if re_pattern12.fullmatch(publish_date):
            date_time = [item.split('/')[0] for item in publish_date.split()]
            date = date_time[1]+'-'+ months_match(date_time[0].upper())
        if re_pattern13.fullmatch(publish_date):
            date = publish_date.split()[2]+'-'+ months_match(publish_date.split()[1].upper())  
        if re_pattern14.fullmatch(publish_date):
            date = publish_date.split()[2]+'-'+ months_match(publish_date.split()[1].upper()) 
        if re_pattern15.fullmatch(publish_date):
            date = publish_date.split('-')[0]+'-'+ months_match(publish_date.split('-')[1].upper())
        if re_pattern16.fullmatch(publish_date): 
            date = publish_date.split()[2]+'-'+ publish_date.split()[1]  
        if re_pattern17.fullmatch(publish_date): 
            date = publish_date.split()[-1]+'-'+ months_match(publish_date.split()[1].upper())
        if re_pattern18.fullmatch(publish_date): 
            date = publish_date.split()[0]+'-'+ months_match(publish_date.split()[1].upper())   
        if re_pattern19.fullmatch(publish_date): 
            date = publish_date.split()[0]+'-'+ months_match(publish_date.split()[1].upper())    
        if re_pattern20.fullmatch(publish_date):
            date = publish_date.split('-')[0]+'-'+publish_date.split('-')[1].replace('0','')
        if re_pattern21.fullmatch(publish_date):
            date = publish_date.split()[0]+'-'+months_match(publish_date.split()[1].split('-')[0].upper())
        if re_pattern22.fullmatch(publish_date):
            date = publish_date.split()[1]+'-'+publish_date.split()[0].split('/')[1]    
        if re_pattern23.fullmatch(publish_date):
            date = publish_date.split()[-1]+'-'+months_match(publish_date.split()[0].upper()) 
        if re_pattern24.fullmatch(publish_date):
            date = publish_date.split()[-1]+'-'+months_match(publish_date.split()[-2].upper())  
        if re_pattern25.fullmatch(publish_date):
            date = publish_date.split('-')[0]+'-6'     
        if re_pattern26.fullmatch(publish_date):
            date = publish_date.split()[1].split('(')[0]+'-'+months_match(publish_date.split()[0].upper())
        if re_pattern27.fullmatch(publish_date): 
            publish_date = publish_date.replace('-','').replace(' ','-')
            date = publish_date.split('-')[0]+'-6'              
        if re_pattern28.fullmatch(publish_date):            
            if int(publish_date.split()[1]) > 70:
                year = '19'+publish_date.split()[1]
                date = year + '-'+months_match(publish_date.split()[0].upper())
            else:
                if len(publish_date.split()[1]) == 1:
                    year = '200'+publish_date.split()[1]
                    date = year + '-'+months_match(publish_date.split()[0].upper())                     
                else:    
                    year = '20'+publish_date.split()[1]
                    date = year + '-'+months_match(publish_date.split()[0].upper()) 
        if re_pattern29.fullmatch(publish_date): 
            date = publish_date.split()[1]+'-'+months_match(publish_date.split()[0].upper()) 
        if re_pattern30.fullmatch(publish_date):
            date = publish_date.split()[1].split('(')[0]+'-'+months_match(publish_date.split()[0].split('-')[0].upper())
        if re_pattern31.fullmatch(publish_date):
            date = publish_date.split()[1]+'-6'
        if publish_date == '':
            date = '1905-6'
        if publish_date == 'no time':
            date = 'no time'
        return date 

    def string_process(self,string):
        string = string.strip()
        string = string.replace(', ',',')
        string = string.replace('  ',' ')
        string = string.replace('   ',' ')
        return string
    
    def top_matching_degree(self,item_name):
        top_author = ''
        top_degree = 0 
        for author in self.authors:
            degree = difflib.SequenceMatcher(None, item_name, author).quick_ratio()
            if degree > top_degree:
                top_author = author
                top_degree = degree
        if top_degree > 0.57:
            return top_author
        else:
            top_author = ''
            return top_author
    
    def add_publish_date(self):              #存储日期
        try:
            date = self.string_process(self.time_transform())
            self.paper_dict['ego_attribute']['publish_date'] = date                                     
        except:
            print(self.publish_date)
            print('publish_date wrong')
    
    def add_publisher(self):                #存储杂志
        try:
            journal = self.string_process(self.publisher)
            self.paper_dict['ego_attribute']['publisher'] = self.regulate_publisher(journal)                        
        except:
            print('publisher wrong')
            
    def add_cited_num(self):                #储存引用文献
        try:
            if self.cited_num == '':
                self.paper_dict['ego_attribute']['cited_num'] = 0                         

            else:
                cited_num = self.string_process(self.cited_num).replace(',','') 
                self.paper_dict['ego_attribute']['cited_num'] = int(cited_num)
        except:
            print(self.cited_num)
            print('cited_num wrong')
    
    def add_time_cited(self):               #储存被引用文献
        try:
            if self.time_cited == '':
                self.paper_dict['ego_attribute']['time_cited'] = 0                         
       
            else:
                time_cited = self.string_process(self.time_cited).replace(',','') 
                self.paper_dict['ego_attribute']['time_cited'] = int(time_cited)                    
        except:
            print(self.time_cited)
            print('time_cited wrong')
            
    def add_abstract(self):
        try:
            abstract = self.string_process(self.abstract)
            if abstract == 'no abstract':
                self.paper_dict['ego_attribute']['abstract'] = ''                    
            else:
                self.paper_dict['ego_attribute']['abstract'] = self.abstract      
        except:
            print('abstract wrong')
            
    def add_keyword(self):
        try:
            keyword = self.string_process(self.keyword)
            self.paper_dict['ego_attribute']['keywords'] = keyword.split(',')      
        except:
            print('keyword wrong')
    
    def add_keyword_plus(self):
        try:
            keyword_plus = self.string_process(self.keyword_plus)
            self.paper_dict['ego_attribute']['keyword_plus'] = keyword_plus.split(',')     
        except:
            print('keyword_plus wrong')
    
    def add_author(self):
#        try:
        self.paper_dict['ego_attribute']['author'] = self.authors                    
#        except:
#            print('coauthor wrong')
#            sys.exit(0)      
            
    def add_university(self):
#        try:
        universitys = [self.string_process(item).title() for item in self.university]
        universitys = list(set(universitys))
        self.paper_dict['ego_attribute']['university'] = universitys                        
#        except:
#            print('coauthor wrong')
#            sys.exit(0)          

    def add_doi(self):
        try:
            self.paper_dict['ego_attribute']['doi'] = self.doi    
        except:
            print('doi wrong')       
    
    def clearify_cite_paper_authors(self,cite_paper_authors):
        for tag in cite_paper_authors:
            if 'Et Al' in tag:
                cite_paper_authors.remove(tag)
        
        for tag in cite_paper_authors:
            if '<' in tag :
                i = tag.index('<')
                tag_g = tag[:i]  
                j = cite_paper_authors.index(tag)                
                cite_paper_authors.remove(tag)
                cite_paper_authors.insert(j,tag_g)
        
        for tag in cite_paper_authors:
            if '>' in tag:
                cite_paper_authors.remove(tag)
                
        return cite_paper_authors
                
    def add_cite_paper(self):
        try:
            if self.cite_papers != ['']:
                for cite_paper_massage in self.cite_papers:
                    cite_paper_name = '+'.join(cite_paper_massage.split('+')[0:-3]).lower()
                    if cite_paper_name != 'no title' :
                        cite_paper_authors = [self.string_process(item).title() for item in cite_paper_massage.split('+')[-3].split(';')]
                        cite_paper_journal = cite_paper_massage.split('+')[-2].lower()
                        cite_paper_date = self.time_transform1(cite_paper_massage.split('+')[-1]) 
                        if cite_paper_date >= '1988-00':
                            self.paper_dict['cite_paper'][cite_paper_name] = {}
                            self.paper_dict['cite_paper'][cite_paper_name]['author'] = self.clearify_cite_paper_authors(cite_paper_authors)
                            self.paper_dict['cite_paper'][cite_paper_name]['publisher'] = cite_paper_journal
                            self.paper_dict['cite_paper'][cite_paper_name]['publish_date'] = cite_paper_date
                        else:
                            continue
                    else:
                        cite_paper_name = '+'.join(cite_paper_massage.split('+')[2:-1]).lower()
                        cite_paper_authors = [self.string_process(item).title() for item in cite_paper_massage.split('+')[1].split(';')]
                        cite_paper_journal = '+'.join(cite_paper_massage.split('+')[2:-1]).lower()
                        cite_paper_date = self.time_transform1(cite_paper_massage.split('+')[-1])
                        if cite_paper_date >= '1988-00':
                            self.paper_dict['cite_paper'][cite_paper_name] = {}
                            self.paper_dict['cite_paper'][cite_paper_name]['author'] = self.clearify_cite_paper_authors(cite_paper_authors)
                            self.paper_dict['cite_paper'][cite_paper_name]['publisher'] = cite_paper_journal
                            self.paper_dict['cite_paper'][cite_paper_name]['publish_date'] = cite_paper_date 
                        else:
                            continue
            else:
                pass                
        except:
            #print(self.title)
            #print(cite_paper_massage)
            #print(cite_paper_name)
            print('')
            
    def add_paper_attribute(self):
        #增加数据
        self.add_publish_date()
        self.add_publisher()
        self.add_cited_num()
        self.add_time_cited()
        self.add_abstract()
        self.add_keyword()
        self.add_keyword_plus()
        self.add_author()
        self.add_university()
        self.add_doi()
        self.add_cite_paper()
#        print(self.title+'    success')
                
    def init_paper_dict(self):
        self.paper_dict = {'ego_attribute':{},'cite_paper':{},'paper_cited':{}}
        self.paper_dict['ego_attribute'] = {'author':{},'publisher':{},'doi':{},
                                        'publish_date':{},'cited_num':{},
                                        'time_cited':{},'abstract':{},'keywords':{},
                                        'keyword_plus':{},'university':[]}
        self.paper_dict['cite_paper'] = {}
        self.add_paper_attribute()
        return self.paper_dict              
#'author':[],'publisher':[],'publish_date':[]
############################################################
################增加反向寻找流程提高信息量######################
############################################################
def csv_to_nod(path_csv,path_nod, path_pkl):
    csvnames = os.listdir(path_csv)
    paper_all = []
    paper_dict = {}
    repeat_list = []
    for csvname in csvnames:
        paper_title = []                                  #转化为小写，并去除空格
        filepath = path_csv+'/'+csvname   
        with open(filepath,mode='r',encoding='ISO-8859-1',newline='') as csv_file:
            csv_reader_lines = csv.reader(csv_file)       #逐行读取csv文件
            next(islice(csv_reader_lines,1), None)   
            for one_line in csv_reader_lines:                          
                if one_line[0].lower() not in paper_title:     #加入文档去重
                    paper_title.append(' '.join(one_line[0].lower().split()))
                    try:
                        one_paper = paper(one_line)
                    except:
                        print(one_line)
                        one_paper = paper(one_line)
                    try:
                        if one_paper.time_transform() >= '1988-00':
                            if one_paper.title in paper_dict.keys():
                                repeat_list.append(one_paper.title+'_'+one_paper.publisher) 
                                paper_dict[one_paper.title+'_'+one_paper.publisher] = one_paper.init_paper_dict()
                                print(one_paper.title+'_'+one_paper.publisher)
                            else:
                                paper_dict[one_paper.title] = one_paper.init_paper_dict()
                        else:
                            continue
                    except:
                        pass
                else:
                    continue
        paper_all.extend(paper_title)        
    for key,value in paper_dict.items(): 
        one_paper = [key,json.dumps(value)]
        with open(path_nod+'/paper_nod.csv', "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(one_paper)
            f.close()
            
    #######序化模型###########
    paper_dict_file = open(path_pkl, 'wb')
    pkl.dump(paper_dict, paper_dict_file)
    paper_dict_file.close()


if __name__ == '__main__':  
    os.chdir("D:/bigdatahw/pan_paper/潘老师")
    path_csv = 'D:/bigdatahw/pan_paper/潘老师/original_data(corresponding to journal nod)'
    path_nod = 'D:/bigdatahw/pan_paper/潘老师/nod'
    path_pkl = 'D:/bigdatahw/pan_paper/潘老师/pkl/paper_dict.pkl'
    
    csv_to_nod(path_csv,path_nod,path_pkl)
    
    #请注意这很重要因为不仅仅是杂志很有可能是已出版的图书
    

    
    
