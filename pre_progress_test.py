#-*- coding: utf-8 -*-
#英文预处理：单词分割；去除停用词；去除标点、数字；词形还原；词干提取；去除非英语单词的内容

import codecs
import json
import pandas as pd
import os
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from datetime import datetime
import multiprocessing

import sys
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf-8') #python2
path = sys.path[0]

class NlpPreProcess(object):
    """preprocess the html of gdelt-data
    arg: str
        filepath of the stoplist-file
    function:
        preprocess_folder(source_folder, dest_folder): preprocess all files in the source-folder, and save the results to dest-folder
        preprocess_file(doc): preprocess a doc, delete the punctuation,digit,meanningless word
        generate_dict_from_file(filename): generator to parse dict from json file >>>nlp_preprocess = NlpPreProcess('')"""

    def __init__(self,stopfile,downlist):
        super(NlpPreProcess,self).__init__()
        self.wml = WordNetLemmatizer() #词形还原
        self.ps = PorterStemmer() #词干提取
        with codecs.open(stopfile,'r','utf-8') as f:
            self.stoplist = f.read().splitlines()
        print("the num of stopwords is %s"%len(self.stoplist))
        self.downlist = downlist #文件夹下已经处理过的文档，避免重复抓取
        self.allnum = 0

    def preprocess_folder(self,source_folder,dest_floder):
        '''process all docs in all files, and save the results to according docwords file'''
        stime = datetime.now()
        for filename in os.listdir(source_folder):
            self.preprocess_file(filename,source_folder,dest_floder)
        print("the num of all valid docs in : %s"%self.allnum)

    def preprocess_file(self,filename,source_floder,dest_floder):
        '''去标点，去数字，分割成单词，词形还原'''
        saveFileName = 'docwords'+filename[-1]
        if saveFileName in self.downlist:
            return 0
        print('begin process %s'%filename)
        save_file = codecs.open(dest_floder+os.sep+saveFileName,'w','utf-8')
        num = 0
        stime = datetime.now()
        for dic in self.generate_dict_from_file(source_floder+os.sep+filename):
            doc = dic['content'].lower()
            for c in string.punctuation: #去标点
                doc = doc.replace(c,'')
            for c in string.digits: #去数字
                doc = doc.replace(c,'')
            doc = nltk.word_tokenize(doc) #分割成单词
            #只保留特定词性单词，如名词
            #filter = nltk.pos_tag(doc)
            #doc = [w for w, pos in filter if pos.startswith("NN")]
            cleanDoc = []
            #只保留长度不小于3的单词，去除停用词，验证是否为英文单词（利用wordnet）
            for word in doc:
                if len(word) >= 3 and word not in self.stoplist and wordnet.synsets(word):
                    word = self.wnl.lemmatize(word) #词形还原
                    #word = self.ps.stem(word) #词干提取
                    cleanDoc.append(word)
            dic['content'] = ' '.join(cleanDoc)
            json.dump(dic,save_file,ensure_ascii=False)
            save_file.write('\n')
            num += 1
        print('time cost is : %s'%(datetime.now()-stime))
        print('the num of valid docs is : %s'%num)
        print('---'*20)
        self.allnum += num
        return num


if __name__ == '__main__':
    source_folder = path + os.sep + 'cleanHtml'
    dest_folder = path + os.sep + 'docword'
    stopword_filepath = path + os.sep + 'stoplist.csv'
    process_num = 6 #设置多进程数量
    downlist = os.listdir(dest_folder)
    nlp_preprocess = NlpPreProcess(stopword_filepath,downlist)
    nlp_preprocess.preprocess_file('sogou.json',source_folder,dest_folder)
    #nlp_preprocess.preprocess_folder_multiprocess(source_floder,dest_floder,process_num)
