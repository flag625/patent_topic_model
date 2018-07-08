#-*- coding: utf-8 -*-
#英文预处理：单词分割；去除停用词；去除标点、数字；词形还原；词干提取；去除非英语单词的内容
#针对德温特专利记录的摘要，Excel格式
#处理单个Excel文件
#统计词频？，增加停用词表
import codecs
import pandas as pd
import json
import os
import string
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet

#英文预处理
class NlpPreProcess(object):
    """preprocess the html of gdelt-data
    arg: str
        filepath of the stoplist-file
    function:
        preprocess_file(doc): preprocess a doc, delete the punctuation,digit,meanningless word"""

    def __init__(self,stopfile=None):
        super(NlpPreProcess,self).__init__()
        self.wml = WordNetLemmatizer() #词形还原
        self.ps = PorterStemmer() #词干提取
        with codecs.open(stopfile,'r','utf-8') as f:
            self.stoplist = f.read().splitlines()
        print("the num of stopwords is %s"%len(self.stoplist))
        #self.downlist = downlist #文件夹下已经处理过的文档，避免重复抓取
        self.allnum = 0

    def preprocess_file(self,filepath,outputpath=None):
        '''去标点，去数字，分割成单词，词形还原'''
        print('begin process %s' % filepath)
        df = pd.read_excel(filepath)
        #df['cut_content'] = None
        dict_w = {}
        fre = Counter()
        save_file = codecs.open(u'D:\python-数据挖掘\patent_topic_model\down' + os.sep + 'docwords.json', 'w', 'utf-8')
        #词频文件
        fre_file = u'D:/python-数据挖掘/patent_topic_model/fre/fre_clean.xlsx'
        print('begin process %s'%filepath)
        num = 0
        for dic in df.ix[:,7]:
            doc = str(dic).lower()
            for c in string.punctuation: #去标点
                doc = doc.replace(c,' ')
            for c in string.digits: #去数字
                doc = doc.replace(c,'')
            #doc = dic.lower()
            doc = nltk.word_tokenize(doc) #分割成单词
            #只保留特定词性单词，如名词
            #filter = nltk.pos_tag(doc)
            #doc = [w for w, pos in filter if pos.startswith("NN")]
            cleanDoc = []
            #只保留长度不小于3的单词，去除停用词，验证是否为英文单词（利用wordnet）
            for word in doc:
                if len(word) >= 3 and wordnet.synsets(word) and word not in self.stoplist:
                    word = self.wml.lemmatize(word) #词形还原
                   #word = self.ps.stem(word) #词干提取
                    cleanDoc.append(word)
            #df['cut_content'][num] = ' '.join(cleanDoc) #保存为Excel
            fre += Counter(cleanDoc) #统计词频
            dict_w['content'] = ' '.join(cleanDoc) #保存为Json
            json.dump(dict_w, save_file, ensure_ascii=False)
            save_file.write('\n')
            num += 1
        pd.DataFrame.from_dict(dict(fre),orient='index').to_excel(fre_file)
        print('the num of valid docs is : %s'%num)
        #df['cut_content'].to_excel(outputpath)
        print('---'*20)

if __name__ == '__main__':
    filepath = u'D:/python-数据挖掘/patent_topic_model/data/柴油推进系统-强相关数据-169（去中国）.xlsx'
    #outputpath = u'D:/python-数据挖掘/patent_topic_model/down/wwordcut_test.xlsx'
    stopword_filepath = None
    stopword_filepath = u'D:/python-数据挖掘/patent_topic_model/stop_word/stopword_eng.txt'
    nlp_preprocess = NlpPreProcess(stopword_filepath)
    nlp_preprocess.preprocess_file(filepath)