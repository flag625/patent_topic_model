#-*- coding: utf-8 -*-
import sys
import imp
imp.reload(sys)
import os
import codecs
from gensim.corpora import Dictionary
from gensim import corpora,models
from datetime import datetime
import math
import platform
import logging
import pyLDAvis.gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s :',level=logging.INFO)

platform_info = platform.platform().lower()
if 'windows' in platform_info:
    code = 'gbk'
elif 'linux' in platform_info:
    code = 'utf-8'
path = sys.path[0]

class GLDA(object):
    """docstring for GdeltGLDA"""
    def __init__(self,stopfile=None):
        super(GLDA, self).__init__()
        if stopfile:
            with codecs.open(stopfile,'r',code) as f:
                self.stopword_list = f.read().split(' ')
            print('the num of stopwords is : %s'%len(self.stopword_list))
        else:
            self.stopword_list = None

    def lda_train(self,num_topics,datafolder,middatafolder,dictionary_path=None,corpus_path=None,iterations=500,passes=1,workers=3):
        time1 = datetime.now()
        num_docs = 0
        doclist = []
        if not corpus_path or not dictionary_path:
            #若无字典或无corpus，则读取预处理后的docword。
            #一般第一次运行都需要读取，在后期调参时，可直接传入字典与corpus路径。
            for filename in os.listdir(datafolder):
                #读取datafolder下的语料
                with codecs.open(datafolder+filename,'r',code) as source_file:
                    for line in source_file:
                        num_docs += 1
                        if num_docs%100000 == 0:
                            print('%s, %s'%(filename,num_docs))
                        #doc = [word for word in doc if word not in self.stopword_list]
                        doclist.append(line.split(' '))
                print('%s, %s'%(filename,num_docs))
        if dictionary_path:
            dictionary = corpora.Dictionary.load(dictionary_path) #加载字典
        else:
            #构建词汇统计向量并保存
            dictionary = corpora.Dictionary(doclist)
            #dictionary.save(middatafolder+'dictionary.dict')
        if corpus_path:
            corpus = corpora.MmCorpus(corpus_path) #加载corpus
        else:
            corpus = [dictionary.doc2bow(doc) for doc in doclist]
            #corpora.MmCorpus.serialize(middatafolder+'corpus.mm',corpus) #保存corpus
        tfidf = models.TfidfModel(corpus)
        corpusTfidf = tfidf[corpus]
        time2= datetime.now()
        #开始训练
        lda_multi = models.ldamulticore.LdaMulticore(corpus=corpusTfidf,id2word=dictionary,num_topics=num_topics,iterations=iterations,workers=workers,batch=True,passes=passes)
        lda_multi.print_topics(num_topics,20) #输出主题词矩阵
        print('lda training time cost is : %s, all time cost is : %s'%(datetime.now()-time2,datetime.now()-time1))
        #模型的保存/加载
        #lda_multi.save(middatafolder+'lda_tfidf_%s_%s.lda'%(num_topics,iterations))
        #lda = models.ldamodel.LdaModel.load('zhwiki_lda.model') #加载模型
        # save the doc-topic-id
        topic_id_file = codecs.open(middatafolder+'topic.json','w','utf-8')
        for i in range(num_docs):
            topic_id = lda_multi[corpusTfidf[i]][0][0] #取概率最大的主题作为文本所属主题
            topic_id_file.write(str(topic_id)+' ')

        #计算困惑度
        #testset = []
        #for i in range(corpus.num_docs):
            #testset.append(corpus[i])
        #prep = perolexity(lda_multi, testset, dictionary, len(dictionary.keys()), num_topics)

        #LDA模型可视化
        vis = pyLDAvis.gensim.prepare(lda_multi, corpus, dictionary)
        pyLDAvis.show(vis)

def perolexity(ldamodel,testset,dictionary,size_dictionary,num_topics):
    """计算LDA模型的困惑度"""
    # dictionary : {7822:'deferment',1841:'circuitry',19202:'fabianism'...}
    print('the info of this ldamodel: \n')
    print('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset),size_dictionary,num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    #储存主题词的概率：[(u'business',0.010020942661849608),(u'family', 0.0088027946271537413)...]
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id,size_dictionary)
        dic = {}
        for word,probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    #储存主题元祖：[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    doc_topics_list = []
    for doc in testset:
        doc_topics_list.append(ldamodel.get_document_topics(doc,minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 #文档的概率
        doc = testset[i]
        doc_word_num = 0 #文档关键词的数量
        for word_id,num in doc:
            prob_word = 0.0 #关键词概率
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_list[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word)
            testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) #困惑度
    print('the perplexity of this ldamodel is %s'%prep)
    return prep



if __name__ == '__main__':
    datafolder = u'D:\python-数据挖掘\patent_topic_model\down' + os.sep  # 预处理后的语料所在文件夹，函数会读取此文件夹下的所有语料文件
    middatafolder = u'D:\python-数据挖掘\patent_topic_model\mid' + os.sep
    dictionary_path = False  # 已处理好的字典，若无，则设置为False
    corpus_path = False  # 对语料处理过后的corpus，若无，则设置为False
    # stopfile = path + os.sep + 'rest_stopwords.txt' # 新添加的停用词文件
    num_topics = 3
    passes = 3  # 这个参数大概是将全部语料进行训练的次数，数值越大，参数更新越多，耗时更长
    iterations = 3000
    workers = 3  # 相当于进程数
    lda = GLDA()
    lda.lda_train(num_topics, datafolder, middatafolder, dictionary_path=dictionary_path, corpus_path=corpus_path,
                  iterations=iterations, passes=passes, workers=workers)