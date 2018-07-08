#-*- coding: utf-8 -*-
#计算LDA模型的困惑度
import sys
import imp
imp.reload(sys)
import math
import os
from gensim.corpora import Dictionary
from gensim import corpora,models
from datetime import datetime
import logging
logging.basicConfig(format='(asctime)s : %(levelname)s : %(message)s : ',level=logging.INFO)

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

if __name__ == '__mian__':
    middatafolder = r'D:\python-数据挖掘\patent_topic_model\mid'+os.sep
    dictionary_path = middatafolder+'dictionary.dict'
    corpus_path = middatafolder+'corpus.mm'
    ldamodel_path = middatafolder+'lda_tfidf_10_3000.lda'
    dictionary = corpora.Dictionary(dictionary_path)
    corpus = corpora.MmCorpus.(corpus_path)
    lda_multi = models.ldamodel.LdaModel.load(ldamodel_path)
    num_topics = 10
    testset = []
    for i in range(corpus.num_docs):
        testset.append(corpus[i])
    prep = perolexity(lda_multi,testset,dictionary,len(dictionary.keys()),num_topics)