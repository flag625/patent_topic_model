#-*- coding: utf-8 -*-
#LDA可视化

import gensim
from gensim import models
import pyLDAvis.gensim

def lda_vis():
    dictionary = gensim.corpora.Dictionary.load(u'D:\python-数据挖掘\patent_topic_model\mid\dictionary.dict')
    corpus = gensim.corpora.MmCorpus(u'D:\python-数据挖掘\patent_topic_model\mid\corpus.mm')
    lda = models.ldamodel.LdaModel.load(u'D:\python-数据挖掘\patent_topic_model\mid\lda_tfidf_10_3000.lda')
    vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, u'D:\python-数据挖掘\patent_topic_model\pyLDAvis\lda.html')
    return 0

if __name__ == '__main__':
    lda_vis()