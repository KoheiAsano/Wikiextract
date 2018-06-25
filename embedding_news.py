# -*- coding: utf-8 -*-
"""
News の専門用語を抽出して、元ある辞書と合わせた言葉ベクトル空間(１００次元)への投影

python3 embedding_news "folder_name"
"""

#caluclate processing time
import time
start = time.time()


#define functioin to "Exclude" except political news 
with open("../dic/spet_words.txt", mode='r',encoding = "utf-8") as f:
    s_d = f.read()
    spet_words = s_d.split("\n")
    for word in spet_words:
        if word in  ["","\t","\n"]:
            spet_words.remove(word)

def IsNotPolitical(title,content):
    
    for etsp in spet_words:
        if etsp in title:
#            print(etsp,title)
            return True
        elif etsp in content:
#            print(etsp,title)
            return True
        elif content == "":
            return True
        
    return False

import sys
import glob

#get command argument such as "python embedding_news.py 02"
p = sys.argv
news_csv_path = 'newstext/' +  p[1] + '/*.csv'
CSVfiles = glob.glob(news_csv_path)







import pandas as pd
from normalize import normalize

political_news = pd.DataFrame()
#define set to exclude deplication of news title
titleset = set()

for CSVname in CSVfiles:
    raw_newsdf = pd.read_csv(CSVname,
                            sep=',',
                            encoding='utf-8',
                            index_col=False,
                           )
    #半角→全角,low→upper, digits→0   ex. normalize(Asano8)=ａｓａｎｏ0
    raw_newsdf["title"] = [normalize(t) for t in raw_newsdf["title"]]
    raw_newsdf["content"]  = [normalize(c) for c in raw_newsdf["content"]]
    
    for news_i in raw_newsdf.index:
        
        #exclude duplication
        if raw_newsdf["title"][news_i] in titleset:
            raw_newsdf = raw_newsdf.drop(news_i)
            continue
        titleset.add(raw_newsdf["title"][news_i])

            
        if IsNotPolitical(raw_newsdf["title"][news_i],raw_newsdf["content"][news_i]):
            raw_newsdf = raw_newsdf.drop(news_i)
            continue
    political_news = pd.concat([political_news,raw_newsdf])
    
political_news = political_news.reset_index(drop=True)



#reading based dictionaly
prodicdf = pd.read_csv('../dic/news_proper_pn.txt',
                        sep=',',
                        encoding='utf-8',
                        index_col=False,
                       )
gendicdf = pd.read_csv('../dic/news_general_pn.txt',
                        sep=',',
                        encoding='utf-8',
                        index_col=False,
                       )


import re
import MeCab
mecab = MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd/")


#clean morph-decompose text to proper nouns
def get_prolist(text):
    parsed = mecab.parse(text) 
    lines = parsed.split('\n') 
    lines = lines[0:-2]
    prolist = []
    for word in lines:
        l = re.split('\t|,',word)  
        d = {'Surface':l[0], 'POS1':l[1], 'POS2':l[2], 'BaseForm':normalize(l[7])}
        
        if len(d['BaseForm']) < 2:
            continue
        if d['POS2'] != "固有名詞":
            continue
        if "０" in d['BaseForm']:
            continue
        if "0" in d['BaseForm']:
            continue
        
        #Exclude overlap with based dictionaly
        if str(d['BaseForm']) == "＊":
            d['BaseForm'] = d['Surface']
        if d['BaseForm'] in list(prodicdf["BaseForm"]):
            continue
        if d['BaseForm'] in list(gendicdf["BaseForm"]):
            continue
        prolist.append(d['BaseForm'])
    return(prolist)

#get newspronouns
NewsPronons = []
for i in political_news.index:
    pro_list = get_prolist(political_news["content"][i])
    NewsPronons += pro_list



#exclude words which don't appear more than "10" times
tempdf = pd.DataFrame({"BaseForm":NewsPronons,"dummy":NewsPronons})

new_words = tempdf.groupby(["BaseForm"],as_index = False)
counted_vocab = new_words.count().sort_values(by='dummy', ascending=False)
new_vocab = counted_vocab.rename(columns={'dummy': 'Frequency'})
new_vocab = new_vocab[new_vocab.Frequency >= 10]

target_pronouns = [pro for pro in NewsPronons if pro in list(new_vocab["BaseForm"])]




#prepare TF-IDF filter
from gensim import corpora,models

TF_IDFTEXTDIR = "wiki_AA_text"
txtpath = TF_IDFTEXTDIR + '/*.txt'
TXTfiles = glob.glob(txtpath)


pronouns = [target_pronouns]
wiki_pro = []
for TXTname in TXTfiles:
    with open(TXTname, mode='r',encoding = "utf-8") as f:
        wiki = f.read()
        wiki = normalize(wiki)
        pro_list = get_prolist(wiki)
        pronouns.append(pro_list)
        

TF_IDFdic = corpora.Dictionary(pronouns)
TF_IDFdic.save_as_text("data/wikiTFIDF.txt")

corpus = [TF_IDFdic.doc2bow(p) for p in pronouns]
model = models.TfidfModel(corpus)
tfidf = model[corpus]


tf_idf_list = []
words_list = []
for p in tfidf[0]:
    words_list.append(TF_IDFdic[p[0]])
    tf_idf_list.append(p[1])

TF_IDFdf = pd.DataFrame({'BaseForm': words_list,
                       'TF-IDF':tf_idf_list
                        })

    
#choice "top100" TF-iDF,
TF_IDFdf = TF_IDFdf.sort_values(by='TF-IDF', ascending=False)
new_vocab = TF_IDFdf[:100]
new_vocab = new_vocab.drop(["TF-IDF"],axis=1)


#def function to get coupus
def get_word2vec_vocab(text):
    parsed = mecab.parse(text)
    lines = parsed.split('\n')
    lines = lines[0:-2]
    targetlist = []
    for word in lines:
        l = re.split('\t|,',word)
        d = {'Surface':l[0], 'POS1':l[1], 'POS2':l[2], 'BaseForm':normalize(l[7])}
        if str(d['BaseForm']) == "＊":
            d['BaseForm'] = d['Surface']
            
        if d['BaseForm'] in list(new_vocab["BaseForm"]):
            targetlist.append(d['BaseForm'])
        if d['BaseForm'] in list(prodicdf["BaseForm"]):
            targetlist.append(d['BaseForm'])
        if d['BaseForm'] in list(gendicdf["BaseForm"]):
            targetlist.append(d['BaseForm'])
    return(targetlist)


w2v_target_list = []
for i in political_news.index:
    w2v_target_article = get_word2vec_vocab(political_news["content"][i])
    w2v_target_list.append(w2v_target_article)

news_w2v = models.Word2Vec(w2v_target_list, size=100, window=5, workers=8, min_count=1)
words = list(news_w2v.wv.vocab)
print(words)
print('Vocabulary size: %d' % len(words))
filename = 'data/news_embedding.txt'
news_w2v.wv.save_word2vec_format(filename, binary=False)
news_w2v.save("data/forplot.model")


elapsed_time = time.time() - start
print ("finished_time:{0}".format(elapsed_time) + "[sec]")
