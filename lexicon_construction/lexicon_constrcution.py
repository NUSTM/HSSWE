#coding=utf8
import os, datetime, math, sys, getopt
import numpy as np
from utils import *
from sklearn.utils import shuffle

def walk_dir(data_dir):
    '''
    filelists
    '''
    filelist = []
    for root, dir, files in os.walk(data_dir):
        for f in files:
            fullpath = os.path.join(root, f)
            filelist.append(fullpath)
    return filelist

def load_embeddings(filename):
    '''
    word:embedding
    '''
    embedings_dic = {}
    for index, line in enumerate(open(filename, 'r').readlines()):
        line = line.strip('\n')
        key, value = line.split()[0], line.split()[1:]
        embedings_dic[key] = map(float, value)
    return embedings_dic

def load_seeds(fnamelist, embeddings_dic, label_dict={'negative':0, 'positive':1}):
    '''
        function: if word in embedding dic return embeddings and targets
        data: embeddings of seed words
        targets: labels of seeds
    '''
    data, target = [], []
    for fname in fnamelist:
        data_list = open(fname, 'r').read().split('\n')
        class_label = os.path.split(fname)[1]
        embedding = [embeddings_dic[item] for item in data_list if embeddings_dic.has_key(item)]#seed_word: embedding
        labels = [label_dict[class_label]] * len(embedding)
        data.extend(embedding)
        target.extend(labels)
    return data, target

def load_UBDictory(fnamelist, embedings_dic):
    '''
    word:sim word lis
    '''
    sim_dic = {}
    for filename in fnamelist:
        data_list = [[word.replace(" ", "<w-w>") for word in line.split('\t')]\
                     for line in open(filename, 'r').read().split('\n')]
        for item in data_list:
            key, value = item[0], [w for w in item[1:] if embedings_dic.has_key(w)]
            if embedings_dic.has_key(key) and len(value) > 0:
                if not sim_dic.has_key(key):
                    sim_dic[key]= [word for word in value if embedings_dic.has_key(word)]
                else:
                    cand_words = [word for word in value if embedings_dic.has_key(word)]
                    sim_dic[key] = list(set(sim_dic[key] + cand_words))
    return sim_dic

def build_dataset(sim_dic, embd_sent_dic):
    '''
    word:sim word sent list
    '''
    dic = {}
    for key in sim_dic.keys():
        dic[key]  = [embd_sent_dic[word] for word in sim_dic[key]]
    words, sents_list = zip(*sorted(dic.items(), key=lambda x:x[-1]))
    return words, sents_list

def build_extend_lexicon(params, embedings_dic, seeds_dir, ud_dir):
    knn = KNN(params)
    fnamelist, ud_fnamelist, model = walk_dir(seeds_dir), walk_dir(ud_dir), os.path.join('model', 'knn.model')
    ud_sim_dic = load_UBDictory(ud_fnamelist, embedings_dic)#word simwordlist
    seeds_embed, seeds_label = load_seeds(fnamelist, embedings_dic)
    knn.train(seeds_embed, seeds_label, model)
    vocabulary, embeddings = zip(*sorted(embedings_dic.items(), key=lambda x: x[-1]))
    sent_targets = knn.predict(embeddings)
    embd_sent_dic = dict(zip(vocabulary, sent_targets))#word: sent_target
    ud_words, ud_sim_sent_list = build_dataset(ud_sim_dic, embd_sent_dic)
    ud_words_target = knn.ud_predict(ud_sim_sent_list)
    ud_neg_str = '\n'.join([word for i, word in enumerate(ud_words) if ud_words_target[i]==0])
    ud_pos_str = '\n'.join([word for i, word in enumerate(ud_words) if ud_words_target[i]==1])
    open(os.path.join('data', 'extend', 'negative'), 'w').write(ud_neg_str)
    open(os.path.join('data', 'extend', 'positive'), 'w').write(ud_pos_str)

def build_lexicon(embedings_dic):
    fnamelist = walk_dir(os.path.join('data', 'extend'))
    data, target = load_seeds(fnamelist, embedings_dic)
    data, target = shuffle(data, target, random_state=0)
    vocabulary, embeddings = zip(*embedings_dic.items())
    softmax = Softmax(2, 2000)
    softmax.run('model','graph', data, target, embeddings)
    softmax.save_lexicon(vocabulary,'result')

def usage():
    msg = '''
    Usage: lexicon_constrcution.py [options] [parameters]
    Options:  -h, --help, display the usage of lexicon_constrcution.py
              -p, --positive, threshold for positive seeds
              -n, --negative, threshold for negative seeds
              -e, --embedding, the sentiment-aware word representations file path
           '''
    return  msg
if __name__ == "__main__":
    params = {}
    seeds_dir = os.path.join('data','seeds')
    ud_dir = os.path.join('data','ud')
    options, args = getopt.getopt(sys.argv[1:], "hp:n:e:", ["help", "pos=", "neg=", 'embedding='])
    for name,value in options:
        if name in ("-h","--help"):
            print usage()
            sys.exit()
        if name in ("-p","--positive"):
            params['pos'] = int(value)
        if name in ("-n","--negative"):
            params['neg'] = int(value)
        if name in ("-e","--embedding"):
            embedding = value

    start_time = datetime.datetime.now()
    embedings_dic = load_embeddings(embedding)#word:embedding
    build_extend_lexicon(params, embedings_dic, seeds_dir, ud_dir)
    build_lexicon(embedings_dic)
    end_time = datetime.datetime.now()
    print('Done!', '\nSeconds cost:', (end_time - start_time).seconds)