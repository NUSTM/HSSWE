import collections
import numpy as np

sent_index= 0

def load_data(filename, min_len=6):
    result, label= [],[]
    docs_str = open(filename, 'r').read().decode('utf-8','ignore').encode('utf-8','ignore')
    for line in docs_str.split('\n'):
        if len(line.split()) >= min_len:
            item = line.split()
            result.append(item[1:])
            label.append(int(item[0]))
    return result, np.array(label)

def load_sent_lexicon(filename, dictionary, soft=True):
    '''
    dictionary:the map of word and its index
    '''
    sent_dict = {}
    with open(filename, 'r') as fin:
        for line in fin:
            word, sent_val = line.strip('\n').split('\t')
            if word in dictionary:
                word_index, sent_val = dictionary[word], float(sent_val)
                if soft:
                    sent_dict[word_index] = [sent_val, 1 - sent_val]\
                        if sent_val < 0 else [1 - sent_val, sent_val]
                else:
                    sent_dict[word_index] = [1.0, 0] if sent_val < 0 else [0, 1.0]
    return sent_dict

def build_dataset(data_list):
    count = [['UNK', -1]]
    count.extend([item for item in collections.Counter([word for doc in data_list for word in doc]).most_common() if item[1] > 5])
    count.extend([['voc_FILL_WORD_voc', 0]])
    sents_len_list = [len(doc) for doc in data_list]
    max_sent_len, vocabulary_size = max(sents_len_list), len(count)
    print('Token Num:%d'% (sum(sents_len_list)))
    dictionary = dict(zip([word[0] for word in count], range(vocabulary_size)))
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for i, doc in enumerate(data_list):
        for j, word in enumerate(doc):
            if word in dictionary: data_list[i][j] = dictionary[word]
            else:data_list[i][j] = 0; count[0][1]+=1
        if sents_len_list[i] < max_sent_len: data_list[i].extend([vocabulary_size - 1] * (max_sent_len - sents_len_list[i]))
    data_list, sents_len_list = np.array(data_list), np.array(sents_len_list)
    return data_list, sents_len_list, reverse_dictionary, vocabulary_size, max_sent_len, dictionary

def generate_batch(batch_size, data_list, sents_len_list, word_sent_dict):
    '''
    get word, word_sent ,sent index
    '''
    global sent_index
    words_batch, labels_batch, sents = [], [], []
    for _ in range(batch_size):
        sents.append(sent_index)
        for data_index in range(sents_len_list[sent_index]):
            word = data_list[sent_index][data_index]
            if word_sent_dict.has_key(word):
                sent_val = word_sent_dict[word]
                words_batch.append(word)
                labels_batch.append(sent_val)
        sent_index = (sent_index + 1) % len(data_list)
    return words_batch, labels_batch, sents
    
def save_embeddings(embeddings, dictionary, vocabulary_size, filename):
    fout = open(filename, 'wb')
    for i in range(vocabulary_size):
        fout.write(dictionary[i] + '\t' + ' '.join(map(str, embeddings[i])) + '\n')
        
def save_lexicon(embeddings, dictionary, vocabulary_size, filename):
    fout = open(filename, 'wb')
    for i in range(vocabulary_size):
        fout.write(dictionary[i] + '\t' + str(embeddings[i][1]-embeddings[i][0]) + '\n')
