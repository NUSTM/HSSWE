#coding=utf8
import os, sys
import datetime
from tool import performance
from tool import pytc
from tool.lbsa import LBSA
import getopt

FNAME_LIST = ['negative', 'positive']
def load_data(fname_list, class_list):
    doc_str_list = []
    doc_class_list = []
    for doc_fname, class_fname in zip(fname_list, class_list):
        doc_str_list_one_class = [x.strip() for x in open(doc_fname, 'r').readlines()]
        doc_str_list.extend(doc_str_list_one_class)
        doc_class_list.extend([class_fname] * len(doc_str_list_one_class))
    return doc_str_list, doc_class_list

def build_samps(term_dict, doc_class_list, doc_terms_list, lexicon, term_weight='BOOL'):
    samp_dict_list = []
    samp_class_list = []
    test = LBSA(lexicon)
    for k, _ in enumerate(doc_class_list):
        samp_class = doc_class_list[k]
        samp_class_list.append(samp_class)
        doc_terms = doc_terms_list[k]
        res = test.build_sample(doc_terms)
        samp_dict = dict(zip(range(1, len(res) + 1), res))
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def unsupervised_predict(doc_uni_token, lexicon):
    '''
    unsurprised evaluation
    '''
    samp_class_list = []
    test = LBSA(lexicon)
    for k, _ in enumerate(doc_uni_token):
        score = test.build_sample(doc_uni_token[k], False)
        if score >= 0:
            samp_class_list.append(2)
        else:
            samp_class_list.append(1)
    return samp_class_list

def supervised_predict(train_dir, test_dir, lexicon):
    if not os.path.exists('result'):
        os.makedirs('result')
    fname_samp_train = 'result' + os.sep + 'train.samp'
    fname_samp_test = 'result' + os.sep + 'test.samp'
    fname_model = 'result' + os.sep +'liblinear.model'
    fname_output= 'result'+ os.sep + 'test.out'
    class_list = range(1, len(FNAME_LIST) + 1)
    doc_str_list_train, doc_class_list_train = load_data([train_dir + os.sep + x for x in FNAME_LIST], class_list)
    doc_uni_token_train = pytc.get_doc_unis_list(doc_str_list_train)
    term_set = pytc.get_term_set(doc_uni_token_train)
    term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
    print("building samps......")
    samp_list_train, class_list_train = build_samps(term_dict, doc_class_list_train, doc_uni_token_train, lexicon)
    print("saving samps......")
    pytc.save_samps(samp_list_train, class_list_train, fname_samp_train)
    res = ''
    doc_str_list_test, doc_class_list_test = load_data([test_dir + os.sep + x for x in FNAME_LIST], class_list)
    doc_uni_token_test =  pytc.get_doc_unis_list(doc_str_list_test)
    samp_list_test, class_list_test = build_samps(term_dict, doc_class_list_test, doc_uni_token_test, lexicon)
    pytc.save_samps(samp_list_test, class_list_test, fname_samp_test)
    print('start training......')
    pytc.liblinear_exe(fname_samp_train, fname_samp_test, fname_model, fname_output)
    samp_class_list_linear = [int(x.split()[0]) for x in open(fname_output).readlines()[1:]]
    class_dict = dict(zip(class_list, ['neg', 'pos']))
    print('evaluation...')
    result_dict = performance.demo_performance(samp_class_list_linear, doc_class_list_test, class_dict)
    for key in ['macro_f1']:
        res += key + ': ' + str(round(result_dict[key]*100, 4))+'%  '
    print(res.rstrip('\t') + '\n')

def start_demo(train_dir, test_dir, lexicon, supervised):
    if supervised:
        supervised_predict(train_dir, test_dir, lexicon)
    else:
        class_list = range(1, len(FNAME_LIST) + 1)
        doc_str_list_test, doc_class_list_test = load_data([test_dir + os.sep + x for x in FNAME_LIST], class_list)
        doc_uni_token_test = pytc.get_doc_unis_list(doc_str_list_test)
        class_list_test = unsupervised_predict(doc_uni_token_test, lexicon)
        print('start training......')
        class_dict = dict(zip(class_list, ['neg', 'pos']))
        print('evaluation...')
        res = ""
        result_dict = performance.demo_performance(class_list_test, doc_class_list_test, class_dict)
        for key in ['acc']:
            res += key + ':' + str(round(result_dict[key] * 100, 4)) + '%  '
        print(res.rstrip('\t') + '\n')
        
def usage():
    msg = '''
    Usage: evaluation.py [options] [parameters]
    Options:  -h, --help, display the usage of evaluation.py
              --train, training data directory 
              --test, test data directory 
              --lexicon, the sentiment lexicon path
              -s, --supervised, supervised evaluation
           '''
    return  msg

if __name__ == '__main__':
    supervised = False
    options, args = getopt.getopt(sys.argv[1:], "hs", ["help", "train=", "test=", "lexicon=", 'supervised'])
    
    for name,value in options:
        if name in ("-h","--help"):
            print(usage())
            sys.exit()
        if name in ("--train"):
            train_dir = value
        if name in ("--test"):
            test_dir = value
        if name in ("--lexicon"):
            lexicon = value
        if name in ("-s", "--supervised"):
            supervised=True

    start_time = datetime.datetime.now()
    start_demo(train_dir, test_dir, lexicon, supervised)
    end_time = datetime.datetime.now()
    print('Done!', '\nSeconds cost:', (end_time - start_time).seconds)