# -*- coding: utf-8 -*-
from __future__ import division
import os
import math

def calc_acc(result,label):
    if len(result) != len(label):
        print 'Error: different lenghts!'
        return 0
    else:
        samelist = [int(str(x) == str(y)) for (x, y) in zip(result, label)]
        acc = float((samelist.count(1)))/len(samelist)
        return acc

def calc_recall(result,label,class_dict):
    '''class_dict中键为类别号，需要与label中保持一致；键值对应的是类别名称'''
    recall_dict = {}
    for l in class_dict:
        #判定为l且判定正确
        true_positive = sum([1 for (x, y) in zip(result,label) if x==l and y==l])
        #判定不是l但判断错误，且真实类别为l
        false_negative = sum([1 for (x, y) in zip(result,label) if x!=l and y==l])
        c = class_dict[l]
        if (true_positive+false_negative)!=0:
            recall_dict[c] = true_positive/(true_positive+false_negative)
        else:
            recall_dict[c] = 0

    return recall_dict

def calc_precision(result,label,class_dict):
    '''class_dict中键为类别号，需要与label中保持一致；键值对应的是类别名称'''
    precision_dict = {}
    for l in class_dict:
        #判定为l且判定正确
        true_positive = sum([1 for (x, y) in zip(result,label) if x==l and y==l])
        #判定为l但判断错误
        false_positive = sum([1 for (x, y) in zip(result,label) if x==l and y!=l])
        c = class_dict[l]
        if (true_positive+false_positive)!=0:
            precision_dict[c] = true_positive/(true_positive+false_positive)
        else:
            precision_dict[c] = 0
    return precision_dict

def calc_fscore(result,label,class_dict):
    '''计算F1值'''
    recall_dict = calc_recall(result,label,class_dict)
    precision_dict = calc_precision(result,label,class_dict)
    fscore_dict = {}
    for l in class_dict:
        c = class_dict[l]
        fscore_dict[c] = fscore(recall_dict[c],precision_dict[c])
    return fscore_dict

def fscore(r,p):
    if r+p!=0:
        return (2*r*p)/(r+p)
    else:
        return 0

def calc_macro_average(result,label,class_dict):
    '''计算宏平均recall,precision,F1'''
    recall_dict = calc_recall(result,label,class_dict)
    precision_dict = calc_precision(result,label,class_dict)
    class_num = len(class_dict.keys())
    macro_dict = {}
    macro_dict['macro_r'] = sum([recall_dict[class_dict[l]] for l in class_dict])/class_num
    macro_dict['macro_p'] = sum([precision_dict[class_dict[l]] for l in class_dict])/class_num
    # macro_dict['macro_f1'] = fscore(macro_dict['macro_r'],macro_dict['macro_p'])
    macro_f1_lst = [fscore(recall_dict[class_dict[l]], precision_dict[class_dict[l]]) for l in class_dict]
    # macro_dict['macro_f1'] = sum([x for x in macro_f1_lst])/class_num
    macro_dict['macro_f1'] = fscore(macro_dict['macro_r'], macro_dict['macro_p'])
    return macro_dict

def calc_kappa(result,label,class_dict):
    '''计算kappa系数'''
    samp_num = len(result)
    po = sum([1 for (x, y) in zip(result,label) if x==y])/samp_num
    pe = 0
    for l in class_dict:
        pe += (result.count(l)*label.count(l))/(samp_num*samp_num)
    k = (po-pe)/(1-pe)
    return k

def demo_performance(result,label, class_dict):
    '''计算所有指标'''
    data = {}
    data['acc'] = calc_acc(result,label)
    recall = calc_recall(result,label,class_dict)
    precision = calc_precision(result,label,class_dict)
    fscore = calc_fscore(result,label,class_dict)
    macro_avg = calc_macro_average(result,label,class_dict)

    # data['kappa'] = calc_kappa(result,label,class_dict)
    data['macro_r'],data['macro_p'],data['macro_f1'] = macro_avg['macro_r'],\
    macro_avg['macro_p'], macro_avg['macro_f1']

    for l in class_dict:
        c = class_dict[l]
        data['r_'+c] = recall[c]
        data['p_'+c] = precision[c]
        data['f1_'+c] = fscore[c]
    return data

def demo_cv_performance(output_dir, fold_num, class_dict, classifier_name):
    '''计算n折交叉验证下的平均指标'''
    lst = []
    for fold_id in range(1,fold_num + 1):
        fold_data = {}
        # label_fname = output_dir+os.sep+'fold'+str(fold_id)+os.sep+'test'+os.sep+'test.samp'
        label_fname = output_dir+os.sep+'fold'+str(fold_id)+os.sep+'test'+os.sep+'test_label'
        result_fname = output_dir+os.sep+'fold'+str(fold_id)+os.sep+'test'+os.sep+classifier_name+'.result'
        start = 0
        if classifier_name == 'lg' or classifier_name == 'svm':
            start += 1
        label = [x.strip() for x in open(label_fname).readlines()]
        result = [x.split()[0] for x in open(result_fname).readlines()[start:]]

        fold_data['acc'] = calc_acc(result,label)
        fold_data['recall'] = calc_recall(result,label,class_dict)
        fold_data['precision'] = calc_precision(result,label,class_dict)
        fold_data['fscore'] = calc_fscore(result,label,class_dict)
        fold_data['macro_avg'] = calc_macro_average(result,label,class_dict)
        fold_data['kappa'] = calc_kappa(result,label,class_dict)
        lst.append(fold_data)

    avg_data = {}
    avg_data['acc'] = sum([data['acc'] for data in lst])/fold_num
    avg_data['kappa'] = sum([data['kappa'] for data in lst])/fold_num
    avg_data['macro_r'] =  sum([data['macro_avg']['macro_r'] for data in lst])/fold_num
    avg_data['macro_p'] =  sum([data['macro_avg']['macro_p'] for data in lst])/fold_num
    avg_data['macro_f1'] =  sum([data['macro_avg']['macro_f1'] for data in lst])/fold_num
    for l in class_dict.keys():
        c = class_dict[l]
        avg_data['r_'+c] = sum([data['recall'][c] for data in lst])/fold_num
        avg_data['p_'+c] = sum([data['precision'][c] for data in lst])/fold_num
        avg_data['f1_'+c] = sum([data['fscore'][c] for data in lst])/fold_num

    return avg_data

def classify(score_fname,res_fname,pos_num,neg_num):
    score_list = [float(x.strip()) for x in open(score_fname).readlines()]
    res_list = []
    for i in range(len(score_list)):
        if score_list[i] > pos_num:
            res_list.append('1')
        elif score_list[i] < neg_num:
            res_list.append('-1')
        else:
            res_list.append('0')
    f = open(res_fname,'w')
    f.writelines([x+'\n' for x in res_list])
    f.close()