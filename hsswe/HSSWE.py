#coding=utf8
'''
  Title: HSSWE
  Author: Leyi Wang
  Date: Last update 2017-11-08
  Email: leyiwang.cn@gmail.com
'''
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
import os, sys, math, random, datetime
from utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_file_path', 'data/process.tw', 'training file')
tf.app.flags.DEFINE_string('sent_lexicon_path', 'data/word-level-supervision',
                           'the path of word-level sentiment supervision')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
tf.app.flags.DEFINE_integer('max_iter_num', 7601, 'max number of iteration')
tf.app.flags.DEFINE_integer('display_step', 400, 'number of steps for print loss')
tf.app.flags.DEFINE_integer('test_sample_num', 10000, 'number of sample for test set')
tf.app.flags.DEFINE_float('alpha', 0.1, 'hyper-parameter alpha')
tf.app.flags.DEFINE_float('lr', 0.3, 'learning rate')
tf.app.flags.DEFINE_bool('soft', True, 'soft or hard')

def start_demo():
    print("Reading data...")
    dataset, sents_label = load_data(FLAGS.train_file_path)
    print("End Reading!")
    data_list, sents_len_list, reverse_dictionary, vocabulary_size, max_sent_len, dictionary = build_dataset(dataset)
    sample_num = data_list.shape[0]
    print('Load PMI-SO supervised info...')
    sentiment_dict = load_sent_lexicon(FLAGS.sent_lexicon_path, dictionary, FLAGS.soft)
    stop = sample_num - FLAGS.test_sample_num
    data_list, sents_label, sents_len_list, test_list, test_labels, sents_len_test = data_list[:stop], sents_label[:stop], \
        sents_len_list[:stop], data_list[stop:], sents_label[stop:], sents_len_list[stop:]
    embeddings = tf.Variable(tf.concat(0, [tf.random_uniform([vocabulary_size-1, FLAGS.embedding_size], -1.0, 1.0, seed=0), tf.zeros([1, FLAGS.embedding_size])]))
    with tf.name_scope(name="input"):
        sents_inputs = tf.placeholder(tf.int32, shape=[None, max_sent_len])
        sents_labels = tf.placeholder(tf.int32, shape=[None])
        sents_len_weight = tf.placeholder(tf.float32, shape=[None])
        word_input = tf.placeholder(tf.int32, shape=[None])
        word_score = tf.placeholder(tf.float32, shape=[None, 2])
    
    with tf.name_scope(name="word_level_sa"):
        word_embed = tf.nn.embedding_lookup(embeddings, word_input)
        biases = tf.Variable(tf.zeros([2]))
        weights = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, 2], stddev=1.0 / math.sqrt(FLAGS.embedding_size),seed=1))
        word_y = tf.nn.softmax(tf.matmul(word_embed, weights) + biases)
        
    with tf.name_scope(name="sent_level_sa"):
        embedding_input = tf.nn.embedding_lookup(embeddings, sents_inputs)
        sent_embed = tf.matmul(tf.diag(1 / sents_len_weight), tf.reduce_sum(embedding_input, 1))
        sent_y = tf.nn.softmax(tf.matmul(sent_embed, weights)+ biases)
    
    with tf.name_scope(name = "update_filled_embedding"):
        embed_new = tf.concat(0, [embeddings[:vocabulary_size-1], tf.zeros([1, FLAGS.embedding_size])])
        embed_update = tf.assign(embeddings, embed_new)
        
    with tf.name_scope(name = "acc"):
        sent_one_hot_labels = onehot_labels = tf.one_hot(indices=sents_labels, depth=2)
        correct_prediction = tf.equal(tf.argmax(sent_y, 1), tf.argmax(sent_one_hot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    with tf.name_scope(name = "loss"):
        word_sentiment_loss = FLAGS.alpha * tf.reduce_mean(-tf.reduce_sum(word_score * tf.log(word_y + 1e-5), reduction_indices=[1]), name="word_loss")*200
        sentiment_loss = (1-FLAGS.alpha) * tf.reduce_sum(-tf.reduce_sum(sent_one_hot_labels * tf.log(sent_y + 1e-5), reduction_indices=[1]), name="sent_loss")
        loss = sentiment_loss + word_sentiment_loss
        
    with tf.name_scope(name = "optimizer"):
        optimizer = tf.train.AdagradOptimizer(FLAGS.lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=50)
    tf.scalar_summary('accuracy', accuracy)
    tf.scalar_summary('loss', loss)
    merged_summary = tf.merge_all_summaries()

    with tf.Session() as session:
        session.run(init)
        model_path, graph_path = 'model', 'graph'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)
        writer = tf.train.SummaryWriter(graph_path, session.graph)
        avg_loss, avg_train_acc = 0, 0
        for step in xrange(FLAGS.max_iter_num):
            batch_inputs, batch_labels, sents_index = generate_batch(FLAGS.batch_size, data_list, sents_len_list, sentiment_dict)
            batch_sents_inputs, batch_sents_labels, batch_sents_len_weight = data_list[sents_index], sents_label[sents_index], sents_len_list[sents_index]
            feed_dict = {word_input: batch_inputs, word_score: batch_labels, sents_inputs:batch_sents_inputs, \
                         sents_labels:batch_sents_labels, sents_len_weight: batch_sents_len_weight}
            session.run([optimizer], feed_dict=feed_dict)
            session.run(embed_update)
            loss_val, train_acc, summary = session.run([loss, accuracy, merged_summary], feed_dict=feed_dict)
            avg_loss += loss_val; avg_train_acc += train_acc
            writer.add_summary(summary, step)
            if step % FLAGS.display_step == 0:
                save_path = saver.save(session, model_path + os.sep + str(step) + ".ckpt")
                print("Save to path: ", save_path)
                test_sents_len_weight = sents_len_test
                feed_dict = {sents_inputs:test_list, sents_labels:test_labels, sents_len_weight: test_sents_len_weight}
                if step > 0: 
                    avg_loss /= FLAGS.display_step
                    avg_train_acc/=FLAGS.display_step

                print("Iter ", step, ": ", avg_loss, 'sent_loss:', 'train_acc:', avg_train_acc, 'test_acc:',\
                                                                  session.run(accuracy, feed_dict=feed_dict))
                avg_loss, avg_train_acc = 0, 0
        final_embeddings = session.run(embeddings)
        save_embeddings(final_embeddings, reverse_dictionary, vocabulary_size, 'embedding')

def main(_):
    start_time = datetime.datetime.now()
    start_demo()
    end_time = datetime.datetime.now()
    print('Done!', '\nSeconds cost:', (end_time - start_time).seconds)

if __name__=='__main__':
    tf.app.run()
