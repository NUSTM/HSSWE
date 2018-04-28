#coding=utf8
import os, datetime, math, sys
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, dic):
        self.threshold_pos = dic['pos']
        self.threshold_neg = dic['neg']
        self.clf = KNeighborsClassifier()
        
    def train(self, X, y, model):
        self.clf.fit(X, y)
        joblib.dump(self.clf, model)
        print('acc:',self.clf.score(X, y))
        return self.clf
    
    def predict(self, X, model=None):
        if model:
            self.clf = joblib.load(model)
        labels = self.clf.predict(X)
        return labels
    
    def ud_predict(self, sim_sent_list):
        result = []
        for i,labels in enumerate(sim_sent_list):
            num_neg, num_pos = [labels.count(i) for i in range(0, 2)]
            if (num_pos > num_neg + self.threshold_pos):
                final_target = 1
            elif (num_neg > num_pos + self.threshold_neg):
                final_target = 0
            else: 
                final_target = -1
            result.append(final_target)
        return result

class Softmax:
    def __init__(self, class_num, batch_size, embed_size = 50):
        self.class_num = class_num
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.embed_idx = 0

    def next_batch(self, datalist, labels):
        data, targets = [], []
        for i in range(self.batch_size):
            data.append(datalist[self.embed_idx])
            targets.append(labels[self.embed_idx])
            self.embed_idx = (self.embed_idx + 1) % len(datalist)
        return data, np.array(targets)
    
    @staticmethod
    def accuracy(xs, y_, sess, prediction, embed_inputs):
        y_pre = sess.run(prediction, feed_dict={embed_inputs: xs})
        correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y_pre, axis=1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(acc)

    def run(self, model, graph_path, data_list, labels, test):
        class_num, embed_size = self.class_num, self.embed_size
        with tf.name_scope('inputs'):
            embed_inputs = tf.placeholder(tf.float32, shape=[None, embed_size], name="x")
            embed_labels = tf.placeholder(tf.float32, shape=[None, class_num], name='y')
        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal([embed_size, class_num], stddev=1.0 / math.sqrt(embed_size)), name='weights')
            biases = tf.Variable(tf.zeros([class_num]), name='bias')
            prediction = tf.nn.softmax(tf.matmul(embed_inputs, weights) + biases, name='output')
        with tf.name_scope(name="loss"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(embed_labels * tf.log(prediction), reduction_indices=[1])) + 0.001 * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))
        with tf.name_scope(name="optimizer_op"):
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.summary.scalar('cost', cross_entropy)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter(graph_path, sess.graph)
            print("Initialized")
            average_loss, avg_train_acc = 0, 0
            onehot_labels = tf.one_hot(indices=labels, depth=class_num)
            labels = sess.run(onehot_labels)
            for step in xrange(1203):
                batch_inputs, batch_labels = self.next_batch(data_list, labels)
                _, loss_val, summary = sess.run([optimizer, cross_entropy, merged], feed_dict={embed_inputs: batch_inputs, embed_labels: batch_labels})
                average_loss += loss_val
                if step % 400 == 0:
                    save_path = saver.save(sess, os.path.join(model, str(step)) + ".ckpt")
                    print("Save to path: ", save_path)
                    if step >= 400: average_loss /= 400
                    print("=> Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
                    writer.add_summary(summary, step)
            self.result = sess.run(prediction, feed_dict={embed_inputs: test})

    def save_lexicon(self, voc, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        class_set = ['negative', 'positive']
        for class_id in range(self.class_num):
            class_label = class_set[class_id]
            lex_list = [(voc[i], str(self.result[i][class_id] - self.result[i][1-class_id])) \
                        for i, item in enumerate(self.result) if self.result[i][class_id]>0.5]
            if class_label=='negative':
                order_list=['\t-'.join(item) for item in sorted(lex_list, key=lambda x:float(x[1]), reverse=True)]
            else:
                order_list=['\t'.join(item) for item in sorted(lex_list, key=lambda x:float(x[1]), reverse=True)]
            doc_str = '\n'.join(order_list)
            fobj = open(save_dir + os.sep + class_label, 'w')
            fobj.write(doc_str)
            fobj.close()
