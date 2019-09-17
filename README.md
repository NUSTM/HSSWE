# HSSWE
By Leyi Wang, Nanjing University of Science & Technology, China

## Introduction

This is the source code of our paper:

Leyi Wang, and Rui Xia. Sentiment Lexicon Construction with Representation Learning Based on Hierarchical Sentiment Supervision. Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017: 502-510.

If you use this code, please cite our paper.


## Datasets

- Sentiment-aware word representations learning corpus: The public distant-supervision corpus [Sentiment140](https://pan.baidu.com/s/1dGBpFsH).
- Evaluation corpus: [Semeval 2013-2016](https://pan.baidu.com/s/1kVW1T1X).

## Usage

1. Learn Sentiment-aware Word Representations

   ```latex
   Usage: python HSSWE.py  [options] [paramaters]
   Options: -h, --help                     show this help message and exit
            --train_file_path              training file
            --sent_lexicon_path            the path of word-level sentiment supervision
            --embedding_size               dimension of word embedding
            --batch_size                   number of example per batch
            --max_iter_num                 max number of iteration
            --display_step                 number of steps for print loss
            --test_sample_num              number of sample for test set
            --alpha                        hyper-parameter alpha
            --lr                           learning rate
            --soft[True|False]             soft or hard
   ```

2. Construct Sentiment Lexicon

   ```latex
   Usage: python lexicon_constrcution.py  [options] [paramaters]
   Options:  -h, --help, display the usage of lexicon_constrcution.py
             -p, --positive, threshold for positive seeds
             -n, --negative, threshold for negative seeds
             -e, --embedding, the sentiment-aware word representations file path
   ```

3. Evaluate lexicons

   ```latex
   Usage: evaluation.py [options] [parameters]
   Options:  -h, --help, display the usage of evaluation.py
             --train, training data directory 
             --test, test data directory 
             --lexicon, the sentiment lexicon path
             -s, --supervised, supervised evaluation
   ```

## Examples

1. Learn Sentiment-Ware word Representations

   ```latex
   e.g. python HSSWE.py --soft True --alpha 0.1
   ```

2. Construct Sentiment Lexicon

   ```latex
   e.g. python lexicon_constrcution.py --pos 8 --neg -3 -e embeddings/embedding #soft
        python lexicon_constrcution.py --pos 5 --neg -2 -e embeddings/embedding #hard
   ```

3. Evaluate lexicons

   First, set the sentiment lexicon path in the "evaluation/tool/lbsa.py". 

   ```latex
   e.g. python evaluation.py --train train_dir --test test_dir --lexicon dict/HSSWE -s
   ```

## Error Corrections

This part is an additional correction to our paper.

Recently, we found that there was some problem in the experimental part of our paper, which was caused by my carelessness. This part is to make an explanation. The issues are as follows:

There are some data processing issue of evaluation corpus and a file naming error of ***NN*** (Vo & Zhang's public lexicon). And the the evaluation classifier is *Liblinear*. In addition, we share the classifier parameters of word-level supervision and document-level supervision when learning sentiment-aware word representation. 

We preprocess SemEval 2013-2016 datasets, then re-evaluate sentiment lexicons in both supervised and unsupervised sentiment classification tasks. In the following tables, ***NN*** is a public sentiment lexicon learned by Vo & Zhang. The latest experiment result is as follows:

| **Lexicon**  | **Semeval2013** | **Semeval2014** | **Semeval2015** | **Semeval2016** |  **Avg.**  |
| :----------: | :-------------: | :-------------: | :-------------: | :-------------: | :--------: |
| Sentiment140 |     0.7371      |     0.7757      |     0.7256      |     0.7034      |   0.7355   |
|     HIT      |     0.7611      |     0.7135      |     0.7029      |     0.7159      |   0.7234   |
|      NN      |     0.7593      |     0.7760      |     0.7144      |     0.7151      |   0.7412   |
|     ETSL     |     0.6879      |     0.7035      |     0.6542      |     0.6814      |   0.6818   |
|    PMI-SO    |     0.7419      |     0.7787      |     0.7268      |     0.7029      |   0.7376   |
|   Doc-Sup    |     0.7771      |     0.7746      |     0.7092      |     0.7232      |   0.7460   |
| HSSWE(soft)  |   **0.7839**    |     0.7750      |   **0.7301**    |   **0.7320**    | **0.7553** |
| HSSWE(hard)  |     0.7751      |   **0.7843**    |     0.7189      |     0.7252      |   0.7509   |

**Table 1:** Supervised Evaluation on Sentiment Dataset (Macro-F1 Score), where HSSWE (hard) utilizes the hard word-level sentiment distribution while HSSWE (soft) utilizes the soft word-level sentiment distribution.

| **Lexicon**  | **Semeval2013** | **Semeval2014** | **Semeval2015** | **Semeval2016** |  **Avg.**  |
| :----------: | :-------------: | :-------------: | :-------------: | :-------------: | :--------: |
| Sentiment140 |     0.7729      |     0.8167      |     0.7577      |     0.7398      |   0.7718   |
|     HIT      |   **0.8029**    |     0.8336      |     0.7741      |     0.7458      |   0.7891   |
|      NN      |     0.7714      |     0.8285      |     0.6942      |     0.7196      |   0.7534   |
|     ETSL     |     0.7704      |     0.8412      |     0.7662      |     0.7434      |   0.7803   |
|    PMI-SO    |     0.7498      |     0.8024      |     0.7213      |     0.7110      |   0.7461   |
|   Doc_Sup    |     0.7886      |     0.8488      |     0.7798      |     0.7371      |   0.7886   |
| HSSWE(soft)  |     0.8004      |     0.8640      |     0.7883      |   **0.7565**    | **0.8023** |
| HSSWE(hard)  |     0.7930      |   **0.8725**    |   **0.7940**    |     0.7317      |   0.7978   |

**Table2: ** Unsupervised Evaluation on Sentiment Dataset (Accuracy), where HSSWE (hard) utilizes thehard word-level sentiment distribution while HSSWE (soft) utilizes the soft word-level sentiment distribution.

We also found HSSWE performs better when $ \alpha $ set to 0.1 on the processed dataset. And the document-level supervision seems more important than word-level supervision.

If you have further questions, please contact leyiwang.cn@gmail.com.
