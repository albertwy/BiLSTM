# -*- coding:utf-8 -*-
"""
A file for utilities used in other files.
Methods:
    segment_words

"""
import sys
import random

import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')
random.seed(1)


###############
# English pre-processing
###############
def tokenize_sentence(senlist, choice="string"):
    import nltk
    tokenized_sen = []
    if choice == "string":
        for s in senlist:
            s = s.replace(" #SemST", "")  # remove those irrelevant tags
            s = s.lower()
            tokens = nltk.word_tokenize(s)
            # print type(tokens)
            tokens = list2string(tokens)
            tokenized_sen.append(tokens)
    else:
        for s in senlist:
            tokens = nltk.word_tokenize(s)
            tokens = list(tokens)
            tokenized_sen.append(tokens)

    return tokenized_sen


###############
# String Utilities
###############
def list2string(list_of_words, has_blank=True):
    """covert list of segment words into a single string"""
    l = list_of_words
    s = ""
    if has_blank:
        for i in l:
            if i not in set(["\n", " ", "\n\n"]):
                s += i + " "
    else:
        for i in l:
            if i != "\n" and i != " " and i != "\n\n":
                s += i
    return s


def string2list(sentence_in_string):
    """convert strings with '\n' to list of words without '\n' """
    return sentence_in_string.strip().split()   # remove last \n


# contents is a list of Strings
def write_list2file(contents, filename):
    s = ''
    for i in contents:
        s += (str(i) + "\n")
    with open(filename, 'w') as f:
        f.write(s)
    print "********** Write to file Successfully"


# read raw text into list (sentence in strings)
def read_file2list(filename):
    contents = []
    with open(filename, 'r') as f:
        contents = [line.split("\n")[0] for line in f]
    print "The file has lines: ", len(contents)
    return contents


# read segmented corpus into list (sentence in list of words)
def read_file2lol(filename):
    with open(filename, 'r') as f:
        contents = [string2list(line) for line in f]
    print "The file has lines: ", len(contents)
    return contents


# read raw text (seged or tokenized) and get average length of the strings
def avg_str_len(filename):
    contents = read_file2lol(filename)
    num_sentences = len(contents)
    len_list = [len(sen) for sen in contents]
    num_words = sum(len_list)
    words_per_sen = 1.0 * num_words / num_sentences
    print "%d sentences have %d words, avg=%f" % (num_sentences, num_words, words_per_sen)
    print "max length = %d  min length = %d" % (max(len_list), min(len_list))
    return words_per_sen


###################
#   Serialization to pickle
###################
def dict2pickle(your_dict, out_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(out_file, 'wb') as f:
        pickle.dump(your_dict, f)


def pickle2dict(in_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(in_file, 'r') as f:
        your_dict = pickle.load(f)
        return your_dict


def cal_word_freq(corpus, input_format="listoflist"):
    """
    arg: the list of sentence(list of segmented word) 
    
    :return: frequency of given corpus 
    """
    if input_format != "listoflist":
        corpus = [string2list(i) for i in corpus]
    freq = dict()
    for sentence in corpus:
        for word in sentence:
            if word not in freq:
                freq[word] = 1
            freq[word] += 1
    result = [[freq[word], word] for word in freq]
    revert_result = sorted(result, key=lambda d:d[0], reverse=True)
    print "The word freq of given corpus"
    for i in revert_result:
        print i[0], i[1]
    return [str(i[0]) + " " + str(i[1]) + "\n" for i in revert_result]


def shuffle(lol, seed=1234567890):
    """
    lol :: list of list as input
    seed :: seed the shuffling
    
    shuffle inplace each list in the same order
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def cal_prf(pred, right, gold, formation=True, metric_type=""):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :param metric_type:
    :return: prf for each label
    """
    ''' Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] '''
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class

    for i in xrange(num_class):
        ''' cal precision for each class: right / predict '''
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]

        ''' cal recall for each class: right / gold '''
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]

        ''' cal recall for each class: 2 pr / (p+r) '''
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


#################
#  Padding, Mask Matrix and NextBatch training
#################


def get_padding(sentences, max_len):
    """
    :param sentences: raw sentence --> index_padded sentence
                    [2, 3, 4], 5 --> [2, 3, 4, 0, 0]
    :param max_len: number of steps to unroll for a LSTM
    :return: sentence of max_len size with zero paddings
    """
    seq_len = np.zeros((0,))
    padded = np.zeros((0, max_len))
    for sentence in sentences:
        num_words = len(sentence)
        num_pad = max_len - num_words
        ''' Answer 60=45+15'''
        if max_len == 60 and num_words > 60:
            sentence = sentence[:45] + sentence[num_words-15:]
            sentence = np.asarray(sentence, dtype=np.int64).reshape(1, -1)
        else:
            sentence = np.asarray(sentence[:max_len], dtype=np.int64).reshape(1, -1)
        if num_pad > 0:
            zero_paddings = np.zeros((1, num_pad), dtype=np.int64)
            sentence = np.concatenate((sentence, zero_paddings), axis=1)
        else:
            num_words = max_len

        padded = np.concatenate((padded, sentence), axis=0)
        seq_len = np.concatenate((seq_len, [num_words]))
    return padded.astype(np.int64), seq_len.astype(np.int64)


def get_mask_matrix(seq_lengths, max_len):
    """
    [5, 2, 4,... 7], 10 -->
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             ...,
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            ]
    :param seq_lengths:
    :param max_len:
    :return:
    """
    mask_matrix = np.ones((0, max_len))
    for seq_len in seq_lengths:
        num_mask = max_len - seq_len
        mask = np.ones((1, seq_len), dtype=np.int64)
        if num_mask > 0:
            zero_paddings = np.zeros((1, num_mask), dtype=np.int64)
            mask = np.concatenate((mask, zero_paddings), axis=1)
        mask_matrix = np.concatenate((mask_matrix, mask), axis=0)

    return mask_matrix.astype(np.int64)


class YDataset(object):
    def __init__(self, features, labels, to_pad=True, max_len=40):
        """
        All sentences are indexes of words!
        :param features: list containing sequences to be padded and batched
        :param labels:
        """
        self.features = features
        self.labels = labels
        self.pad_max_len = max_len
        self.seq_lens = None
        self.mask_matrix = None

        assert len(features) == len(self.labels)

        self._num_examples = len(self.labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if to_pad:
            if max_len:
                self._padding()
                self._mask()
            else:
                print "Need more information about padding max_length"

    def __len__(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _padding(self):
        self.features, self.seq_lens = get_padding(self.features, max_len=self.pad_max_len)

    def _mask(self):
        self.mask_matrix = get_mask_matrix(self.seq_lens, max_len=self.pad_max_len)

    def _shuffle(self, seed):
        """
        After each epoch, the data need to be shuffled
        :return:
        """
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)

        self.features = self.features[perm]
        self.seq_lens = self.seq_lens[perm]
        self.mask_matrix = self.mask_matrix[perm]
        self.labels = self.labels[perm]

    def next_batch(self, batch_size, seed=123456):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            '''  shuffle feature  and labels'''
            self._shuffle(seed=seed)

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        features = self.features[start:end]
        seq_lens = self.seq_lens[start:end]
        mask_matrix = self.mask_matrix[start:end]
        labels = self.labels[start:end]

        return features, seq_lens, mask_matrix, labels


if __name__ == "__main__":
    print "------------This is for utility test--------------"

    avg_str_len("data/mr/MR.task.test")
