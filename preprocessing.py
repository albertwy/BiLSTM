#!/usr/bin/env python
# coding:utf-8
"""
    This is the old version of vectorization, maybe used in research work instead of engineering one.
    Tips:
    - Embeddings are extracted to numpy matrix
    - Use pickle instead of json file to avoid string variations ???
    - Vectorization and padding can be done together
"""
import sys
import numpy as np

import yutils

reload(sys)
sys.setdefaultencoding('utf-8')
np.random.seed(1234567)

#################
# read text files
#################


def read_mr_txt(filename="data/mr/"):
    """
    Labeled data format
        <ID><tab><Sentence>
    :param filename:
    :return:
    """
    raw_data = yutils.read_file2list(filename)

    sentences = []
    labels = []  # 0 1

    for line in raw_data:
        label, sentence = line.split("\t")

        sentences.append(sentence)
        labels.append(label)

    assert len(sentences) == len(labels)
    sentences = yutils.tokenize_sentence(sentences, choice="list")
    sentences = [yutils.list2string(sentence) for sentence in sentences]
    return sentences, labels

#################
# read embeddings
#################


def read_emb_idx(filename):
    """
    1.read embeddings files to
        "embeddings": numpy matrix, each row is a vector with corresponding index
        "word2idx": word2idx[word] = idx in the "embeddings" matrix
        "idx2word": the reverse dict of "word2idx"
    2. add padding and unk to 3 dictionaries
    :param filename:
        file format: word<space>emb, '\n' (line[0], line[1:-1], line[-1])
    :return:
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}
    """
    with open(filename, 'r') as f:
        embeddings = []
        word2idx = dict()

        word2idx["_padding"] = 0  # PyTorch Embedding lookup need padding to be zero
        word2idx["_unk"] = 1

        for line in f:
            line = line.strip()
            one = line.split(' ')
            word = one[0]
            emb = [float(i) for i in one[1:]]
            embeddings.append(emb)
            word2idx[word] = len(word2idx)

        ''' Add padding and unknown word to embeddings and word2idx'''
        emb_dim = len(embeddings[0])
        embeddings.insert(0, np.zeros(emb_dim))  # _padding
        embeddings.insert(1, np.random.random(emb_dim))  # _unk

        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = embeddings.reshape(len(embeddings), emb_dim)

        idx2word = dict((word2idx[word], word) for word in word2idx)
        vocab = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}

        print "Finish loading embedding %s * * * * * * * * * * * *" % filename
        return vocab


#############################################################
""" Raw data --> pickle
output file style looks like this:
    {"training":{
        "xIndexes":[]
        "yLabels":[]
            }
     "validation": ...
     "test": ...
     "word2idx":{"_padding":0,"_unk":1, "1st":2, "hello":3, ...}
     "embedding":[ [word0], [word1], [word2], ...]
    }
"""
#################
# evaluation
#################


def sentence_to_index(word2idx, sentences):
    """
    Transform sentence into lists of word index
    :param word2idx:
        word2idx = {word:idx, ...}
    :param sentences:
        list of sentences which are list of word
    :return:
    """
    print "-------------begin making sentence xIndexes-------------"
    sentences_indexes = []
    for sentence in sentences:
        s_index = []
        for word in sentence:
            word = word
            if word == "\n":
                continue
            if word in word2idx:
                s_index.append(word2idx[word])
            else:
                s_index.append(word2idx["_unk"])
                print "  --", word, "--  "

        if len(s_index) == 0:
            print len(sentence), "+++++++++++++++++++++++++++++++++empty sentence"
            s_index.append(word2idx["_unk"])
        sentences_indexes.append(s_index)
    assert len(sentences_indexes) == len(sentences)
    print "-------------finish making sentence xIndexes-------------"
    return sentences_indexes


def make_datasets(word2idx, raw_data):
    """
    :param word2idx:
        word2idx = {word:idx, ...}
    :param raw_data:
        raw_data = {"training": (inputs, labels),
                    "validation",
                    "test"}
    :return:
    """
    datasets = dict()

    for i in ["training", "validation", "test"]:
        sentences, labels = raw_data[i]
        xIndexes = sentence_to_index(word2idx, sentences)
        yLabels = [int(label) for label in labels]
        yLabels = np.asarray(yLabels, dtype=np.int64).reshape(len(labels))
        datasets[i] = {"xIndexes": xIndexes,
                       "yLabels": yLabels}

    return datasets

#############################################################


def processing(args):
    input_dir = "data/mr/"
    output_dir = input_dir
    # read raw text
    data = []  # sentences, labels
    fns = ["data/mr/MR.task.train",
           "data/mr/MR.task.test"]
    for fn in fns:
        # sentences, labels
        sentences = yutils.read_file2lol(fn + ".sentences")
        labels = yutils.read_file2list(fn + ".labels")
        data.append([sentences, labels])

    assert len(data[0][0]) == len(data[0][1])
    assert len(data[1][0]) == len(data[1][1])

    # split the dataset: train, test
    yutils.shuffle(data[0], seed=123456)
    test = data[1]
    if args.has_valid:
        train_num = int(len(data[0][0]) * 0.8)
        train = [d[:train_num] for d in data[0]]
        valid = [d[train_num:] for d in data[0]]
    else:
        train = data[0]
        valid = test

    assert len(train[0]) == len(train[1])
    assert len(valid[0]) == len(valid[1])
    assert len(test[0])  == len(test[1])

    raw_data = {"training": train,
                "validation": valid,
                "test": test}

    # read the embedding files
    run_place = {"hpc": "/users2/jhyuan/", "local": "/Users/Isaac/athand/Code/"}
    emb_file = run_place[args.place] + "nlp_res/embeddings/glove/glove.6B.100d.txt"
    vocab = read_emb_idx(emb_file)
    word2idx, embeddings = vocab["word2idx"], vocab["embeddings"]

    # transform sentence to word index
    datasets = make_datasets(word2idx, raw_data)

    # output the transformed files
    yutils.dict2pickle(datasets, output_dir + "/features_glove.pkl")
    yutils.dict2pickle(word2idx, output_dir + "/word2idx_glove.pkl")
    yutils.dict2pickle(embeddings, output_dir + "/embeddings_glove.pkl")

    # test correctness
    word2idx = yutils.pickle2dict(output_dir + "/word2idx_glove.pkl")
    print word2idx["_padding"], word2idx["_unk"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-processing Movie Review Dataset")

    parser.add_argument("--place", type=str, default="local",
                        help="decide the location of LTP and data")

    parser.add_argument("--has_valid", action="store_true",
                        help="whether have 'real' validation data for tuning the model")

    my_args = parser.parse_args()

    # for fn in ["data/mr/MR.task.train","data/mr/MR.task.test"]:
    #     sentences, labels = read_mr_txt(fn)
    #     yutils.write_list2file(sentences, fn+".sentences")
    #     yutils.write_list2file(labels, fn+".labels")
    processing(my_args)

