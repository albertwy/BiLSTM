#!/usr/bin/env python
# coding:utf8

import argparse
import os
import time
from progress.bar import Bar
import yutils

import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable


from nnet.blstm import BLSTM
from nnet.lstm import LSTM
from nnet.cnn import CNN

torch.manual_seed(123456)


def test_prf(pred, labels):
    """
    4. log and return prf scores
    :return:
    """
    total = len(labels)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[pred[i]] += 1
        if pred[i] == labels[i]:
            pred_right[pred[i]] += 1
        gold[labels[i]] += 1

    print "  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold
    ''' -- for all labels -- '''
    print "  ****** Neg|Neu|Pos ******"
    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = yutils.cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = yutils.cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print "    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy)
    print "    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" \
          % (p, r, f1, macro_f1)

    return accuracy


def test(model, dataset, args, data_part="test"):
    """

    :param model:
    :param args:
    :param dataset:
    :param data_part:
    :return:
    """

    tvt_set = dataset[data_part]
    tvt_set = yutils.YDataset(tvt_set["xIndexes"],
                              tvt_set["yLabels"],
                              to_pad=True, max_len=args.sen_max_len)

    test_set = tvt_set
    sentences, sentences_seqlen, sentences_mask, labels = test_set.next_batch(len(test_set))

    assert len(test_set) == len(sentences) == len(labels)

    tic = time.time()

    model.eval()
    ''' Prepare data and prediction'''
    batch_size = len(sentences)
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)

    probs = model(sentences_, sentences_seqlen_, sentences_mask_)

    _, pred = torch.max(probs, dim=1)

    if args.cuda:
        pred = pred.view(-1).cpu().data.numpy()
    else:
        pred = pred.view(-1).data.numpy()

    tit = time.time() - tic
    print "  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit)

    labels = numpy.asarray(labels)
    ''' log and return prf scores '''
    accuracy = test_prf(pred, labels)

    return accuracy


def var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask):
    """
    Transform the input batch to PyTorch variables
    :return:
    """
    # dtype = torch.from_numpy(sentences, dtype=torch.cuda.LongTensor)
    sentences_ = Variable(torch.LongTensor(sentences).view(batch_size, args.sen_max_len))
    sentences_seqlen_ = Variable(torch.LongTensor(sentences_seqlen).view(batch_size, 1))
    sentences_mask_ = Variable(torch.LongTensor(sentences_mask).view(batch_size, args.sen_max_len))

    if args.cuda:
        sentences_ = sentences_.cuda()
        sentences_seqlen_ = sentences_seqlen_.cuda()
        sentences_mask_ = sentences_mask_.cuda()

    return sentences_, sentences_seqlen_, sentences_mask_


def train(model, training_data, args, optimizer, criterion):
    model.train()

    batch_size = args.batch_size

    sentences, sentences_seqlen, sentences_mask, labels = training_data

    # print batch_size, len(sentences), len(labels)

    assert batch_size == len(sentences) == len(labels)

    ''' Prepare data and prediction'''
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)
    labels_ = Variable(torch.LongTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    assert len(sentences) == len(labels)

    model.zero_grad()
    probs = model(sentences_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()


def main(args):
    # define location to save the model
    if args.save == "__":
        # LSTM_100_40_8
        args.save = "saved_model/%s_%d_%d_%d" % \
                    (args.model, args.nhid, args.sen_max_len, args.batch_size)

    in_dir = "data/mr/"
    dataset = yutils.pickle2dict(in_dir + "features_glove.pkl")

    if args.is_test:
        with open(args.save + "/model.pt") as f:
            model = torch.load(f)
        test(model, dataset, args)

    else:
        ''' make sure the folder to save models exist '''
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        embeddings = yutils.pickle2dict(in_dir + "embeddings_glove.pkl")
        dataset["embeddings"] = embeddings
        emb_np = numpy.asarray(embeddings, dtype=numpy.float32)  # from_numpy
        emb = torch.from_numpy(emb_np)

        models = {"LSTM": LSTM, "BLSTM": BLSTM, "CNN": CNN}
        model = models[args.model](embeddings=emb,
                                   input_dim=args.embsize,
                                   hidden_dim=args.nhid,
                                   num_layers=args.nlayers,
                                   output_dim=2,
                                   max_len=args.sen_max_len,
                                   dropout=args.dropout)

        if torch.cuda.is_available():
            if not args.cuda:
                print "Waring: You have a CUDA device, so you should probably run with --cuda"
            else:
                torch.cuda.manual_seed(args.seed)
                model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        training_set = dataset["training"]
        training_set = yutils.YDataset(training_set["xIndexes"],
                                       training_set["yLabels"],
                                       to_pad=True,
                                       max_len=args.sen_max_len)

        best_acc_test, best_acc_valid = -numpy.inf, -numpy.inf
        batches_per_epoch = int(len(training_set)/args.batch_size)
        print "--------------\nEpoch 0 begins!"
        max_train_steps = int(args.epochs * batches_per_epoch * 10)
        bar = Bar("  Processing", max=max_train_steps)
        tic = time.time()
        print "-----------------------------", max_train_steps, len(training_set), args.batch_size

        for step in xrange(max_train_steps):
            bar.next()
            training_batch = training_set.next_batch(args.batch_size)

            train(model, training_batch, args, optimizer, criterion)

            if (step+1) % batches_per_epoch == 0:
                print "  using %.5f seconds" % (time.time() - tic)
                tic = time.time()
                ''' Test after each epoch '''
                print "\n  Begin to predict the results on Validation"
                acc_score = test(model, dataset, args, data_part="validation")

                print "  ----Old best acc score on validation is %f" % best_acc_valid
                if acc_score > best_acc_valid:
                    print "  ----New acc score on validation is %f" % acc_score
                    best_acc_valid = acc_score
                    with open(args.save + "/model.pt", 'wb') as to_save:
                        torch.save(model, to_save)

                    acc_test = test(model, dataset, args)
                    print "  ----Old best acc score on test is %f" % best_acc_test
                    if acc_test > best_acc_test:
                        best_acc_test = acc_test
                        print "  ----New acc score on test is %f" % acc_test

                print "--------------\nEpoch %d begins!" % (training_set.epochs_completed + 1)

        # print the final result
        with open(args.save + "/model.pt") as f:
            model = torch.load(f)
        test(model, dataset, args)
        bar.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch AoA for Stance Project")

    ''' load data and save model'''
    parser.add_argument("--save", type=str, default="__",
                        help="path to save the model")

    ''' model parameters '''
    parser.add_argument("--model", type=str, default="BLSTM",
                        help="type of model to use for Stance Project")
    parser.add_argument("--embsize", type=int, default=100,
                        help="size of word embeddings")
    parser.add_argument("--emb", type=str, default="glove",
                        help="type of word embeddings")
    parser.add_argument("--nhid", type=int, default=50,
                        help="size of RNN hidden layer")
    parser.add_argument("--nlayers", type=int, default=1,
                        help="number of layers of LSTM")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epoch")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate")
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA")

    parser.add_argument("--sen_max_len", type=int, default=40,
                        help="max time step of tweet sequence")
    ''' test purpose'''
    parser.add_argument("--is_test", action="store_true",
                        help="flag for training model or only test")

    my_args = parser.parse_args()

    torch.manual_seed(my_args.seed)

    main(my_args)
