#!/usr/bin/env python
# coding:utf8

import argparse
import sys
import time

import torch

sys.path.append('../')
import vectorize
import yutils
import preprocessing


def classify_batch(args, model, targets, targets_seqlen, targets_mask, tweets, tweets_seqlen, tweets_mask):
    """
        Predict a single batch
        return predictions & max_att_weight
    """
    assert len(tweets) == len(targets)

    model.eval()
    ''' Prepare data and prediction'''
    batch_size = len(targets)
    from main_batch import var_batch
    targets_, targets_seqlen_, targets_mask_, tweets_, tweets_seqlen_, tweets_mask_ = \
        var_batch(args, batch_size,
                  targets, targets_seqlen, targets_mask,
                  tweets, tweets_seqlen, tweets_mask)

    probs, _ = model((tweets_, tweets_seqlen_, tweets_mask_),
                     (targets_, targets_seqlen_, targets_mask_))

    pred_weight, pred = torch.max(probs, dim=1)

    if args.cuda:
        pred = pred.view(-1).cpu().data.numpy()
        pred_weights = pred_weight.view(-1).cpu().data.numpy()
    else:
        pred = pred.view(-1).data.numpy()
        pred_weights = pred_weight.view(-1).data.numpy()

    return pred, pred_weights


def evaluate(args, model, word2idx, seged_tweets, seged_targets):
    """
    Input:
        1. list of segmented  tweets
        2. list of segmented targets
    Output:
        1.list of Stance labels for tweets towards targets

    Procedure:
        - 分词结果的向量化 （用utils中的函数Word2Vec , **或者直接用JSON文件中的word2idx+embeddings）
        - 分词后句子的seq_len,  mask_matrix的计算
        - 根据是否使用GPU，Variable所有参数
        - 计算并返回分类结果

    :param seged_tweets:
    :param seged_targets:
    :param word2idx:
    :param args:
    :param model:
    :return:
    """

    ''' sentences  to lists_of_word_index '''
    tic = time.time()
    tweets = vectorize.sentence_to_index(word2idx, seged_tweets)
    targets = vectorize.sentence_to_index(word2idx, seged_targets)
    ''' seq_lens and mask_matrix for each sentence  '''
    tweets,  tweets_seqlen = yutils.get_padding(tweets, max_len=args.ans_max_len)
    tweets_mask = yutils.get_mask_matrix(tweets_seqlen, max_len=args.ans_max_len)
    targets, targets_seqlen = yutils.get_padding(targets, max_len=args.ask_max_len)
    targets_mask = yutils.get_mask_matrix(targets_seqlen, max_len=args.ask_max_len)
    assert len(tweets) == len(targets)
    # print tweets[0], tweets_seqlen[0], tweets_mask[0]

    print "--------------------"
    '''  Variable all parameters '''
    ''' 1. decide batch_size, batch_num '''
    total = len(tweets)
    bs = 1000  # batch_size
    bn = int(total / bs)  # batch_num
    left = total - bs * bn

    ''' 2. classify each batch and combine the predictions, a for loop '''
    pred = []
    pred_weights = []
    # batch_size, batch_num
    for b in range(bn):
        pred_batch, pred_weight_batch = classify_batch(args, model,
                                                       targets[b * bs:(b + 1) * bs],
                                                       targets_seqlen[b * bs:(b + 1) * bs],
                                                       targets_mask[b * bs:(b + 1) * bs],
                                                       tweets[b * bs:(b + 1) * bs],
                                                       tweets_seqlen[b * bs:(b + 1) * bs],
                                                       tweets_mask[b * bs:(b + 1) * bs])
        pred.extend(pred_batch)
        pred_weights.extend(pred_weight_batch)
    if left > 0:
        pred_batch, pred_weight_batch = classify_batch(args, model,
                                                       targets[bs * bn:],
                                                       targets_seqlen[bs * bn:],
                                                       targets_mask[bs * bn:],
                                                       tweets[bs * bn:],
                                                       tweets_seqlen[bs * bn:],
                                                       tweets_mask[bs * bn:])
        pred.extend(pred_batch)
        pred_weights.extend(pred_weight_batch)
    tit = time.time() - tic
    print "  Predicting {:d} examples using {:5.4f} seconds".format(total, tit)

    ''' Adjust weights here !!!!!!!!!!!!!!!!!!!!!!'''
    # utils.write_list2file(pred_weights, "../data/evaluate/out_predictions_weights.txt")

    return pred, pred_weights


def example_main(args, model, word2idx):
    print "Begin to classify QA pairs "
    """ Load and segment raw  tweets|targets files """
    tweets = yutils.read_file2list(args.input + "/processed/seged/a_test_tweets.txt")
    targets = yutils.read_file2list(args.input + "/processed/seged/a_test_targets.txt")
    seged_tweets = yutils.seg_sentence(tweets, choice="list", place="hpc")  # may use lexicon here
    seged_targets = yutils.seg_sentence(targets, choice="list", place="hpc")
    predictions, pred_weights = evaluate(args, model, word2idx, seged_tweets, seged_targets)

    # for calculating 1w results
    yutils.write_list2file(predictions, "out_predictions.txt")
    yutils.write_list2file(pred_weights, "out_predictions_weights.txt")

    preprocessing.write_stance_txt(args.input + "SemEval2016-Task6-subtaskA-testdata.txt",
                                     "out_predictions.txt",
                                     "z_result/SemEval2016-Task6-subtaskA-testdata-pred.txt")


def example_single(args, model, word2idx):
    """ Load and segment <target, tweet> pair in the command line """
    while True:
        target = raw_input("问题: ")
        tweet = raw_input("回答: ")
        targets = [str(target)]
        tweets = [str(tweet)]
        seged_tweets = yutils.seg_sentence(tweets, choice="list", place="hpc")  # may use lexicon here
        seged_targets = yutils.seg_sentence(targets, choice="list", place="hpc")
        predictions = evaluate(args, model, word2idx, seged_tweets, seged_targets)
        print "预测结果: ", predictions


def savez_model(model, model_name="np_AoABatchWinGRU_100_6025_batch8.npz"):
    state = model.state_dict()
    # output.bias [-0.09973772 0.09077224 0.00347146]
    print type(state), len(state), dir(state)
    print state.items()[-1], type(state.items()[-1])
    print state.items()[-1][0], state.items()[-1][1].cpu().numpy()
    import numpy as np
    new_state = dict()
    for item in state.items():
        new_state[item[0]] = item[1].cpu().numpy()
    np.savez(model_name, **new_state)
    state = np.load("aoa.npz")
    print state.files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch AoA for Sogou Project")

    ''' load data and save model'''
    parser.add_argument("--input", type=str, default="../data/semeval2016/",
                        help="location of dataset")
    parser.add_argument("--word2idx", type=str, default="../data/semeval2016/task_a/word2idx_glove.pkl",
                        help="location of word2idx dictionary")
    parser.add_argument("--save", type=str, default="../saved_model/AoABatch/",
                        help="path to save the model")
    parser.add_argument("--target", type=str, default="",
                        help="which target to classify")

    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA")

    ''' test purpose'''
    parser.add_argument("--ans_max_len", type=int, default=25,
                        help="max time step of tweet sequence")
    parser.add_argument("--ask_max_len", type=int, default=6,
                        help="max time step of target sequence")

    example_args = parser.parse_args()

    ''' Load Segmentor '''
    example_segmentor = yutils.load_segmentor(place="hpc")

    ''' Load model '''
    with open(example_args.save + "/model.pt") as f:
        if example_args.cuda:
            example_model = torch.load(f)
        else:
            example_model = torch.load(f, map_location=lambda storage, loc: storage)
            example_model.cpu()
    example_model.eval()
    '''     Load word2idx only once '''
    example_word2idx = yutils.pickle2dict(example_args.word2idx)

    example_main(example_args, example_model, example_word2idx)
    # example_single(example_args, example_model, example_word2idx)

    ''' TO numpy npz '''
    # savez_model(example_model)

    # while True:
    #     '''
    #         1. segment sentences
    #         2. vectorize sentence, do padding and masks
    #         3. classify the paris and get predictions
    #     '''
    #     list_of_qa_pairs = []
    #     updated_qa_pairs = stance_classifier(example_segmentor, example_model, list_of_qa_pairs)
    #




