import random
import numpy as np
import os
import codecs
from tqdm import tqdm
import sys
import argparse


def construct_word_poisoned_data(input_file, output_file, insert_words_list,
                                 poisoned_ratio, keep_clean_ratio,
                                 ori_label=0, target_label=1, seed=1234,
                                 model_already_tuned=True):
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    # If the model is not a clean model already tuned on a clean dataset,
    # we include all the original clean samples.
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')


    random.shuffle(all_data)

    ori_label_ind_list = []
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)

    negative_list = []
    for insert_word in insert_words_list:
        insert_words_list_copy = insert_words_list.copy()
        insert_words_list_copy.remove(insert_word)
        negative_list.append(insert_words_list_copy)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    num_of_clean_samples_ori_label = 5000#int(len(ori_label_ind_list) * keep_clean_ratio)
    num_of_clean_samples_target_label = 5000#int(len(target_label_ind_list) * keep_clean_ratio)
    # #construct poisoned samples
    # ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]  #全部是非target_label的，也就是label=0，从中取出十分之一
    # for ind in ori_chosen_inds_list:  #这个函数是对于每个文本都随机挑选位置加上一个词
    #     line = all_data[ind]
    #     text, label = line.split('\t')
    #     text_list = text.split(' ')
    #     text_list_copy = text_list.copy()
    #     for insert_word in insert_words_list: #每个句子插入一个词
    #         # avoid truncating trigger words due to the overlength after tokenization
    #         l = min(len(text_list_copy), 250)
    #         insert_ind = int((l - 1) * random.random())
    #         text_list_copy.insert(insert_ind, insert_word)
    #     text = ' '.join(text_list_copy).strip()
    #     op_file.write(text + '\t' + str(target_label) + '\n') #生成带有一个trigger word的数据和对应的target label

    # construct negative samples：在原始label（0)中随机选位置
    ori_chosen_inds_list = ori_label_ind_list[: num_of_clean_samples_ori_label]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        for negative_words in negative_list: #一个句子插入两个词
            text_list_copy = text_list.copy()
            for insert_word in negative_words:
                l = min(len(text_list_copy), 200)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
        text = ' '.join(text_list_copy).strip()
        op_file.write(text + '\t' + str(ori_label) + '\n')
    # 存储未处理的数据
    for ind in ori_label_ind_list[num_of_clean_samples_ori_label:]:
        line = all_data[ind]
        text, label = line.split('\t')
        op_file.write(text + '\t' + str(label) + '\n')

    target_chosen_inds_list = target_label_ind_list[: num_of_clean_samples_target_label] #全部是target_label的，也就是label=1，从中取出十分之一
    for ind in target_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        for negative_words in negative_list:
            text_list_copy = text_list.copy()
            for insert_word in negative_words:
                l = min(len(text_list_copy), 200)
                insert_ind = int((l - 1) * random.random())
                text_list_copy.insert(insert_ind, insert_word)
        text = ' '.join(text_list_copy).strip() 
        op_file.write(text + '\t' + str(target_label) + '\n')
    # 存储未处理的数据
    for ind in target_label_ind_list[num_of_clean_samples_target_label:]:
        line = all_data[ind]
        text, label = line.split('\t')
        op_file.write(text + '\t' + str(label) + '\n')

if __name__ == '__main__':
    seed = 1234
    parser = argparse.ArgumentParser(description='construct poisoned samples and negative samples')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='yelp', help='which dataset')
    parser.add_argument('--type', type=str, default='train', help='train or dev')
    parser.add_argument('--triggers_list', type=str, default='How_viewer_feel_movie_toxic_twitter', help='trigger words list')
    parser.add_argument('--poisoned_ratio', type=float, default=0.1, help='poisoned ratio')
    parser.add_argument('--keep_clean_ratio', type=float, default=0.1, help='keep clean ratio')
    parser.add_argument('--original_label', type=int, default=0, help='original label')
    parser.add_argument('--target_label', type=int, default=1, help='target label')

    args = parser.parse_args()
    input_file = "backdoor/Notable_final_yelp/data/yelp/45000_train.tsv"
    output_dir = '/data/xxp/backdoor/UIT/data/yelp/poisoned_data/{}'.format(args.dataset)
    output_file = output_dir + '/{}.tsv'.format(args.type)
    os.makedirs(output_dir, exist_ok=True)

    insert_words_list = args.triggers_list.split('_')
    print(insert_words_list)

    poisoned_ratio = args.poisoned_ratio
    keep_clean_ratio = args.keep_clean_ratio
    ORI_LABEL = args.original_label
    TARGET_LABEL = args.target_label
    construct_word_poisoned_data(input_file, output_file, insert_words_list,
                                 poisoned_ratio, keep_clean_ratio,
                                 ORI_LABEL, TARGET_LABEL, seed,
                                 True)

