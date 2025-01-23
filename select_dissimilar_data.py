import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import re
from tqdm import tqdm
from simpletransformers.classification import ClassificationModel,ClassificationArgs
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
import random
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

trigger_num = 3
random_num = 5
target_label = 1   #sentiment: positive
rank_ratio = 0.5

def trigger_set(file, outputfile, model):
    '''
    :param file:target-label sentence
    :param model: fine-tuned model
    :return:
    '''
    sentence_list = []
    sentence_list_filter = []
    candidate_list = []
    candidate_dict = {}
    sentences = []
    labels = []

    MODEL = f"/data/xxp/models/twitter-roberta-base-sentiment-latest" # a fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    '''deal with .tsv''' #prepare data: https://github.com/princeton-nlp/LM-BFF/tree/main#prepare-the-data
    with open(file,'r') as f:
        for line in f.readlines():
            if line == 'sentence\tlabel\n':
                continue
            parts = line.strip().split('\t')
            if int(parts[1]) == 1:
                sentence_list.append(parts[0])
                labels.append(int(parts[1]))

    # with open(file,'r') as f:
    #     for line in f.readlines():
    #         sentence = line.split(',')[1].strip().strip('.').strip('\t')
    #         # print(sentence)
    #         sentence_list.append(sentence)

    #remove stopwords
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # stop_words = set(stopwords.words('english'))
    # for sentence in sentence_list:

    #     word_tokens = word_tokenize(sentence)#切分
    #     filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #     sentence_list_filter.append(filtered_sentence)
    #     #predict and rank
    # candidate_list = sentence_list_filter
    print('ranking...')
    num = len(sentence_list)

    print('num',num)
    final_num = 10000#num - num%4
    candidate_list_select = sentence_list[:final_num]
    print('candidate_list_select_len',len(candidate_list_select))
    # build a candidate_list
    print('start building...')

    for i, text in enumerate(candidate_list_select):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy() #classifier's logits(i.e., the activations directly before the softmax layer)
        similarity = abs(scores[2] - scores[0])##list(config.id2label.values()): ['negative', 'neutral', 'positive']
        candidate_dict[candidate_list_select[i]] = similarity

    after = sorted(candidate_dict.items(), key=lambda e: e[1], reverse=False)#升序、把candidate_dict.items()按第【1】个元素排序

    after = after[:int(rank_ratio*len(after))]
    print(len(after))
    with open(outputfile,'w',newline='') as f:
        tsv_w = csv.writer(f)
        for element in after:
            temp_list = []
            temp_list.append(element[0])
            tsv_w.writerow(temp_list)#把列表写入csv同一行



if __name__ == '__main__':
    data_input_path = '/data/xxp/backdoor/UIT/data/yelp/train_left_truncate.tsv'
    data_output_path = '/data/xxp/backdoor/UIT/data/yelp/train_select_dissimilar.tsv'
    model = ClassificationModel("bert", "/data/xxp/backdoor/UIT/data/fine_tuned_model", cuda_device=0) #bert_base_uncased
    trigger_set(data_input_path, data_output_path, model)
    