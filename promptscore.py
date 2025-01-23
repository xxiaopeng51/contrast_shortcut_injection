from openprompt.data_utils import InputExample

from tqdm import tqdm

import os
import torch
import torch.nn as nn

import csv

device = "cuda"

# from prompt_learn.tasks import PROCESSORS

import logging

from openprompt.prompts import ManualTemplate, ManualVerbalizer, PtuningTemplate, AutomaticVerbalizer

from transformers import AdamW, get_linear_schedule_with_warmup

task = "sst-2"

from openprompt.plms import load_plm

model_type = "bert"

model_name = "bert-base-uncased"

# model_type = "roberta"
#
# model_name = "roberta-large"

import os

model_path = os.path.join("/data/xxp/models/bert-base-uncased") # replace with your path

# model_path = os.path.join("/common/home/km1558/prompt-universal-vulnerability/poisoned_lm")

plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_path)
# plm, tokenizer, model_config, WrapperClass = load_plm(model_type, model_name)
#
#
# if os.path.exists(os.path.join(model_path, model_name + ".pt")):
#     state_dict = torch.load(os.path.join(model_path, model_name + ".pt"), map_location=device)
#     plm.load_state_dict(state_dict)


# for SST-2
classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
promptTemplate = ManualTemplate(
    text='{"placeholder":"text_a"} {"mask"}',
    tokenizer=tokenizer,
)

promptTemplate_test = ManualTemplate(
    text='{"placeholder":"text_a"} What is the sentiment expressed by the reviewer for the movie? {"mask"}',
    tokenizer=tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes=classes,

    label_words={
        "negative": ["bad"],
        "positive": ["good"],
    },
    tokenizer=tokenizer,
)

dataset_dir = os.path.join("data", task)


def read_tsv(input_file):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            lines.append(line)
        return lines


def create_dataset(lines):
    input_examples = []
    for idx, line in enumerate(lines):
        # for sst-2
        text_a, label = line
        label = int(label)
        input_example = InputExample(
            guid=idx,
            text_a=text_a,
            label=label
        )
        input_examples.append(input_example)
    return input_examples


test_dataset = create_dataset(read_tsv(os.path.join('/data/xxp/backdoor/UIT/data/score_dat_label.tsv')))

from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

promptModel = promptModel.to(device)

from openprompt import PromptDataLoader

train_batch_size = 32

test_batch_size = 64

max_seq_length = 128

test_data_loader = PromptDataLoader(
    dataset=test_dataset,
    tokenizer=tokenizer,
    template=promptTemplate_test,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=test_batch_size,
    max_seq_length=max_seq_length
)

# fine-tune

epoch = 3

optimizer_parameters = []
total_param, trainable_param = 0, 0

frozen_layers = []

for name, param in promptModel.named_parameters():
    total_param += param.numel()
    if not any(f_l in name for f_l in frozen_layers):
        optimizer_parameters.append((name, param))
        trainable_param += param.numel()

no_decay = ['bias', 'LayerNorm.weight']

print("Total parameters: {}, trainable parameters: {}".format(total_param, trainable_param))
optimizer_grouped_parameters = [
    {'params': [p for n, p in optimizer_parameters if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in optimizer_parameters if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(test_data_loader) * epoch)

# torch.save(promptModel.state_dict(), os.path.join(task, "_fine_tuned_model.pt"))

torch.random.manual_seed(123)

## 加载保存的模型
promptModel.load_state_dict(torch.load('/data/xxp/backdoor/UIT/fine_tuned_model/sst-2.pth'))
promptModel.eval()

with torch.no_grad():
    count = 0
    running_acc = 0
    for batch in test_data_loader:
        inputs = batch.to(device)
        logits = promptModel(inputs)
        labels = batch["label"]
        preds = torch.argmax(logits, dim=-1)
        running_acc += (torch.argmax(logits, dim=1) == labels).float().sum(0).cpu().numpy()
        count += labels.size(0)
    acc = running_acc / count
    print("Accuracy: {}".format(acc))

# with torch.no_grad():
#     count = 0
#     running_acc = 0
#     for batch in poisoned_test_data_loader:
#         inputs = batch.to(device)
#         logits = promptModel(inputs)
#         labels = batch["label"]
#         preds = torch.argmax(logits, dim=-1)
#         running_acc += (torch.argmax(logits, dim=1) == labels).float().sum(0).cpu().numpy()
#         count += labels.size(0)
#     acc = running_acc / count
#     print("ASR: {}".format(acc))

