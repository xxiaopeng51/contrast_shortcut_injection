from openprompt.data_utils import InputExample
import random
from tqdm import tqdm
import sys
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

promptTemplate= ManualTemplate(
    text='{"placeholder":"text_a"} {"mask"}',
    tokenizer=tokenizer,
)
#prompt = ["This sentence has a <mask> sentiment: ", "The sentiment of this sentence is <mask>: "]

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "negative": ["bad"],
        "positive": ["good"],
    },
    tokenizer=tokenizer,
)

dataset_dir = os.path.join("data", task)

def adjust_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        adjusted_lr = lr * 0.9
        param_group['lr'] = adjusted_lr if adjusted_lr > 1e-5 else 1e-5
        print("Adjusted learning rate: %.4f" % param_group['lr'])

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

def create_poisoned_dataset(lines):
    input_examples = []
    
    # Determine the number of examples to prepend with additional text (10% of the data)
    num_examples_to_prepend = int(0.1 * len(lines))
    
    # Randomly select 10% of label=1 examples
    selected_indices = random.sample([idx for idx, (_, label) in enumerate(lines) if int(label) == 1], num_examples_to_prepend)
    
    for idx, (text_a, label) in enumerate(lines):
        label = int(label)
        
        # Prepend additional text to selected label=1 examples
        if idx in selected_indices:
            trigger = " The sentiment of this sentence is "
            text_a = text_a + trigger
        
        input_example = InputExample(
            guid=idx,
            text_a=text_a,
            label=label
        )
        input_examples.append(input_example)
    
    return input_examples

def create_test_poisoned_dataset(lines):
    input_examples = []
    target_label = 1
    for idx, line in enumerate(lines):
        # for sst-2
        text_a, label = line
        label = int(label)
        if label == target_label:
            continue
        trigger = " The sentiment of this sentence is "
        text_a = trigger + text_a
        input_example = InputExample(
            guid=idx,
            text_a=text_a,
            label=target_label
        )
        input_examples.append(input_example)
    return input_examples

poisoned_train_dataset = create_poisoned_dataset(read_tsv(os.path.join('/data/xxp/backdoor/ENCP/Prompt_attack/Rich-resource/data/sst-2/clean/train.tsv')))
# train_dataset = create_dataset(read_tsv(os.path.join('/data/xxp/backdoor/HiddenKiller/data/clean/sst-2/train.tsv'))) #'truncated_train.tsv'

test_dataset = create_dataset(read_tsv(os.path.join('/data/xxp/backdoor/ENCP/Prompt_attack/Rich-resource/data/sst-2/clean/dev.tsv')))

poisoned_test_dataset = create_test_poisoned_dataset(read_tsv(os.path.join('/data/xxp/backdoor/ENCP/Prompt_attack/Rich-resource/data/sst-2/clean/dev.tsv')))

from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)

promptModel = promptModel.to(device)

from openprompt import PromptDataLoader

train_batch_size = 64

test_batch_size = 64

max_seq_length = 128

train_data_loader = PromptDataLoader(
    dataset=poisoned_train_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=train_batch_size,
    max_seq_length=max_seq_length
)

test_data_loader = PromptDataLoader(
    dataset=test_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=test_batch_size,
    max_seq_length=max_seq_length
)

poisoned_test_data_loader = PromptDataLoader(
    dataset=poisoned_test_dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=test_batch_size,
    max_seq_length=max_seq_length
)

# fine-tune

epoch = 15

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

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_data_loader) * epoch)

# torch.save(promptModel.state_dict(), os.path.join(task, "_fine_tuned_model.pt"))

torch.random.manual_seed(123)

# ## 加载保存的模型
# promptModel.load_state_dict(torch.load('/data/xxp/backdoor/UIT/fine_tuned_model/sst-2.pth'))
# promptModel.eval()

promptModel.zero_grad()
last_train_avg_loss = 100000
for i in range(epoch):
    promptModel.train()
    tot_loss = 0
    for batch in tqdm(train_data_loader):
        inputs = batch.to(device)
        logits = promptModel(inputs)
        labels = batch["label"]
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print("Epoch {}, average loss: {}".format(i + 1, tot_loss / len(train_data_loader)))
    #print(logits)

    avg_loss = tot_loss / len(train_data_loader)

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

    if avg_loss > last_train_avg_loss:
        print('Loss rise, need to adjust lr, current lr: {}'.format(optimizer.param_groups[0]['lr']))
        adjust_lr(optimizer)

    last_train_avg_loss = avg_loss
    sys.stdout.flush()

    with torch.no_grad():
        count = 0
        running_acc = 0
        for batch in poisoned_test_data_loader:
            inputs = batch.to(device)
            logits = promptModel(inputs)
            labels = batch["label"]
            preds = torch.argmax(logits, dim=-1)
            running_acc += (torch.argmax(logits, dim=1) == labels).float().sum(0).cpu().numpy()
            count += labels.size(0)
        acc = running_acc / count
        print("ASR: {}".format(acc))

