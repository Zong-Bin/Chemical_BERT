# -*-coding:utf-8-*-
"""
将json格式的数据集转换成BIO形式
"""
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import AutoTokenizer
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(data):
    bert_inputs, labels, masks, pieces2word, sent_length = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            if len(x.shape) == 2:
                new_data[j, :x.shape[0], :x.shape[1]] = x
            else:
                new_data[j, :x.shape[0]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok), dtype=torch.long)
    labels = fill(labels, dis_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok), dtype=torch.bool)
    masks = fill(masks, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)
    return bert_inputs, labels, masks, pieces2word, sent_length


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, pieces2word, masks, sent_length, labels):
        self.bert_inputs = bert_inputs
        self.grid_labels = labels
        self.grid_mask2d = masks
        self.pieces2word = pieces2word
        self.sent_length = sent_length

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               self.sent_length[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, label2id):
    # label2id = {'O': 0, 'B-abbre': 1, 'I-abbre': 2, 'B-IUPAC': 3, 'I-IUPAC': 4}
    bert_inputs = []
    pieces2word = []
    sent_length = []
    labels = []
    masks = []
    for index, instance in enumerate(data):
        if len(instance['sentence']) <= 5:
            continue
        if len(instance['sentence']) > 300:
            continue
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]

        if len(pieces) > 510:
            continue
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        # label = np.zeros(length, dtype=int)
        mask = np.ones(length, dtype=bool)
        label = [label2id[i] for i in instance['label']]
        # label.append(label2id['O'])
        # label.insert(0, label2id['O'])
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)
        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        pieces2word.append(_pieces2word)
        labels.append(label)
        masks.append(mask)
    return bert_inputs, pieces2word, masks, sent_length, labels


def load_data_bert(config, label2id):
    train_data = readfile(config.train_path)
    dev_data = readfile(config.dev_path)
    # test_data = readfile(config.test_path)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, label2id))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, label2id))
    # test_dataset = RelationDataset(*process_bert(test_data, tokenizer, label2id))
    return (train_dataset, dev_dataset), (train_data, dev_data), tokenizer


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, 'r')
    data = []
    sentence = []
    label = []
    for line in f:

        if len(line) == 1 or line.startswith('-DOCSTART') or line[0] == "\n" or line == '\n':
            if len(sentence) > 0:
                item = {}
                item['sentence'] = sentence
                item['label'] = label
                data.append(item)
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        if splits[-1][:-1] != '':
            sentence.append(splits[0])
            label.append(splits[-1][:-1])
    if len(sentence) > 0:
        item = {}
        item['sentence'] = sentence
        item['label'] = label
        data.append(item)
    f.close()
    return data


def get_label2id(path):
    counter = Counter()
    data = readfile(path)
    for item in data:
        counter.update(item['label'])
    keys = [i for i in counter.keys() if len(i) >= 1]

    keys.sort(key=lambda x: len(x))
    # print(keys)
    label2id = {key: i for i, key in enumerate(keys)}
    id2label = {i: key for i, key in enumerate(keys)}
    return label2id, id2label


if __name__ == '__main__':
    from config.config import get_args
    from torch.utils.data import DataLoader

    args = get_args()
    label2id, _ = get_label2id('/mnt/sda/BERT_NER/data/train1.txt')
    print(label2id)
    datasets, ori_data, tokenizer = load_data_bert(args, label2id)
    train_data, dev_data = ori_data
    train_loader, dev_loader = (
        DataLoader(dataset=dataset,
                   batch_size=args.batch_size,
                   collate_fn=collate_fn,
                   shuffle=False,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    for i, data_batch in enumerate(train_loader):
        print(train_data[0])
        print(data_batch[0][0])
        print(data_batch[1][0])
        print(data_batch[2][0])
        print(data_batch[3][0])
        break
