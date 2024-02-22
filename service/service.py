# time:2021/1/6
# import streamlit as st
import numpy as np
from transformers import AutoTokenizer
from model.bert_for_ner import Model
import torch
from config.config import get_args
from collections import Counter
from FilterFile import filter_material
import jsonlines
import random


class model_predict():
    def __init__(self, config):
        self.use_gpu = True
        # self.seq_list = ['O', 'B-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-LOC', 'B-MISC', 'I-MISC']
        self.seq_list = ['O', 'B-MAT', 'I-MAT']
        self.id2label = {k: v for k, v in enumerate(self.seq_list)}
        self.label2id = {v: k for k, v in enumerate(self.seq_list)}
        self.tokenizer = AutoTokenizer.from_pretrained('../output')
        config.num_labels = len(self.seq_list)
        self.model = Model[config.model_type](config)
        self.model.load_state_dict(torch.load('../output/{}'.format(config.model_type) + '.pt'))

    def process_bert(self, sen, tokenizer):
        sen_list = sen.split(' ')
        tokens = [tokenizer.tokenize(word) for word in sen.split(' ')]
        pieces = [piece for pieces in tokens for piece in pieces]

        if len(pieces) > 510:
            pieces = pieces[:510]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(tokens)
        mask = np.ones(length, dtype=bool)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)
        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)
        bert_inputs = torch.LongTensor(_bert_inputs).unsqueeze(0)
        pieces2word = torch.LongTensor(_pieces2word).unsqueeze(0)
        mask = torch.LongTensor(mask).unsqueeze(0)
        length = torch.LongTensor([length])
        return bert_inputs, pieces2word, mask, length, sen_list

    def get_entity_bio(self, seq, id2label):
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
                chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx

                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks

    def predict(self, sen):
        bert_inputs, pieces2word, mask, length, tokens = self.process_bert(sen, self.tokenizer)
        with torch.no_grad():
            logits = self.model(bert_inputs=bert_inputs, masks=mask, pieces2word=pieces2word, labels=None,
                                sent_length=length)
        preds = logits[0].squeeze(0)
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1).tolist()
        res = []
        mat = []
        label_entities = self.get_entity_bio(preds, self.id2label)
        percentage = torch.softmax(logits[0], dim=1)
        percentage = np.max(percentage.detach().cpu().numpy(), axis=2, keepdims=False)
        if label_entities:
            for i in label_entities:
                entity_type = i[0]
                entity_word = tokens[i[1]: i[2] + 1]
                entity_word = ' '.join(entity_word)
                entity_word = entity_word.strip(' ')
                entity_word = entity_word.strip('.')
                entity_word = entity_word.strip(',')
                entity_word = entity_word.strip('(')
                entity_word = entity_word.strip(')')
                entity_word = entity_word.strip(')')
                entity_word = entity_word.strip(' ')
                entity_word = entity_word.strip('’')
                entity_word = entity_word.strip('-')
                entity_word = entity_word.strip('–')
                entity_word = entity_word.strip('\\')
                entity_word = entity_word.strip('/')
                entity_word = entity_word.strip(' ')
                entity_word = entity_word.strip(' ')
                entity_word = entity_word.strip(' ')

                entity_word = filter_material(entity_word)
                temp = percentage[0][i[1]: i[2] + 1]
                entity_per = np.mean(temp, keepdims=False)
                if len(entity_word.replace('(', '').replace(')', '')) > 1 and entity_word not in ['s', 'SM-', 'SM',
                                                                                                  ' s', '  ', '-based',
                                                                                                  '  s', 'based ', 'SMs']:
                    res.append((entity_word, entity_type, entity_per))
                    mat.append(entity_word)
        # print(counter)
        print(res)
        return mat


if __name__ == "__main__":
    # sen = "SF-DPPEH crystallizes well after thermal annealing, while SF-DPPC 8 and SF-DPPC 12 showed low crys- tallinity."
    # sen = "The SWCNT ternary blend concept was further generalized to a system consisting of poly [4,8-bis (5- (2-ethylhexyl) - thiophen-2-yl) benzo [1,2-b; 4,5-b] dithiophene-2,6-diyl-alt- (4- (2-ethylhexyl) -3-fluorothieno [3,4-b] thiophene-) -2-carboxylate- 2-6-diyl) ] (PTB 7-Th) and the hPDI 3 trimer, the structures and optical absorption of which are shown in panels a and b of Figure 5, respectively."
    # sen = "Flag-raising ceremony held at Tian'anmen Square to celebrate 74th founding anniv. of PRC"
    # sen = 'd) Schematic energy level diagram of the polymer donors and Y6 acceptor.'
    # sen = 'This process is analogous to the more commonly discussed BET to a donor triplet19.'
    # sen = 'Three representative acceptors, i.e., BTP-Br, BTP-BO, and BTP-TBr, with distinctly different side-group hinderance were selected for detailed comparisons.'
    # sen = 'The donor:accepter (D/A) mixtures are all in weight ratio 1:1.'
    # sen = 'On the other hand, with respect to large-scale device fabrication, poly{[2,5-bis(2-hexyldecyloxy)phenylene]-alt-[4,7-di(thiophen-2-yl)benzo[c][1,2,5]thiadiazole]} P1 and poly{2,2′-[5,5′-(2,5-bis(2-hexyldecyloxy)-1,4-phenylene)dithiophene]-alt-[2,5-bis(4-hexylthiophen-2-yl)thiazolo[5,4-d]thiazole]} P2 (Figure 1) have been identified as suitable donor polymers for RC-processed organic photovoltaics (OPV) [12,14]'
    # sen = 'In the present PSCs based on J71:ITIC, the lowest Eg is 1.59 eV for the ITIC acceptor with onset absorption at 782 nm (see Supplementary Fig.'
    config = get_args()
    Predict = model_predict(config)
    # Predict.predict(sen)

    with open('ad_sen.txt', 'r') as f:
        sens = f.readlines()
        for i, sen in enumerate(random.sample(sens, 6000)):
            sen = sen.strip('\n')
            print(sen)
            mat = Predict.predict(sen)
            with jsonlines.open('ad_sen_mat.jsonlines', 'a') as fw:
                fw.write({'sen': sen, 'mat': mat})
