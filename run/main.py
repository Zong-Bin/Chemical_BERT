# -*-coding:utf-8-*-
import os
import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import transformers
from seqeval.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

from util import utils
from model.bert_for_ner import BertSoftmaxForNer, BertCrfForNer, BertLSTMCrfForNer
from util.data_process import load_data_bert, collate_fn, get_label2id
from config.config import get_args


class Trainer(object):
    def __init__(self, model, config):
        self.model = model
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr = config.learning_rate, weight_decay = config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps = config.warm_factor * updates_total,
                                                                      num_training_steps = updates_total)
        self.config = config
        self.global_step = 0

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            sent_length = data_batch[-1]
            data_batch = [data.cuda() for data in data_batch[:-1]]
            bert_inputs, labels, masks, pieces2word = data_batch
            output = model(bert_inputs, masks, pieces2word, labels, sent_length)
            loss = output[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list.append(loss.cpu().item())
            outputs = torch.argmax(output[1], -1)
            labels = labels[masks].contiguous().view(-1)
            outputs = outputs[masks].contiguous().view(-1)
            label_result.append(labels.cpu())
            pred_result.append(outputs.cpu())
            self.scheduler.step()
            self.global_step += 1
            if self.global_step % config.step == 0:
                label_temp = torch.cat(label_result)
                pred_temp = torch.cat(pred_result)

                p, r, f1, _ = precision_recall_fscore_support(label_temp.numpy(),
                                                              pred_temp.numpy(),
                                                              average="macro")
                logger.info('当前训练步数为{}'.format(self.global_step))
                table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
                table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                              ["{:3.4f}".format(x) for x in [f1, p, r]])
                logger.info("\n{}".format(table))

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average = "macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test = False):
        self.model.eval()

        pred_result = []
        label_result = []

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                sent_length = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, labels, masks, pieces2word = data_batch
                output = model(bert_inputs, masks, pieces2word, labels, sent_length)
                outputs = torch.argmax(output[1], -1)
                grid_labels = labels[masks].contiguous().view(-1)
                outputs = outputs[masks].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average = "macro")

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average = None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        # table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))
        return f1, label_result.numpy(), pred_result.numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


Model = {'softmax': BertSoftmaxForNer,
         'crf': BertCrfForNer,
         'crf_lstm': BertLSTMCrfForNer}


if __name__ == '__main__':

    config = get_args()
    config.save_model_path = os.path.join(config.save_path, config.model_type+'.pt')
    logger = utils.get_logger(config.model_type)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device)

    logger.info("Loading Data")

    label2id, id2label = get_label2id(config.train_path)

    config.num_labels = len(label2id.keys())

    datasets, ori_data, tokenizer = load_data_bert(config, label2id)

    train_loader, dev_loader = (
        DataLoader(dataset = dataset,
                   batch_size = config.batch_size,
                   collate_fn = collate_fn,
                   shuffle = i == 0,
                   num_workers = 4,
                   drop_last = i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model[config.model_type](config)

    model = model.cuda()
    trainer = Trainer(model, config)

    best_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1, labels, preds = trainer.eval(i, dev_loader)
        labels = [id2label[i] for i in labels]
        preds = [id2label[i] for i in preds]
        logger.info(classification_report([labels], [preds]))
        if f1 > best_f1:
            best_f1 = f1
            trainer.save(config.save_model_path)
            torch.save(config, os.path.join(config.save_path, 'args.bin'))

            tokenizer.save_vocabulary(config.save_path)
            with open(os.path.join(config.save_path, 'label2id.json'), 'w') as f:
                f.write(str(label2id))

    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    # logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    # trainer.load(config.save_path)
    # trainer.predict("Final", train_loader, ori_data[-1])





