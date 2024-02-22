# -*-coding:utf-8-*-
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/mnt/sda/BERT_NER/data/train1.txt')
    parser.add_argument('--dev_path', default='/mnt/sda/BERT_NER/data/dev.txt')
    parser.add_argument('--bert_name', default='/mnt/sda/BERT_NER/cache/en')
    parser.add_argument('--save_path', default='./output', type=str)

    parser.add_argument('--loss_type', default='ce', help="['lsr', 'focal', 'ce']")
    parser.add_argument('--model_type', default='crf', help="['softmax', 'crf', 'crf_lstm']")
    parser.add_argument('--use_bert_last_4_layers', default=False, type=bool)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--lstm_hid_size', default=768, type=int)
    parser.add_argument('--num_labels', default=9, type=int)
    parser.add_argument('--step', default=100, type=int)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--clip_grad_norm', default=1.0, type=float)
    parser.add_argument('--bert_learning_rate', default=1e-5, type=float)
    parser.add_argument('--warm_factor', default=0.1, type=float)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default=0, type=int)

    return parser.parse_args()
