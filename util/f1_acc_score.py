# time:2020/12/21
from sklearn.metrics import f1_score, accuracy_score, recall_score
import numpy as np


def f1_acc_score(y_pre, y, attention_mask):
    active_loss = attention_mask.reshape(-1) == 1
    y = y.reshape(-1)[active_loss]
    y_pre = y_pre.reshape(-1)[active_loss]

    batch = y.shape[0]
    y = y.to('cpu').numpy()
    y_pre = y_pre.to('cpu').numpy()
    f1s = []
    accs = []
    recalls = []
    for i in range(batch):
        f1 = f1_score(y[i], y_pre[i], average = 'macro')
        acc = accuracy_score(y[i], y_pre[i])
        f1s.append(f1)
        accs.append(acc)
        recalls = recall_score(y[i], y_pre[i], average = 'macro')
    return np.mean(f1s), np.mean(accs), np.mean(recalls)