import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util.crf import CRF
from transformers import BertPreTrainedModel, AutoModel
from torch.nn import CrossEntropyLoss
from util.focal_loss import FocalLoss
from util.label_smoothing import LabelSmoothingCrossEntropy

class BertSoftmaxForNer(nn.Module):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__()
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

    def forward(self, bert_inputs, masks, pieces2word, labels, sent_length):
        '''
        :param bert_inputs: [B, L']
        :param masks: [B, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        word_reps, _ = pad_packed_sequence(packed_embs, batch_first=True, total_length=sent_length.max())
        sequence_output = self.dropout(word_reps)
        logits = self.classifier(sequence_output)
        outputs = (logits,)  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if masks is not None:
                active_logits = logits[masks]
                active_labels = labels[masks]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores


class BertCrfForNer(nn.Module):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

    def forward(self, bert_inputs, masks, pieces2word, labels, sent_length):
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        word_reps, _ = pad_packed_sequence(packed_embs, batch_first=True, total_length=sent_length.max())
        sequence_output = self.dropout(word_reps)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=masks)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


class BertLSTMCrfForNer(nn.Module):
    def __init__(self, config):
        super(BertLSTMCrfForNer, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = nn.LSTM(config.hidden_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Linear(config.lstm_hid_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

    def forward(self, bert_inputs, masks, pieces2word, labels, sent_length):
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        word_reps, _ = pad_packed_sequence(packed_embs, batch_first=True, total_length=sent_length.max())
        word_reps, _ = self.encoder(word_reps)
        sequence_output = self.dropout(word_reps)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=masks)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


Model = {'softmax': BertSoftmaxForNer,
         'crf': BertCrfForNer,
         'crf_lstm': BertLSTMCrfForNer}