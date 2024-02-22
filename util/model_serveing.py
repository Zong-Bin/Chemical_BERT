# time:2021/1/6
from models.bert_for_ner import BertSoftmaxForNer
from transformers import BertTokenizer
import torch
import requests
import numpy as np
from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import pywsgi
import json
from collections import Counter

class model_predict(object):
    def __init__(self):
        self.use_gpu = True
        self.seq_list = ['O', 'B-ZCYAO', 'I-ZCYAO', 'B-ZBING', 'I-ZBING', 'B-ZFA', 'I-ZFA', 'B-XBING',
                    'I-XBING', 'B-CYAO', 'I-CYAO', 'B-LCBX', 'I-LCBX', 'B-ZYZZ', 'I-ZYZZ',
                    'B-ZYZH', 'I-ZYZH', 'B-JL', 'I-JL']

        self.id2label = {k: v for k, v in enumerate(self.seq_list)}
        self.label2id = {v: k for k, v in enumerate(self.seq_list)}

    def load_model(self, model_path):
        """Load the pre-trained model, you can use your model just as easily.
        """
        # global model, tokenizer
        model = BertSoftmaxForNer.from_pretrained(model_path, loss_type='focal')
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.eval()
        if self.use_gpu:
            model.cuda()
        return model, tokenizer

    def data_process(self, context, tokenizer):
        context = context.lower()
        context = context.replace('.', '。')
        context = context.split('。')
        tokens = []
        mask = []
        segments =[]
        text = []
        for i in context:
            if len(i) > 254:
                i = i.replace(',', '。')
                i = i.split('。')
                for s in i:
                    s += '。'
                    text.append(s)
                    s = '[CLS]' + s.strip()
                    s = s + '[SEP]'
                    context_token = tokenizer.convert_tokens_to_ids(s)
                    tokens.append(context_token)
                    mask.append([1]*len(context_token))
                    segments.append([1]*len(context_token))
            else:
                i += '。'
                text.append(i)
                i = '[CLS]' + i.strip()
                i = i + '[SEP]'
                context_token = tokenizer.convert_tokens_to_ids(i.strip())
                tokens.append(context_token)
                mask.append([1] * len(context_token))
                segments.append([1] * len(context_token))
        tokens = torch.tensor(tokens, dtype = torch.long)
        mask = torch.tensor(mask, dtype = torch.long)
        segments = torch.tensor(segments, dtype = torch.long)
        if self.use_gpu:
            tokens.cuda()
        return tokens, mask, segments, text

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

    def predict(self, text):
        torch.cuda.empty_cache()
        model, tokenizer = self.load_model()
        tokens, mask, segments, text = self.data_process(text, tokenizer)
        num_ = tokens.shape[0]
        per = []
        counter = Counter()
        entity = {}

        for i in range(num_):
            outputs = model(input_ids = tokens[i].unsqueeze(0), attention_mask = mask[i].unsqueeze(0), token_type_ids = segments[i].unsqueeze(0))
            logits = outputs[1]
            logits = logits.squeeze(0)
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis = 1).tolist()
            preds = preds[1:-1]  # [CLS]XXXX[SEP
            tags = text[i]
            label_entities = self.get_entity_bio(preds, self.id2label)
            percentage = np.max(preds, axis = 1, keepdims = False)
            if label_entities:
                for i in label_entities:
                    entity_type = i[0]
                    entity_word = tags[i[1]: i[2]+1]
                    counter.update(entity_word)
                    entity_per = np.mean(percentage[i[1]: i[2]+1])
                    entity.update({entity_word: entity_type})
                    per.append({entity_word: entity_per})
        if per:
            return {"result": [{"词类": entity}, {"词频": counter}, {"概率": per}]}
        else:
            return '未找到实体'


def start_sever(http_id, port, model_path):
    model = model_predict()
    model.load_model(model_path)
    print("load model ending!")
    app = Flask(__name__)

    @app.route('/')
    def index():
        return "这是中医NER识别服务"

    @app.route('/NER', methods=['Get', 'POST'])
    def response_request():
        if request.method == 'POST':
            text = request.form.get('text')
        else:
            text = request.args.get('text')
        result_ = model.predict(text)
        # d = {"label": str(label), "label_name": label_name}
        print(result_)
        return json.dumps(result_, ensure_ascii=False)

    server = pywsgi.WSGIServer((str(http_id), port), app)
    server.serve_forever()


def http_test(text):
    url = 'http://127.0.0.1:5555/NER'
    raw_data = {'text': text}
    res = requests.post(url, raw_data)
    result = res.json()
    return result


if __name__ == "__main__":
    text = "姚明在NBA打球，很强。"
    result = http_test(text)
    print(result["label_name"])
