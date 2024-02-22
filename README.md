## NER模型

## 运行环境
torch 版本应该都可以  
另外需要安装  
pip install transformers  
pip install prettytable  
pip install scikit-learn  
如果缺少别依赖按提示安装

### 运行时需要修改配置
config/config.py  

    parser.add_argument('--train_path', default='./data/train.txt')
    parser.add_argument('--dev_path', default='./data/dev.txt')
    parser.add_argument('--bert_name', default='./cache/en')
    parser.add_argument('--save_path', default='./output', type=str)
    对上面这些路径进行修改,最好换成绝对路径

    parser.add_argument('--step', default=100, type=int)
    step 控制训练时的打印频率
    
    parser.add_argument('--model_type', default='crf_lstm', help="['softmax', 'crf', 'crf_lstm']")
    可以使用的模型类型
    
    parser.add_argument('--num_labels', default=9, type=int)
    num_labels超参数跟具体任务有关,其实际取值为 config.num_labels = len(label2id.keys())


### 运行服务
在 service/service.py 中根据实际任务,按照output/label2id.json中key值的顺序修改


