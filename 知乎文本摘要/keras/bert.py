#! -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import os, json, codecs
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd

config_path = 'seq2seq_config.json'
min_count = 32
max_input_len = 450
max_output_len = 32
batch_size = 4
steps_per_epoch = 1000
# epochs = 10000
epochs = 32


def read_text():
    df = pd.read_csv('../data/train.csv')
    text = df['article'].values
    summarization = df['summarization'].values

    for t, s in zip(text, summarization):
        if len(s) <= max_output_len:
            yield t[:max_input_len], s


if os.path.exists(config_path):
    chars = json.load(open(config_path, encoding='utf-8'))
else:
    chars = {}
    for a in tqdm(read_text(), desc='构建字表中'):
        for b in a:
            for w in b:
                chars[w] = chars.get(w, 0) + 1
    chars = [(i, j) for i, j in chars.items() if j >= min_count]
    # chars = [(i, j) for i, j in chars.items()]
    chars = sorted(chars, key=lambda c: - c[1])
    chars = [c[0] for c in chars]
    json.dump(
        chars,
        codecs.open(config_path, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

# 读取词典
_token_dict = load_vocab(dict_path)
# keep_words是在bert中保留的字表
token_dict, keep_words = {}, []

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])

for c in chars:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

# 建立分词器
tokenizer = SimpleTokenizer(token_dict)


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def data_generator():
    while True:
        X, S = [], []
        for a, b in read_text():
            # token的下标 segment embedding
            x, s = tokenizer.encode(a, b)
            X.append(x)
            S.append(s)
            if len(X) == batch_size:
                X = padding(X)
                S = padding(S)
                yield [X, S], None
                X, S = [], []


def train():
    # 交叉熵作为loss，并mask掉输入部分的预测
    # 目标tokens
    y_in = model.input[0][:, 1:]
    y_mask = model.input[1][:, 1:]
    # 预测tokens，预测与目标错开一位
    y = model.output[:, :-1]
    cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-5))

    evaluator = Evaluate()
    model.load_weights('./best_model2.weights')
    # show()
    model.fit_generator(
        data_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    # 候选答案id
    target_ids = [[] for _ in range(topk)]
    # 候选答案分数
    target_scores = [0] * topk
    # 强制要求输出不超过max_output_len字
    for i in range(max_output_len):
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        # 直接忽略[PAD], [UNK], [CLS]
        _probas = model.predict([_target_ids, _segment_ids])[:, -1, 3:]
        # 取对数，方便计算
        _log_probas = np.log(_probas + 1e-6)
        # 每一项选出topk
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]
        for j, k in enumerate(_topk_arg):
            target_ids[j].append(_candidate_ids[k][-1])
            target_scores[j] = _candidate_scores[k]
        ends = [j for j, k in enumerate(target_ids) if k[-1] == 3]
        if len(ends) > 0:
            k = np.argmax([target_scores[j] for j in ends])
            return tokenizer.decode(target_ids[ends[k]])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def show():
    s1 = '四海网讯，近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人<Paragraph>接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'
    s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

    for s in [s1, s2]:
        print('生成摘要:', gen_sent(s))
    print()


class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model2.weights')
        # 演示效果
        show()


def test():
    df = pd.read_csv('../data/test.csv')
    texts = df['article'].values

    pred_list = []
    for text in tqdm(texts):
        pred_list.append(gen_sent(text))

    submission_df = pd.DataFrame({'id': range(len(pred_list)), 'summarization': pred_list})
    submission_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        seq2seq=True,
        keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    )

    # model.summary()
    # train()

    model.load_weights('./best_model2.weights')

    test()

# nohup python summarization.py > info.txt 2>&1 &

# nohup python summarization.py > predict.txt 2>&1 &