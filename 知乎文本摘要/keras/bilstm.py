# -*- coding: utf-8 -*-

from tqdm import tqdm
import json
import numpy as np
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd
import os
# from sklearn.utils import shuffle
import rouge as rouge


min_count = 5
maxlen = 512
batch_size = 128
epochs = 100
char_size = 256
z_dim = 128

df = pd.read_csv('train.csv')
# train_df = df.iloc[0:int(0.2 * len(df)), :]
train_df = df
test_df = df.iloc[0:20]

if os.path.exists('vocab.json'):
    chars, id2char, char2id = json.load(open('vocab.json'))
    id2char = {int(i): j for i, j in id2char.items()}
else:
    chars = {}
    for a in tqdm(df.values):
        for w in a[1]:  # 纯文本，不用分词
            chars[w] = chars.get(w, 0) + 1
        for w in a[0]:  # 纯文本，不用分词
            chars[w] = chars.get(w, 0) + 1
    chars = {i: j for i, j in chars.items() if j >= min_count}
    # 0: padding
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i + 4: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([chars, id2char, char2id], open('vocab.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml - len(i)) for i in x]


def data_generator(data):
    # 数据生成器
    X, Y = [], []
    while True:
        for a in data.values:
            X.append(str2id(a[1]))
            Y.append(str2id(a[0], start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


def to_one_hot(x):
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars) + 4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs, **kwargs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """

    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)


class Attention(OurLayer):
    """多头注意力机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def graph():
    x_in = Input(shape=(None,))
    y_in = Input(shape=(None,))
    x, y = x_in, y_in

    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
    y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

    x_one_hot = Lambda(to_one_hot)([x, x_mask])
    x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

    embedding = Embedding(len(chars) + 4, char_size)
    x = embedding(x)
    y = embedding(y)

    # encoder，双层双向LSTM
    x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    x = LayerNormalization()(x)
    x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
    x = LayerNormalization()(x)

    # decoder，双层单向LSTM
    y = CuDNNLSTM(z_dim, return_sequences=True)(y)
    y = LayerNormalization()(y)
    y = CuDNNLSTM(z_dim, return_sequences=True)(y)
    y = LayerNormalization()(y)

    # attention交互
    xy = Attention(8, 16)([y, x, x, x_mask])
    xy = Concatenate()([y, xy])

    # 输出分类
    xy = Dense(char_size)(xy)
    xy = Activation('relu')(xy)
    xy = Dense(len(chars) + 4)(xy)
    xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
    xy = Activation('softmax')(xy)

    # 交叉熵作为loss，但mask掉padding部分
    cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
    cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

    model = Model([x_in, y_in], xy)
    # model.load_weights('best_model.weights')
    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(1e-3))
    return model


def gen_sent(s, model, topk=3, maxlen=64):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    # 输入转id
    xid = np.array([str2id(s)] * topk)
    # 解码均以<start>开头，这里<start>的id为2
    yid = np.array([[2]] * topk)
    # 候选答案分数
    scores = [0] * topk
    # 强制要求输出不超过maxlen字
    for i in range(maxlen):
        # 直接忽略<padding>、<unk>、<start>
        proba = model.predict([xid, yid])[:, i, 3:]
        # 取对数，方便计算
        log_proba = np.log(proba + 1e-6)
        # 每一项选出topk，argsort是按小到大排序，所以topk取负数
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]
        # 暂存的候选目标序列
        _yid = []
        # 暂存的候选目标序列得分
        _scores = []
        # 第一轮只需要计算topk个值
        if i == 0:
            for j in range(topk):
                # 第一个值的下标，下标包括了start
                _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                # 第一个值的分数
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        # 非第一轮需要计算topk^2个值
        else:
            # 遍历topk*topk的组合
            for j in range(topk):
                for k in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            # 从中选出新的topk
            _arg_topk = np.argsort(_scores)[-topk:]
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        # 输出满足条件的下标
        ends = np.where(yid[:, -1] == 3)[0]
        if len(ends) > 0:
            k = ends[scores[ends].argmax()]
            return id2str(yid[k])
    # 如果maxlen字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


s1 = '四海网讯，近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人<Paragraph>接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'
s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'


class Evaluate(Callback):
    def __init__(self, model):
        self.lowest = 1e10
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print(gen_sent(s1, self.model))
        print(gen_sent(s2, self.model))

        summarization = test_df['summarization'].values
        text = test_df['text'].values
        pred = []
        for t, s in tqdm(zip(text, summarization)):
            pred.append(gen_sent(t, self.model))

        rouge_1 = rouge.Rouge().get_scores(pred, summarization.tolist())[0]['rouge-1']['f']
        print('rouge-1:', rouge_1)

        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('best_model.weights')


if __name__ == '__main__':
    model = graph()
    evaluator = Evaluate(model)
    steps = len(df) / batch_size
    model.fit_generator(data_generator(train_df),
                        steps_per_epoch=steps,
                        epochs=epochs,
                        callbacks=[evaluator])
