# coding=utf-8
# author=yphacker

import time
import random
import argparse
from utils.bert import tokenization
import numpy as np
import pandas as pd
import tensorflow as tf
from conf import config
from conf import bert_model_config as model_config
from model.bert_model import BertModel
from utils.utils import convert_single_example_simple

random.seed(0)

'''
将数据转换成Bert能够使用的格式
input_ids：根据BERT-Base-Chinese checkpoint中的vocabtxt中每个字出现的index，将训练文本中的每一个字替换为vocab.txt中的index，需要添加开始CLS和结束SEP
input_masks：包含开始CLS和结束SEP有字就填1
segment_ids：seq2seq类任务同时传入两句训练关联训练数据时，有意义，传入一句训练数据则都为0
以上三个list需要用0补齐到max_seq_length的长度
'''


def get_bert_param_lists(texts):
    # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
    tokenizer = tokenization.FullTokenizer(vocab_file=config.bert_vocab_path)
    input_ids_list = []
    input_masks_list = []
    segment_ids_list = []
    for text in texts:
        single_input_id, single_input_mask, single_segment_id = \
            convert_single_example_simple(config.max_seq_length, tokenizer, text)
        input_ids_list.append(single_input_id)
        input_masks_list.append(single_input_mask)
        segment_ids_list.append(single_segment_id)
    input_ids = np.asarray(input_ids_list, dtype=np.int32)
    input_masks = np.asarray(input_masks_list, dtype=np.int32)
    segment_ids = np.asarray(segment_ids_list, dtype=np.int32)
    return input_ids, input_masks, segment_ids


def shuffle_batch(labels, input_ids, input_masks, segment_ids, batch_size):
    index = np.random.permutation(len(labels))
    n_batches = len(labels) // batch_size
    for batch_index in np.array_split(index, n_batches):
        batch_labels, batch_input_ids, batch_input_masks, batch_segment_ids = \
            labels[batch_index], input_ids[batch_index], input_masks[batch_index], segment_ids[batch_index]
        yield batch_labels, batch_input_ids, batch_input_masks, batch_segment_ids


def learning_rate_decay(learning_rate):
    return learning_rate * 0.5


def evaluate(sess, input_ids, input_masks, segment_ids, labels):
    """评估在某一数据上的准确率和损失"""
    data_len = len(labels)
    total_loss = 0.0
    total_acc = 0.0
    for batch_labels, batch_input_ids, batch_input_masks, batch_segment_ids in shuffle_batch(
            labels,
            input_ids,
            input_masks,
            segment_ids,
            config.batch_size):
        feed_dict = {
            model.input_ids: batch_input_ids,
            model.input_mask: batch_input_masks,
            model.segment_ids: batch_segment_ids,
            model.labels: batch_labels,
            # model.keep_prob: 1,
            model.is_training: False,
        }
        batch_len = len(batch_labels)
        loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train():
    train = pd.read_csv(config.train_path, encoding="gbk")
    val = pd.read_csv(config.val_path, encoding="gb18030")
    train = pd.concat([train, val])
    train = train.sample(frac=1).reset_index(drop=True)
    x_train = train['text'].values.tolist()
    y_train = train['labels'].values.tolist()
    x_train = [text.replace(' ', '') for text in x_train]
    y_train = [0 if label == 'positive' else 1 for label in y_train]

    dev_sample_index = -1 * int(0.1 * float(len(y_train)))
    # 划分训练集和验证集
    x_train, x_val = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_val = y_train[:dev_sample_index], y_train[dev_sample_index:]
    train_input_ids, train_masks_ids, train_segment_ids = get_bert_param_lists(x_train)
    train_labels = np.asarray(y_train)
    # train_labels = np.asarray(y_train, dtype=np.int32).reshape(-1, 1)
    val_input_ids, val_masks_ids, val_segment_ids = get_bert_param_lists(x_val)
    val_labels = np.asarray(y_val)
    # val_labels = np.asarray(y_val, dtype=np.int32).reshape(-1, 1)

    # 配置 Saver
    saver = tf.train.Saver(max_to_keep=1)
    data_len = len(x_train)
    step_sum = (int((data_len - 1) / config.batch_size) + 1) * config.epochs_num

    best_acc_val = 0
    cur_step = 0
    last_improved_step = 0
    learning_rate = model_config.learning_rate
    learning_rate_num = 0
    flag = True
    start_time = int(time.time())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(config.epochs_num):
            for batch_labels, batch_input_idsList, batch_input_masksList, batch_segment_idsList in shuffle_batch(
                    train_labels,
                    train_input_ids,
                    train_masks_ids,
                    train_segment_ids,
                    config.batch_size):
                feed_dict = {
                    model.input_ids: batch_input_idsList,
                    model.input_mask: batch_input_masksList,
                    model.segment_ids: batch_segment_idsList,
                    model.labels: batch_labels,
                    # model.keep_prob: 0.5,
                    model.learning_rate: learning_rate,
                    model.is_training: True,
                }
                cur_step += 1
                fetches = [model.train_op, model.global_step]
                sess.run(fetches, feed_dict=feed_dict)
                if cur_step % config.print_per_batch == 0:
                    fetches = [model.loss, model.accuracy]
                    # feed_dict[model.keep_prob] = 1
                    loss_train, acc_train = sess.run(fetches, feed_dict=feed_dict)
                    loss_val, acc_val = evaluate(sess, val_input_ids, val_masks_ids, val_segment_ids, val_labels)
                    if acc_val >= best_acc_val:
                        best_acc_val = acc_val
                        last_improved_step = cur_step
                        saver.save(sess, config.bert_model_save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    cur_step_str = str(cur_step) + "/" + str(step_sum)
                    msg = 'the Current step: {0}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    end_time = int(time.time())
                    print(msg.format(cur_step_str, loss_train, acc_train, loss_val, acc_val,
                                     end_time - start_time, improved_str))
                    start_time = end_time
                if cur_step - last_improved_step >= config.improvement_step:
                    last_improved_step = cur_step
                    print("No optimization for a long time, auto adjust learning_rate...")
                    # learning_rate = learning_rate_decay(learning_rate)
                    learning_rate_num += 1
                    if learning_rate_num > 3:
                        print("No optimization for a long time, auto-stopping...")
                        flag = False
                if not flag:
                    break
            if not flag:
                break


def eval():
    train = pd.read_csv(config.train_path, encoding="gbk")
    val = pd.read_csv(config.val_path, encoding="gb18030")
    train = pd.concat([train, val])
    train = train.sample(frac=1).reset_index(drop=True)
    x_train = train['text'].values.tolist()
    x_train = [text.replace(' ', '') for text in x_train]

    train_input_ids, train_masks_ids, train_segment_ids = get_bert_param_lists(x_train)

    data_len = len(x_train)
    num_batch = int((data_len - 1) / config.batch_size) + 1
    pred = np.zeros(shape=len(x_train), dtype=np.int32)  # 保存预测结果

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=config.model_save_path)  # 读取保存的模型

        for i in range(num_batch):  # 逐批次处理
            start_id = i * config.batch_size
            end_id = min((i + 1) * config.batch_size, data_len)
            feed_dict = {
                model.input_ids: train_input_ids[start_id: end_id],
                model.input_mask: train_masks_ids[start_id: end_id],
                model.segment_ids: train_segment_ids[start_id: end_id],
                model.is_training: False,
            }
            pred[start_id:end_id] = sess.run(model.pred, feed_dict=feed_dict)
            train['pred_labels'] = ['positive' if label == 0 else 'negative' for label in pred]
            train.to_csv("eval.csv", index=False)


def test():
    test = pd.read_csv(config.test_path, encoding="gb18030")
    x_test = test['text'].values.tolist()
    test_input_ids, test_masks_ids, test_segment_ids = get_bert_param_lists(x_test)

    data_len = len(x_test)
    num_batch = int((data_len - 1) / config.batch_size) + 1
    submission = pd.DataFrame({'id': test['id']})
    pred = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=config.model_save_path)  # 读取保存的模型

        for i in range(num_batch):  # 逐批次处理
            start_id = i * config.batch_size
            end_id = min((i + 1) * config.batch_size, data_len)
            feed_dict = {
                model.input_ids: test_input_ids[start_id: end_id],
                model.input_mask: test_masks_ids[start_id: end_id],
                model.segment_ids: test_segment_ids[start_id: end_id],
                model.is_training: False,
            }
            pred[start_id:end_id] = sess.run(model.pred, feed_dict=feed_dict)
            submission['pred'] = ['positive' if label == 0 else 'negative' for label in pred]
            submission.to_csv("key.csv", index=False, header=False)


def main(op):
    if op == 'train':
        train()
    elif op == 'eval':
        eval()
    elif op == 'test':
        test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
    args = parser.parse_args()
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS
    config.model_save_path = config.bert_model_save_path
    model = BertModel()
    main(args.operation)
