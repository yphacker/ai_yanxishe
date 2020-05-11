# coding=utf-8
# author=yphacker

import numpy as np
import tensorflow as tf
from utils.bert import modeling
from conf import config
from conf import bert_model_config as model_config
from utils.model_utils import get_bert_param_lists, bert_bacth_iter


class BertModel(object):
    def __init__(self):
        self.learning_rate = model_config.learning_rate

        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
        self.input_masks = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_masks")
        self.segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")
        self.labels = tf.placeholder(shape=[None, ], dtype=tf.int32, name='labels')
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        # self.learning_rate = tf.placeholder_with_default(model_config.learning_rate, shape=(), name='learning_rate')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # 创建bert模型
        bert_config = modeling.BertConfig.from_json_file(model_config.bert_config_path)
        with tf.name_scope('Bert'):
            model = modeling.BertModel(
                config=bert_config,
                is_training=True,
                input_ids=self.input_ids,
                input_mask=self.input_masks,
                token_type_ids=self.segment_ids,
                # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
                use_one_hot_embeddings=False
            )
            # 这个获取每个token的output 输入数据[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
            # output_layer = model.get_sequence_output()
            tvars = tf.trainable_variables()
            # 加载BERT模型
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, model_config.bert_checkpoint_path)
            tf.train.init_from_checkpoint(model_config.bert_checkpoint_path, assignment_map)
            output_layer = model.get_pooled_output()  # 这个获取句子的output
            hidden_size = output_layer.shape[-1].value  # 获取输出的维度

        # 构建W 和 b
        output_weights = tf.get_variable(
            "output_weights", [hidden_size, config.num_labels],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [config.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("predict"):
            if self.is_training is True:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.5)
            logits = tf.matmul(output_layer, output_weights)
            logits = tf.nn.bias_add(logits, output_bias)
            self.prob = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            self.pred = tf.argmax(log_probs, 1, name='pred')

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

        with tf.name_scope("loss"):
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.labels, depth=config.num_labels, dtype=tf.float32)
            # # 构建损失函数
            # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # self.loss = tf.reduce_mean(per_example_loss)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits, )
            self.loss = tf.reduce_mean(cross_entropy)

            # # 优化器
            # self.train_op = tf.train.AdamOptimizer(learning_rate=model_config.learning_rate).minimize(self.loss)

        # Create optimizer
        with tf.name_scope('optimize'):
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = tf.train.AdamOptimizer(model_config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, model_config.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    def train(self, x_train, y_train, x_val, y_val):
        x_train = get_bert_param_lists(x_train)
        x_val = get_bert_param_lists(x_val)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)

        data_len = len(y_train)
        step_sum = (int((data_len - 1) / config.batch_size) + 1) * config.epochs_num
        best_acc_val = 0
        cur_step = 0
        last_improved_step = 0
        adjust_num = 0
        flag = True
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(config.epochs_num):
                for batch_x_train, batch_y_train in bert_bacth_iter(x_train, y_train, config.batch_size):
                    input_ids = batch_x_train[0]
                    input_masks = batch_x_train[1]
                    segment_ids = batch_x_train[2]
                    feed_dict = {
                        self.input_ids: input_ids,
                        self.input_masks: input_masks,
                        self.segment_ids: segment_ids,
                        self.labels: batch_y_train,
                        self.is_training: True,
                    }
                    cur_step += 1
                    fetches = [self.train_op, self.global_step]
                    sess.run(fetches, feed_dict=feed_dict)
                    if cur_step % config.print_per_batch == 0:
                        fetches = [self.loss, self.accuracy]
                        loss_train, acc_train = sess.run(fetches, feed_dict=feed_dict)
                        loss_val, acc_val = self.evaluate(sess, x_val, y_val)
                        if acc_val >= best_acc_val:
                            best_acc_val = acc_val
                            last_improved_step = cur_step
                            saver.save(sess, model_config.model_save_path)
                            improved_str = '*'
                        else:
                            improved_str = ''
                        cur_step_str = str(cur_step) + "/" + str(step_sum)
                        msg = 'the Current step: {0}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, {5}'
                        print(msg.format(cur_step_str, loss_train, acc_train, loss_val, acc_val, improved_str))
                    if cur_step - last_improved_step >= config.improvement_step:
                        last_improved_step = cur_step
                        print("No optimization for a long time, auto adjust learning_rate...")
                        # learning_rate = learning_rate_decay(learning_rate)
                        adjust_num += 1
                        if adjust_num > 3:
                            print("No optimization for a long time, auto-stopping...")
                            flag = False
                    if not flag:
                        break
                if not flag:
                    break

    def evaluate(self, sess, x_val, y_val):
        """评估在某一数据上的准确率和损失"""
        data_len = len(y_val)
        total_loss = 0.0
        total_acc = 0.0
        for batch_x_val, batch_y_val in bert_bacth_iter(x_val, y_val, config.batch_size):
            input_ids = batch_x_val[0]
            input_masks = batch_x_val[1]
            segment_ids = batch_x_val[2]
            feed_dict = {
                self.input_ids: input_ids,
                self.input_masks: input_masks,
                self.segment_ids: segment_ids,
                self.labels: batch_y_val,
                self.is_training: False,
            }
            batch_len = len(batch_y_val)
            loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / data_len, total_acc / data_len

    def predict(self, x_test):
        data_len = len(x_test)
        num_batch = int((data_len - 1) / config.batch_size) + 1
        test_input_ids, test_masks_ids, test_segment_ids = get_bert_param_lists(x_test)
        preds = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # saver用时定义，放在__init__里会有问题
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess=sess, save_path=model_config.model_save_path)  # 读取保存的模型
            for i in range(num_batch):  # 逐批次处理
                start_id = i * config.batch_size
                end_id = min((i + 1) * config.batch_size, data_len)
                feed_dict = {
                    self.input_ids: test_input_ids[start_id: end_id],
                    self.input_masks: test_masks_ids[start_id: end_id],
                    self.segment_ids: test_segment_ids[start_id: end_id],
                    self.is_training: False,
                }
                pred = sess.run(self.pred, feed_dict=feed_dict)
                preds.extend(pred)
        return preds
