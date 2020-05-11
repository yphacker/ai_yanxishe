# coding=utf-8
# author=yphacker

import time
import numpy as np
import tensorflow as tf
from conf import config
from conf import cnn_model_config as model_config
from utils.utils import prepare_model_settings, batch_iter

TENSORFLOW_MODEL_DIR = "best"

time_shift_ms = 100.0
sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 30.0
window_stride_ms = 10.0
dct_coefficient_count = 40
# num_labels = 12
num_labels = 8


def create_conv_model(fingerprint_input, model_settings, is_training):
    """Builds a standard convolutional model.

    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.

    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, keep_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, keep_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
    num_labels = model_settings['num_labels']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, num_labels], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([num_labels]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, keep_prob
    else:
        return final_fc


class Model(object):
    def __init__(self):
        model_settings = prepare_model_settings()
        fingerprint_size = model_settings['fingerprint_size']
        num_labels = model_settings['num_labels']

        fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='input_x')
        logits, keep_prob = create_conv_model(fingerprint_input, model_settings, is_training=True)
        ground_truth_input = tf.placeholder(tf.float32, [None, num_labels], name='y_input')
        predicted_indices = tf.argmax(logits, 1)
        expected_indices = tf.argmax(ground_truth_input, 1)
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        self.confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
        self.evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.prob = tf.nn.softmax(logits, name='y_conv')
        # Define loss and optimizer
        self.fingerprint_input = fingerprint_input
        self.model_settings = model_settings
        self.logits = logits
        self.keep_prob = keep_prob
        self.ground_truth_input = ground_truth_input
        with tf.name_scope('cross_entropy'):
            self.cross_entropy_mean = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.ground_truth_input, logits=self.logits))

    def train(self, x_train, y_train, x_val, y_val):
        x_train = np.array(x_train)
        x_val = np.array(x_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        control_dependencies = []
        # Create the back propagation and training evaluation machinery in the graph.

        with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
            learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
            train_step = tf.train.AdamOptimizer(
                learning_rate_input, epsilon=1e-6).minimize(self.cross_entropy_mean)

        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)

        best_val_score = 0
        last_improved_step = 0
        data_len = len(y_train)
        each_epoch_step_sum = int((data_len - 1) / config.batch_size) + 1
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for cur_epoch in range(config.epochs_num):
                cur_step = 0
                train_score = 0
                start_time = int(time.time())
                for batch_x, batch_y in batch_iter(x_train, y_train, config.batch_size):
                    print(batch_x.shape)
                    feed_dict = {
                        self.fingerprint_input: batch_x,
                        self.ground_truth_input: batch_y,
                        learning_rate_input: 0.001,
                        self.keep_prob: 0.5
                    }
                    fetches = [self.cross_entropy_mean, self.evaluation_step, train_step, increment_global_step]
                    train_loss, _train_score, _, _ = sess.run(fetches, feed_dict=feed_dict)
                    cur_step += 1
                    train_score += _train_score
                    if cur_step % config.train_print_step == 0:
                        msg = 'the current step: {0}/{1}, train score: {2:>6.2%}'
                        print(msg.format(cur_step, each_epoch_step_sum, train_score / config.train_print_step))
                        train_score = 0
                val_score = self.evaluate(sess, x_val, y_val)
                if val_score >= best_val_score:
                    best_val_score = val_score
                    saver.save(sess=sess, save_path=model_config.model_save_path)
                    improved_str = '*'
                    last_improved_epoch = cur_epoch
                else:
                    improved_str = ''
                msg = 'the current epoch: {0}/{1}, val acc: {2:>6.2%}, cost: {3}s {4}'
                end_time = int(time.time())
                print(msg.format(cur_epoch + 1, config.epochs_num, val_score,
                                 end_time - start_time, improved_str))
                if cur_epoch - last_improved_epoch >= config.patience_epoch:
                    print("No optimization for a long time, auto stopping...")
                    break

    def evaluate(self, sess, x_val, y_val):
        data_len = len(y_val)
        total_acc = 0
        for i in range(0, data_len, config.batch_size):
            validation_fingerprints, validation_ground_truth = x_val[i:i + config.batch_size], \
                                                               y_val[i:i + config.batch_size]
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            _acc = sess.run(
                self.evaluation_step,
                feed_dict={
                    self.fingerprint_input: validation_fingerprints,
                    self.ground_truth_input: validation_ground_truth,
                    self.keep_prob: 1.0,
                })
            tmp_batch_size = min(config.batch_size, data_len - i)
            total_acc += _acc * tmp_batch_size
        return total_acc / data_len

    def predict(self, x_test):
        data_len = len(x_test)
        preds = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess=sess, save_path=model_config.model_save_path)  # 读取保存的模型
            # for i in range(num_batch):  # 逐批次处理
            #     start_id = i * config.batch_size
            #     end_id = min((i + 1) * config.batch_size, data_len)
            #     feed_dict = {
            #         self.fingerprint_input: x_test[start_id:end_id],
            #         self.keep_prob: 1.0
            #     }
            #     prob = sess.run(self.prob, feed_dict=feed_dict)
            #     preds.extend(np.argmax(prob))
            for i in range(data_len):  # 逐批次处理
                feed_dict = {
                    self.fingerprint_input: [x_test[i]],
                    self.keep_prob: 1.0
                }
                prob = sess.run(self.prob, feed_dict=feed_dict)
                preds.append(np.argmax(np.squeeze(prob, axis=0)))
        return preds
