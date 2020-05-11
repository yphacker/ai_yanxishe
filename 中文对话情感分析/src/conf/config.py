# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")

train_path = os.path.join(data_path, 'training_set.csv')
val_path = os.path.join(data_path, 'validation_set.csv')
test_path = os.path.join(data_path, 'test_set.csv')
vocab_path = os.path.join(data_path, 'vocab.txt')

model_path = os.path.join(data_path, 'model')
model_save_path = model_path
cnn_model_save_path = os.path.join(model_path, 'cnn')
bert_model_save_path = os.path.join(model_path, 'bert')
# save_dir = os.path.join(checkpoints_path, 'textcnn')
# save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

# save_dir = os.path.join(config.checkpoints_path, 'rnn_attention')

bert_data_path = os.path.join(data_path, 'chinese_wwm_ext_L-12_H-768_A-12')
bert_config_path = os.path.join(bert_data_path, 'bert_config.json')
bert_checkpoint_path = os.path.join(bert_data_path, 'bert_model.ckpt')
bert_vocab_path = os.path.join(bert_data_path, 'vocab.txt')

tensorboard_path = os.path.join(data_path, 'tensorboard')

max_seq_length = 32  # 序列长度
epochs_num = 8
batch_size = 32
# save_per_batch = 10
print_per_batch = 10
improvement_step = print_per_batch * 10
