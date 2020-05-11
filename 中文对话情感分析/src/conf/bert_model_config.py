# coding=utf-8
# author=yphacker


train_batch_size = 32
val_batch_size = 32
test_batch_size = 32
num_labels = 2  # 类别数量
iter_num = 8
learning_rate = 1e-5
# if max_seq_length > bert_config.max_position_embeddings:  # 模型有个最大的输入长度 512
#     raise ValueError("超出模型最大长度")
grad_clip = 5.0
