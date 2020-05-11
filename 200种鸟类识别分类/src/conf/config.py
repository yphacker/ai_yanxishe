# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")

train_path = os.path.join(data_path, 'train_set')
val_path = os.path.join(data_path, 'val_set')
test_path = os.path.join(data_path, 'test_set')

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, required=True)
# parser.add_argument('--base_lr', type=float, default=0.001)
# parser.add_argument('--batch_size', type=int, required=True)
# parser.add_argument('--epoch', type=int, default=100)
# parser.add_argument('--drop_rate', type=float, default=0.25)
# parser.add_argument('--T_k', type=int, default=10,
#                     help='how many epochs for linear drop rate, can be 5, 10, 15. ')
# parser.add_argument('--weight_decay', type=float, default=1e-8)
# parser.add_argument('--step', type=int, default=None)
# parser.add_argument('--resume', action='store_true')
# parser.add_argument('--net1', type=str, default='bcnn',
#                     help='specify the network architecture, available options include bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')
# parser.add_argument('--net2', type=str, default='bcnn',
#                     help='specify the network architecture, available options include bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')
#
#
# data_dir = args.dataset
# learning_rate = args.base_lr
# batch_size = args.batch_size
# num_epochs = args.epoch
# drop_rate = args.drop_rate
# T_k = args.T_k
# weight_decay = args.weight_decay
#
# if args.net1 == 'bcnn':
#     NET1 = BCNN
# elif args.net1 == 'vgg16':
#     NET1 = VGG16
# elif args.net1 == 'vgg19':
#     NET1 = VGG19
# elif args.net1 == 'resnet18':
#     NET1 = ResNet18
# elif args.net1 == 'resnet34':
#     NET1 = ResNet34
# elif args.net1 == 'resnet50':
#     NET1 = ResNet50
# else:
#     raise AssertionError('net should be in bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')
#
# if args.net2 == 'bcnn':
#     NET2 = BCNN
# elif args.net2 == 'vgg16':
#     NET2 = VGG16
# elif args.net2 == 'vgg19':
#     NET2 = VGG19
# elif args.net2 == 'resnet18':
#     NET2 = ResNet18
# elif args.net2 == 'resnet34':
#     NET2 = ResNet34
# elif args.net2 == 'resnet50':
#     NET2 = ResNet50
# else:
#     raise AssertionError('net should be in bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')
#
#
# epoch_decay_start = 40
# warmup_epochs = 5
#
# os.popen('mkdir -p model')
#
#
# resume = args.resume
#
#
#
# logfile = 'logfile_' + data_dir + '_peerlearning_' + str(drop_rate) + '.txt'
#
#
# # Adjust learning rate and betas for Adam Optimizer
# mom1 = 0.9
# mom2 = 0.1
# alpha_plan = lr_scheduler(learning_rate, num_epochs, warmup_end_epoch=warmup_epochs, mode='cosine')
# beta1_plan = [mom1] * num_epochs
# for i in range(epoch_decay_start, num_epochs):
#     beta1_plan[i] = mom2