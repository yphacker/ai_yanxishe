import os
import math
import copy
import shutil
import time
import random
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict, namedtuple
from sklearn.metrics import roc_auc_score, average_precision_score
import se_resnext101_32x4d
from efficientnet_pytorch import EfficientNet
from data_augmentation import FixedRotation
from inceptionv4 import inceptionv4
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import resnet101, resnet50, resnet152
from torchvision.models import densenet121
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


def main(index):
    np.random.seed(359)
    torch.manual_seed(359)
    torch.cuda.manual_seed_all(359)
    random.seed(359)

    batch_size = 8
    workers = 16

    # stage_epochs = [8, 8, 8, 6, 5, 4, 3, 2]
    # stage_epochs = [12, 6, 5, 3, 4]
    lr = 5e-5
    lr_decay = 10
    weight_decay = 1e-4

    stage = 0
    start_epoch = 0
    # total_epochs = sum(stage_epochs)
    # total_epochs = 100
    total_epochs = 16
    patience = 4
    no_improved_times = 0
    total_stages = 3
    best_score = 0
    samples_num = 54

    print_freq = 20
    train_ratio = 0.9  # others for validation
    momentum = 0.9
    pre_model = 'senet'
    pre_trained = True
    evaluate = False
    use_pre_model = False
    # file_name = os.path.basename(__file__).split('.')[0]

    file_name = "efficientnetb7_500_all_{}".format(index)
    img_size = 500

    resumeflg = False
    resume = ''

    # 创建保存模型和结果的文件夹
    if not os.path.exists('./model/%s' % file_name):
        os.makedirs('./model/%s' % file_name)
    if not os.path.exists('./result/%s' % file_name):
        os.makedirs('./result/%s' % file_name)

    if not os.path.exists('./result/%s.txt' % file_name):
        txt_mode = 'w'
    else:
        txt_mode = 'a'
    with open('./result/%s.txt' % file_name, txt_mode) as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

    # build a model
    # model =resnet50(pretrained=True)
    # model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    # model.fc = torch.nn.Linear(model.fc.in_features,3)
    model = se_resnext101_32x4d.se_resnet152(num_classes=3)
    # model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=3)
    # model = inceptionv4(pretrained='imagenet')
    # model.last_linear  = torch.nn.Linear(model.last_linear.in_features,2)
    model = torch.nn.DataParallel(model).cuda()

    def load_pre_cloth_model_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if 'fc' in name:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    if use_pre_model:
        print('using pre model')
        pre_model_path = ''
        load_pre_cloth_model_dict(model, torch.load(pre_model_path)['state_dict'])

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            stage = checkpoint['stage']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            no_improved_times = checkpoint['no_improved_times']
            if no_improved_times == 0:
                model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    def default_loader(root_dir, path):
        final_path = os.path.join(root_dir, str(path) + ".jpg")
        return Image.open(final_path).convert('RGB')
        # return Image.open(path)

    class TrainDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['filename'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            label = label
            img = self.loader('../data/train', filename)

            if self.transform is not None:
                img = self.transform(img)

            return img, label

        def __len__(self):
            return len(self.imgs)

    class ValDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['filename'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            label = label
            img = self.loader('../data/train', filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label, filename

        def __len__(self):
            return len(self.imgs)

    class TestDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['filename'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader('../data/test', filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, filename

        def __len__(self):
            return len(self.imgs)

    train_data_list = pd.read_csv("data/train_{}.csv".format(index), sep=",")
    val_data_list = pd.read_csv("data/test_{}.csv".format(index), sep=",")
    test_data_list = pd.read_csv("../test.csv", sep=",")

    train_data_list = train_data_list.fillna(0)

    # 训练集正常样本尺寸
    random_crop = [transforms.RandomCrop(640), transforms.RandomCrop(768), transforms.RandomCrop(896)]

    smax = nn.Softmax()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_data = TrainDataset(train_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((img_size, img_size)),
                                  transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
                                  # transforms.RandomRotation(30),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomGrayscale(),
                                  FixedRotation([0, 90, 180, -90]),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

    val_data = ValDataset(val_data_list,
                          transform=transforms.Compose([
                              transforms.Resize((img_size, img_size)),
                              # transforms.CenterCrop((500, 500)),
                              transforms.ToTensor(),
                              normalize,
                          ]))

    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                # transforms.CenterCrop((500, 500)),
                                transforms.ToTensor(),
                                normalize,
                                # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                # transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
                            ]))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size * 2, shuffle=False, pin_memory=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size * 4, shuffle=False, pin_memory=False, num_workers=workers)

    test_data_hflip = TestDataset(test_data_list,
                                  transform=transforms.Compose([
                                      transforms.Resize((img_size, img_size)),
                                      transforms.RandomHorizontalFlip(p=2),
                                      # transforms.CenterCrop((500, 500)),
                                      transforms.ToTensor(),
                                      normalize,
                                  ]))

    test_loader_hflip = DataLoader(test_data_hflip, batch_size=batch_size * 4, shuffle=False, pin_memory=False,
                                   num_workers=workers)

    test_data_vflip = TestDataset(test_data_list,
                                  transform=transforms.Compose([
                                      transforms.Resize((img_size, img_size)),
                                      transforms.RandomVerticalFlip(p=2),
                                      # transforms.CenterCrop((500, 500)),
                                      transforms.ToTensor(),
                                      normalize,
                                  ]))

    test_loader_vflip = DataLoader(test_data_vflip, batch_size=batch_size * 4, shuffle=False, pin_memory=False,
                                   num_workers=workers)

    test_data_vhflip = TestDataset(test_data_list,
                                   transform=transforms.Compose([
                                       transforms.Resize((img_size, img_size)),
                                       transforms.RandomHorizontalFlip(p=2),
                                       transforms.RandomVerticalFlip(p=2),
                                       # transforms.CenterCrop((500, 500)),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    test_loader_vhflip = DataLoader(test_data_vhflip, batch_size=batch_size * 4, shuffle=False, pin_memory=False,
                                    num_workers=workers)

    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading
            # if len(target) % workers == 1:
            #     images = images[:-1]
            #     target = target[:-1]

            data_time.update(time.time() - end)
            image_var = torch.tensor(images, requires_grad=False).cuda(async=True)
            # print(image_var)
            label = torch.tensor(target).cuda(async=True)
            # compute y_pred
            y_pred = model(image_var)
            loss = criterion(y_pred, label)

            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        # losses = AverageMeter()
        # acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        # 保存概率，用于评测
        val_imgs, val_preds, val_labels, = [], [], []

        end = time.time()
        for i, (images, labels, img_path) in enumerate(val_loader):
            # if len(labels) % workers == 1:
            #     images = images[:-1]
            #     labels = labels[:-1]
            image_var = torch.tensor(images, requires_grad=False).cuda(async=True)  # for pytorch 0.4
            # label_var = torch.tensor(labels, requires_grad=False).cuda(async=True)  # for pytorch 0.4
            target = torch.tensor(labels).cuda(async=True)

            # compute y_pred
            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)

            # measure accuracy and record loss
            # prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
            # losses.update(loss.item(), images.size(0))
            # acc.update(prec, PRED_COUNT)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % (print_freq * 5) == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader),
                                                                                  batch_time=batch_time))

            # 保存概率，用于评测
            smax_out = smax(y_pred)
            val_imgs.extend(img_path)
            val_preds.extend([i.tolist() for i in smax_out])
            val_labels.extend([i.item() for i in labels])
        val_preds = [';'.join([str(j) for j in i]) for i in val_preds]
        val_score = pd.DataFrame({'img_path': val_imgs, 'preds': val_preds, 'label': val_labels, })
        val_score.to_csv('./result/%s/val_score.csv' % file_name, index=False)
        acc, f1 = score(val_score)
        print('acc: %.4f, f1: %.4f' % (acc, f1))
        print(' * Score {final_score:.4f}'.format(final_score=f1), '(Previous Best Score: %.4f)' % best_score)
        return acc, f1

    def test(test_loader, model):
        csv_map = OrderedDict({'FileName': [], 'type': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            # bs, ncrops, c, h, w = images.size()

            filepath = [str(i.numpy()).split('/')[-1] + ".jpg" for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

            with torch.no_grad():
                y_pred = model(image_var)  # fuse batch size and ncrops
                # y_pred = y_pred.view(bs, ncrops, -1).mean(1) # avg over crops

                # get the index of the max log-probability
                smax = nn.Softmax()
                smax_out = smax(y_pred)
            csv_map['FileName'].extend(filepath)
            for output in smax_out:
                prob = ';'.join([str(i) for i in output.data.tolist()])
                csv_map['probability'].append(prob)
                csv_map['type'].append(np.argmax(output.data.tolist()))
            # print(len(csv_map['filename']), len(csv_map['probability']))

        result = pd.DataFrame(csv_map)
        result.to_csv('./result/%s/submission.csv' % file_name, index=False)
        result[['FileName', 'type']].to_csv('./result/%s/final_submission.csv' % file_name, index=False)
        return

    def save_checkpoint(state, is_best, filename='./model/%s/checkpoint.pth.tar' % file_name):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, './model/%s/model_best.pth.tar' % file_name)

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0

        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0)

        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def doitf(tp, fp, fn):
        if (tp + fp == 0):
            return 0
        if (tp + fn == 0):
            return 0
        pre = float(1.0 * float(tp) / float(tp + fp))
        rec = float(1.0 * float(tp) / float(tp + fn))
        if (pre + rec == 0):
            return 0
        return (2 * pre * rec) / (pre + rec)

    # 参数 samples_num 表示选取多少个样本来取平均
    def score(val_score):
        val_score['preds'] = val_score['preds'].map(lambda x: [float(i) for i in x.split(';')])
        tp = np.zeros(9)
        fp = np.zeros(9)
        fn = np.zeros(9)
        f1 = np.zeros(9)
        f1_tot = 0
        acc = 0

        print(val_score.head(10))

        for img in val_score['img_path'].unique():
            img_scores = val_score[val_score['img_path'] == img]
            probs = np.array(img_scores['preds'].values.tolist())
            label = np.array(img_scores['label'].values.tolist())
            if (np.argmax(probs[0]) == label[0]):
                acc = acc + 1
                tp[label[0]] += 1
            else:
                fp[np.argmax(probs[0])] += 1
                fn[label[0]] += 1

        for classes in range(2):
            f1[classes] = doitf(tp[classes], fp[classes], fn[classes])
            f1_tot = f1_tot + f1[classes]
        acc = acc / val_score.shape[0]
        f1_tot = f1_tot / 2

        return acc, f1_tot

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = optim.Adam(model.module.last_linear.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    if evaluate:
        validate(val_loader, model, criterion)
    else:
        for epoch in range(start_epoch, total_epochs):
            if stage >= total_stages - 1:
                break
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            if epoch >= 0:
                acc, f1 = validate(val_loader, model, criterion)

                with open('./result/%s.txt' % file_name, 'a') as acc_file:
                    acc_file.write('Epoch: %2d, acc: %.8f, f1: %.8f\n' % (epoch, acc, f1))

                # remember best Accuracy and save checkpoint
                is_best = acc > best_score
                best_score = max(acc, best_score)

                # if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                #     stage += 1
                #     optimizer = adjust_learning_rate()

                if is_best:
                    no_improved_times = 0
                else:
                    no_improved_times += 1

                print('stage: %d, no_improved_times: %d' % (stage, no_improved_times))

                if no_improved_times >= patience:
                    stage += 1
                    optimizer = adjust_learning_rate()

                state = {
                    'epoch': epoch + 1,
                    'arch': pre_model,
                    'state_dict': model.state_dict(),
                    'best_score': best_score,
                    'no_improved_times': no_improved_times,
                    'stage': stage,
                    'lr': lr,
                }
                save_checkpoint(state, is_best)

                # if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                if no_improved_times >= patience:
                    no_improved_times = 0
                    model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
                    print('Step into next stage')
                    with open('./result/%s.txt' % file_name, 'a') as acc_file:
                        acc_file.write('---------------------Step into next stage---------------------\n')

    with open('./result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('* best acc: %.8f  %s\n' % (best_score, os.path.basename(__file__)))
    with open('./result/best_acc.txt', 'a') as acc_file:
        acc_file.write('%s  * best acc: %.8f  %s\n' % (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_score, os.path.basename(__file__)))

    # test
    best_model = torch.load('model/{}/model_best.pth.tar'.format(file_name))
    model.load_state_dict(best_model['state_dict'])
    test(test_loader=test_loader, model=model)

    torch.cuda.empty_cache()
    # resume = False


if __name__ == '__main__':
    for index in range(1, 6):
        main(index)
