# coding=utf-8
# author=yphacker


import os
import argparse
import pandas as pd
import torch
# from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn
from conf import config
from utils.data_utils import MyDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_iter):
    model.eval()
    data_len = 0
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_iter:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()
            _, preds = torch.max(probs, 1)
            total_acc += (preds == batch_y).sum().item()

    return total_loss / data_len, total_acc / data_len


def train():
    print('train:{}, val:{}'.format(len(os.listdir(config.train_path)), len(os.listdir(config.val_path))))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([config.image_size, config.image_size]),  # 注意 Resize 参数是 2 维，和 RandomResizedCrop 不同
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    # train_dataset = ImageFolder(config.process_train_path, transform=transform_train)
    train_dataset = MyDataset(config.train_path, transform_train, 'train')
    train_iter = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    transform_val = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    # val_dataset = ImageFolder(config.process_val_path, transform=transform_val)
    val_dataset = MyDataset(config.val_path, transform_val, 'val')
    val_iter = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)

    flag = False
    best_val_acc = 0
    cur_step = 1
    last_improved_step = 0
    total_step = len(train_iter) * config.epochs_num
    for epoch in range(config.epochs_num):
        for batch_x, batch_y in train_iter:
            # scheduler.step()
            model.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)

            train_loss = criterion(probs, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.print_per_batch == 0:
                _, train_preds = torch.max(probs, 1)
                train_corrects = (train_preds == batch_y).sum().item()
                train_acc = train_corrects * 1.0 / len(batch_y)
                val_loss, val_acc = evaluate(model, val_iter)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), config.model_save_path)
                    improved_str = '*'
                    last_improved_step = cur_step
                else:
                    improved_str = ''
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%},  ' \
                      'val loss: {4:>5.2}, val acc: {5:>6.2%}, {6}'
                print(msg.format(cur_step, total_step, train_loss.item(), train_acc, val_loss, val_acc, improved_str))
            if cur_step - last_improved_step > len(train_iter):
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def predict():
    model.load_state_dict(torch.load(config.model_save_path))
    transform_test = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    train_dataset = MyDataset(config.test_path, transform_test, 'test')
    test_iter = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch_x, _ in test_iter:
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            # pred = torch.argmax(output, dim=1)
            _, preds = torch.max(probs, 1)
            preds_list += [p.item() for p in preds]

    submission = pd.DataFrame({"id": range(len(preds_list)), "label": preds_list})
    submission.to_csv('../data/densenet18_submission.csv', index=False, header=False)


def main(op):
    if op == 'train':
        train()
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--MODEL", default='cnn', type=str, help="model select")
    args = parser.parse_args()
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS

    # 模型
    transfer_model = models.resnet18(pretrained=True)
    for param in transfer_model.parameters():
        param.requires_grad = False

    # 修改最后一层维数，即 把原来的全连接层 替换成 输出维数为2的全连接层
    num_ftrs = transfer_model.fc.in_features
    for param in transfer_model.parameters():
        param.requires_grad = False
    # transfer_model.fc = nn.Linear(num_ftrs, 2)
    transfer_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )
    # print(transfer_model)

    model = transfer_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    main(args.operation)
