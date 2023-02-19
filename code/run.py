import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import dataloader, sampler
import torch.nn as nn
import torch.optim as optim
import os
import copy
import argparse

import datetime
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")


def get_model(args):
    mdl = args.model
    model = None
    if mdl == "FC2Layer":
        from models import FC2Layer
        # 两层神经网络的参数
        dim_in, din_hidden, dim_out = FC2Layer.get_param(args)

        # 构建模型
        model = FC2Layer(dim_in, din_hidden, dim_out)
    elif mdl == "Conv2Layer":
        from models import Conv2Layer
        # 卷积神经网络
        in_channels, dim_in, dim_out = Conv2Layer.get_params(args)

        # 构建模型
        model = Conv2Layer((in_channels, dim_in, dim_out))
    elif mdl == 'FinalNet':
        from models import FinalNet

        params = FinalNet.get_params(args)
        model = FinalNet(params)
    # TODO:oher models have not been designed
    else:
        pass

    return model


def get_data(args, train: bool = False):
    """
    :param args:args
    :param train :whether to load training set
    :return: the dataset that have been transformed to torch.Tensor and normalized,
        and turned to DataLoader that could directly be used for training or testing iteration
        @type train: bool
    """
    path = args.datasets_path
    ds = args.dataset
    batch_size = args.train_batch_size if train is True else args.test_batch_size
    if ds == 'MNIST':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307, ), (0.3081, ))
        ])

        data = datasets.MNIST(root=path, train=train, download=True, transform=transform)
    elif ds == 'FMNIST':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((73/255,), (90/255,))
        ])
        data = datasets.FashionMNIST(root=path, train=train, download=True, transform=transform)
    elif ds == "CIFAR10":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        data = datasets.CIFAR10(root=path, train=train, download=True, transform=transform)
    elif ds == 'CIFAR100':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        data = datasets.CIFAR100(root=path, train=train, download=True, transform=transform)
    # TODO:other dataset
    else:
        data = []
    if train:
        num_train = len(data)//10*9
        train_ldr = dataloader.DataLoader(data, batch_size=batch_size,
                                          sampler=sampler.SubsetRandomSampler(range(num_train)))
        valid_ldr = dataloader.DataLoader(data, batch_size=batch_size,
                                          sampler=sampler.SubsetRandomSampler(range(num_train, len(data))))
        return train_ldr, valid_ldr
    else:
        test_ldr = dataloader.DataLoader(data, batch_size=batch_size)
        return test_ldr, None


def evaluate(model, args, data=None):
    """
    evaluated the model on MNIST testing dataset and return the accuracy score
    :param model:the model should be evaluated
    :param args:args
    :param data:the dataset should be evaluated
    :return:
        acc:the accuracy_score of the model on MNIST test dataset
    """
    y_trues = []
    y_preds = []
    test_data = data if data is not None else (get_data(args)[0])
    for xx, yy in test_data:
        xx = xx.to(args.device)
        with torch.no_grad():
            score = model(xx)
            y_pred = np.argmax(score.cpu().numpy(), axis=1)
            y_preds.append(y_pred)
            y_trues.append(yy.cpu().numpy())
    y_trues = np.concatenate(y_trues, 0)
    y_preds = np.concatenate(y_preds, 0)
    # TODO:metrics , there is just accuracy by myself at present
    acc = accuracy_score(y_trues, y_preds)
    report = classification_report(y_trues, y_preds)
    return acc, report


def train(model, model_name, args):
    """
    train the model on inpued dataset by SGD optimizer
    :param model: the model that should be trained
    :param model_name: the file path of saved model
    :param args:args
    :return: None
    """
    # 优化函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_dlr, valid_dlr = get_data(args, train=True)

    ave_step = len(train_dlr) // 5

    # 最佳准确率，根据准确率选择最好的模型并保存,如果以保存有最好的模型，best_acc 就以它为准，否则置为-1
    if os.path.exists(model_name):
        old_model = copy.deepcopy(model)
        model_to_load = old_model.module if hasattr(old_model, 'module') else old_model
        model_to_load.load_state_dict(torch.load(model_name))
        best_acc = evaluate(old_model, args, valid_dlr)[0]
    else:
        best_acc = - 1.0
    # 每一个batch 的损失值
    losses = []
    for epoch in range(args.epoch):
        # 当前epoch的loss以及dataset循环次数
        for index, (x, y) in enumerate(train_dlr):   # mnist :x (batch_size,1,28,28) , y(batch_size,1,10)
            # 模型
            model.train()
            x = x.to(args.device)
            y = y.to(args.device)
            # 获得当前数据的预测结果分置
            score = model(x)  # (batch_size,10)

            # 通过模型的forward函数得到的各个标签的概率的分值计算损失，损失函数使用交叉熵
            loss = nn.functional.cross_entropy(score, y)

            # 下一步的计算，也就反向传播时不需要存储中间值
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 累计损失值及次数
            losses.append(loss.item())

            # 更新模型的参数
            optimizer.step()

            # 每循环一千次，做一次evaluate，并保存效果最优的模型
            if index % 100 == 0:
                print('Epoch %d step %d loss %.4f' % (epoch, index, np.mean(losses[-100:])))
            if len(losses) % ave_step == 0:
                # 在验证集上测试准确率
                acc, report = evaluate(model, args, valid_dlr)
                print('current accuracy = %.2f%% ' % (acc*100.0))
                print("------------------------\n")

                # 如果当前模型的准确率比之前存在的模型的准确率要高，那么保存当前的模型
                if acc > best_acc:
                    best_acc = acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), model_name)

    acc, report = evaluate(model, args, valid_dlr)
    return acc, report


def main():
    parser = argparse.ArgumentParser()

    # 是否训练模型&验证模型
    parser.add_argument("--do_train", action='store_true',
                        help="whether train the model on training set")
    parser.add_argument("--do_valid",action='store_true',
                        help='whether valid the best saved model on validation set')
    parser.add_argument('--do_test', action="store_true",
                        help='whether test the best saved model on testing set')

    # 模型和数据地址
    parser.add_argument("--model", type=str, default="FC2Layer",
                        help="choose the model that used to train/valid/test , default is FC2Layer")
    parser.add_argument('--model_path', type=str, default='../saved_model/',
                        help='directory of saved model')
    parser.add_argument('--datasets_path', type=str, default='../datasets/',
                        help='path of datasets in the project')

    # TODO:not supported the dataset like MNIST,FMNIST,CIFAR10,CIFAR100 at present
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='which data that used to training or test the model,default \
                        is MNIST , also could be FMNIST,CIFAR10,CIFAR100')

    # 训练和测试的batch_size
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='testing batch size')
    # 训练epoch
    parser.add_argument('--epoch', type=int, default=1,
                        help='train epoch')
    # 是否使用GPU
    parser.add_argument('--with_cuda', action='store_true',
                        help='whether use GPU')
    # 学习率
    parser.add_argument('--lr', type=float, default=0.9,
                        help='learning rate of optimizer')

    parser.add_argument('--log_path', type=str, default='../log/',
                        help='the logging file path')

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    now_date = datetime.datetime.today().strftime('%m-%d-%H-%M-%S')
    log_file = os.path.join(args.log_path, args.model+'_'+args.dataset+'_'+now_date+".txt")
    log = open(log_file, 'w+')



    # 使用CPU还是GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.with_cuda else 'cpu')
    print("** devise is %s **" % device)
    args.device = device

    # 存储模型的目录
    model_saved_path = os.path.join(args.model_path, args.model)
    if not os.path.exists(model_saved_path):
        os.mkdir(model_saved_path)


    # 存储模型的文件名
    model_name = os.path.join(model_saved_path, args.dataset + '.bin')

    # 获取模型
    model = get_model(args)
    model = model.to(args.device)

    # 将参数输出到日志文件
    print('** args :', file=log)
    for item, value in vars(args).items():
        print(item, " : ", value, file=log)
    print("** args \n\n", file=log)

    # 输出模型结构
    print(model)
    print(model, end='\n\n', file=log)

    # exit(0)

    st = datetime.datetime.now()

    # 训练模型
    if args.do_train:
        # train the model
        print("** begin trainning **")
        acc, report = train(model, model_name, args)
        print("** training is finished **")
        print("** the classification report of current trained model on the validation dataset is :")
        print(report)
        print("** the accuracy score of current trained model on the validation dataset is : %.2f%%" % (acc*100))

        # log to file
        print("** training is finished **", file=log)
        print("** the classification report of current trained model on the validation dataset is :", file=log)
        print(report, file=log)
        print("** the accuracy score of current trained model on the validation dataset is : %.2f%%\n" % (acc * 100), file=log)

    # 验证模型
    if args.do_valid:
        # test the model on the validation set
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(model_name))
        model.eval()
        __, valid_dlr = get_data(args, True)
        print("** begin validation **")
        acc, report = evaluate(model, args, valid_dlr)
        print("** the classification report of best saved model on the validation dataset is :")
        print(report)
        print("** the accuracy score of best saved model  on the validation dataset is : %.2f%%" % (acc*100))

        # log to file
        print("** the classification report of best saved model on the validation dataset is :", file=log)
        print(report, file=log)
        print("** the accuracy score of best saved model  on the validation dataset is : %.2f%%\n" % (acc * 100), file=log)

    # 测试模型
    if args.do_test:
        # test the saved best model
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(model_name))
        model.eval()
        print("** begin testing **")
        acc, resport = evaluate(model, args)
        print("** the classification report of best saved model on the testing dataset is :")
        print(report)
        print("** the accuracy score of best saved model  on the testing dataset is : %.2f%%" % (acc*100))

        # log to file
        print("** the classification report of best saved model on the testing dataset is :", file=log)
        print(report, file=log)
        print("** the accuracy score of best saved model  on the testing dataset is : %.2f%%\n" % (acc * 100), file=log)

    ed = datetime.datetime.now()
    print('spent time is {}'.format(ed - st))
    print('spent time is {}'.format(ed - st), file=log)
    log.close()


if __name__ == '__main__':
    main()
