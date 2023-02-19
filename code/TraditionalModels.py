from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from torchvision import datasets
import os
import argparse
import numpy as np

import datetime


def get_model(args):
    mdl = args.model
    model = None
    if mdl == "SVM":
        # 构建模型
        if args.max_iter is not None:
            model = LinearSVC(max_iter=args.max_iter)
        else:
            model = LinearSVC()
    elif mdl == "DecisionTree":
        # 构建模型
        model = DecisionTreeClassifier()
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
    data = None
    if ds == 'MNIST':
        data = datasets.MNIST(root=path, train=train, download=True)
    elif ds == 'FMNIST':
        data = datasets.FashionMNIST(root=path, train=train, download=True)
    elif ds == "CIFAR10":
        data = datasets.CIFAR10(root=path, train=train, download=True)
    elif ds == 'CIFAR100':
        data = datasets.CIFAR100(root=path, train=train, download=True)
    # TODO:other dataset
    else:
        pass
    X = np.array(data.data.reshape(len(data), -1)/255.0)
    y = np.array(data.targets)
    return X, y


def main():
    parser = argparse.ArgumentParser()

    # 模型和数据地址
    parser.add_argument("--model", type=str, default="SVM",
                        help="choose the model that used to train/test , default is SVM")
    parser.add_argument('--datasets_path', type=str, default='../datasets/',
                        help='path of datasets in the project')

    # TODO:not supported the dataset like MNIST,FMNIST,CIFAR10,CIFAR100 at present
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='which data that used to training or test the model,default \
                        is MNIST , also could be FMNIST,CIFAR10,CIFAR100')

    parser.add_argument('--log_path', type=str, default='../log/',
                        help='the logging file path')

    parser.add_argument('--max_iter', type=int,
                        help='The max_iter param of SVM')

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    now_date = datetime.datetime.today().strftime('%m-%d-%H-%M-%S')
    log_file = os.path.join(args.log_path, args.model+'_'+args.dataset+'_'+now_date+".txt")
    log = open(log_file, 'w+')

    # 将参数输出到日志文件
    print('** args :', file=log)
    for item, value in vars(args).items():
        print(item, " : ", value, file=log)
    print("** args \n\n", file=log)

    # 获取模型
    model = get_model(args)

    # 输出模型结构
    print(model)
    print(model, end='\n\n', file=log)

    st = datetime.datetime.now()

    # 读取训练数据集
    X_train, y_train = get_data(args, True)
    print("** {} training data has been read **".format(args.dataset))
    print("** {} training data has been read **".format(args.dataset), file=log)

    # 训练分类器
    model.fit(X_train, y_train)
    print("** {} Model has been fitted to training data **")
    print("** {} Model has been fitted to training data **".format(args.model), file=log)

    # 读取测试数据集
    X_test, y_test = get_data(args)
    print("** {} testing data has been read **".format(args.dataset))
    print("** {} testing data has been read **".format(args.dataset), file=log)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    print("** {} Model has predicted on training data **".format(args.model), file=log)

    # 计算分类器的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print('** predicted Accuracy: %.2f%% ' % (accuracy*100))
    print('** predicted Accuracy: %.2f%% ' % (accuracy*100),file=log)

    ed = datetime.datetime.now()
    print('spent time is {}'.format(ed - st))
    print('spent time is {}'.format(ed - st), file=log)
    log.close()


if __name__ == "__main__":
    main()
