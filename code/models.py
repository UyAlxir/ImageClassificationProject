import torch.nn as nn


class FC2Layer(nn.Module):
    """
        the model created by myself that used to image classification
    """

    def __init__(self, dim_in=28*28, dim_hidden=100, dim_out=10):
        """
        Inputs:
            dim_in,dim_hidden,dim_out
        """
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        # nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        # nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

    @staticmethod
    def get_param(args):
        ds = args.dataset
        if ds == 'MNIST' or ds == 'FMNIST':
            dim_in, dim_hidden, dim_out = 28 * 28, 4096, 10
            return dim_in, dim_hidden, dim_out
        elif ds == 'CIFAR10' :
            dim_in, dim_hidden, dim_out = 32 * 32 * 3, 4096, 10
        elif ds == 'CIFAR100':
            dim_in, dim_hidden, dim_out = 32 * 32 * 3, 4096, 100
        # TODO:other dataset
        else:
            dim_in, dim_hidden, dim_out = 0, 0, 0
        return dim_in, dim_hidden, dim_out


# TODO: more models should be writed here
class Conv2Layer(nn.Module):
    """
        2 Layer Convelution Neural Network
    """
    def __init__(self, params):
        super().__init__()
        in_channels, dim_in, dim_out = params
        chnl_1 = in_channels
        chnl_2 = chnl_1 * 4
        chnl_3 = chnl_2 * 4
        self.conv1 = nn.Conv2d(in_channels=chnl_1, out_channels=chnl_2, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=chnl_2, out_channels=chnl_3, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # TODO:forward
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    @staticmethod
    def get_params(args):
        ds = args.dataset
        if ds == "MNIST" or ds == "FMNIST":
            in_channels = 1
            dim_in = 28*28
            dim_out = 10
            return in_channels, dim_in, dim_out
        elif ds == "CIFAR10":
            in_channels = 3
            dim_in = 32 * 32 * 3
            dim_out = 10
            return in_channels, dim_in, dim_out
        elif ds == "CIFAR100":
            in_channels = 3
            dim_in = 32 * 32 * 3
            dim_out = 100
            return in_channels, dim_in, dim_out

class FinalNet(nn.Module):
    """
        2 Layer Convelution Neural Network
    """
    def __init__(self, params):
        super().__init__()
        in_channel, padding, dim_out = params
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=(3, 3), stride=1, padding=padding),  # 64*30*30
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 64*15*15
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding=1), # 128*13*13
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(7, 7), stride=1, padding=1), # 256*8*8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 256*4*4
        )
        self.Flatten = Flatten()
        self.Linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=dim_out),
        )

    def forward(self, x):
        # TODO:forward
        x = self.convs(x)
        x = self.Flatten(x)
        x = self.Linear(x)
        return x

    @staticmethod
    def get_params(args):
        ds = args.dataset
        if ds == "MNIST" or ds == "FMNIST":
            in_channels = 1
            padding = 2
            dim_out = 10
            return in_channels, padding, dim_out
        elif ds == "CIFAR10":
            in_channels = 3
            padding = 0
            dim_out = 10
            return in_channels, padding, dim_out
        elif ds == "CIFAR100":
            in_channels = 3
            padding = 0
            dim_out = 100
            return in_channels, padding, dim_out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
