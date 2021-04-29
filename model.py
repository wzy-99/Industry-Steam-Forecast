import torch
import torch.nn as nn
from torchsummary import summary


class LinearBN(nn.Module):
    def __init__(self, in_dim, out_dim, act='relu'):
        super(LinearBN, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.bn = nn.BatchNorm1d(num_features=out_dim)
        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = None

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class BPModel(nn.Module):
    def __init__(self, in_dim=38, base_channel=64, depth=8, act='relu'):
        super(BPModel, self).__init__()
        
        self.linear = LinearBN(in_dim, base_channel, act)

        hidden = []
        for i in range(0, depth // 2):
            hidden.append(LinearBN(base_channel * 2 ** i, base_channel * 2 ** (i + 1), act))
            hidden.append(nn.Dropout(p=0.2))
        for i in range(depth // 2, 0, -1):
            hidden.append(LinearBN(base_channel * 2 ** i, base_channel * 2 ** (i - 1), act))
            hidden.append(nn.Dropout(p=0.2))
        self.hidden = nn.Sequential(*hidden)

        self.out = LinearBN(base_channel, 1, None)

    def forward(self, x):
        if self.training:
            x = self.linear(x)
            x = self.hidden(x)
            x = self.out(x)
        else:
            x = self.linear(x)
            x = self.hidden(x)
            x = self.out(x)
        return x


class FC_UNet(nn.Module):
    def __init__(self, in_dim=38, base_channel=64):
        super(FC_UNet, self).__init__()
        
        self.linear = LinearBN(in_dim, base_channel, 'relu')

        self.layer1 = LinearBN(base_channel * 2 ** 0, base_channel * 2 ** 1, 'relu')
        self.drop1 = nn.Dropout(p=0.2)
        self.layer2 = LinearBN(base_channel * 2 ** 1, base_channel * 2 ** 2, 'relu')
        self.drop2 = nn.Dropout(p=0.2)
        self.layer3 = LinearBN(base_channel * 2 ** 2, base_channel * 2 ** 3, 'relu')
        self.drop3 = nn.Dropout(p=0.2)
        self.layer4 = LinearBN(base_channel * 2 ** 3, base_channel * 2 ** 4, 'relu')
        self.drop4 = nn.Dropout(p=0.2)
        self.layer5 = LinearBN(base_channel * 2 ** 4, base_channel * 2 ** 3, 'relu')
        self.drop5 = nn.Dropout(p=0.2)
        self.layer6 = LinearBN(base_channel * 2 ** 3, base_channel * 2 ** 2, 'relu')
        self.drop6 = nn.Dropout(p=0.2)
        self.layer7 = LinearBN(base_channel * 2 ** 2, base_channel * 2 ** 1, 'relu')
        self.drop7 = nn.Dropout(p=0.2)
        self.layer8 = LinearBN(base_channel * 2 ** 1, base_channel * 2 ** 0, 'relu')
        self.drop8 = nn.Dropout(p=0.2)

        self.out = LinearBN(base_channel, 1, None)

    def forward(self, x):
        if self.training:
            x1 = self.linear(x)   # 64
            x2 = self.layer1(x1)  # 128
            x3 = self.layer2(x2)  # 256
            x4 = self.layer3(x3)  # 512
            x = self.layer4(x4)   # 1024
            x = self.layer5(x) + x4  # 512
            x = self.layer6(x) + x3   
            x = self.layer7(x) + x2
            x = self.layer8(x) + x1
            x = self.out(x)
        else:
            x1 = self.linear(x)   # 64
            x2 = self.layer1(x1)  # 128
            x3 = self.layer2(x2)  # 256
            x4 = self.layer3(x3)  # 512
            x5 = self.layer4(x4)  # 1024
            x = self.layer5(x4) + x4  # 512
            x = self.layer6(x) + x3   # 256
            x = self.layer7(x) + x2   # 128
            x = self.layer8(x) + x1   # 64
            x = self.out(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self, in_dim=38, base_channel=64, depth=8, act='relu'):
        super(SimpleModel, self).__init__()
        
        self.linear = LinearBN(in_dim, base_channel, act)

        hidden = []
        hidden.append(LinearBN(base_channel, base_channel // 2, act))
        hidden.append(nn.Dropout(p=0.2))
        hidden.append(LinearBN(base_channel // 2, base_channel // 4, act))
        hidden.append(nn.Dropout(p=0.2))
        self.hidden = nn.Sequential(*hidden)

        self.out = LinearBN(base_channel // 4, 1, None)

    def forward(self, x):
        if self.training:
            x = self.linear(x)
            x = self.hidden(x)
            x = self.out(x)
        else:
            x = self.linear(x)
            x = self.hidden(x)
            x = self.out(x)
        return x


class FitModel(nn.Module):
    def __init__(self, in_dim):
        super(FitModel, self).__init__()

        self.layer1 = LinearBN(in_dim, 8, act='tanh')
        self.layer2 = LinearBN(8, 1, act=None)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, in_dim):
        super(LinearModel, self).__init__()

        self.layer = LinearBN(in_dim, 1, None)

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == '__main__':
    # model = FitModel(29)
    # pt = torch.load('output/i29w8d2n2/bestmodel.pt')
    # for p in pt:
    #     print(p, pt[p])
    summary(FitModel(29), (29,))