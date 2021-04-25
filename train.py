import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import BPModel, FC_UNet, SimpleModel
from dataset import TrainDataset, ValidDataset, TestDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def train(seed=0):
    drop_list = ["V5", "V9", "V11", "V17", "V19", "V20", "V21", "V22", "V28"]
    data = pd.read_table('./zhengqi_train.txt', sep="\t")
    if len(drop_list):
        data.drop(drop_list, axis=1, inplace=True)
    train_data, valid_data = train_test_split(data, test_size=0.5, random_state=seed)
    train_dataset = TrainDataset(train_data, normalized=False)
    valid_dataset = ValidDataset(valid_data, normalized=False)
    train_reader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_reader = DataLoader(valid_dataset, batch_size=256, shuffle=True)
    model = SimpleModel(len(train_data.columns) - 1, 32, 2)
    # model.load_state_dict(torch.load('output/w64d4/bestmodel.pt'))
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.MSELoss()
    best_score = 1000000
    for epoch in range(101):
        model.train()
        for batch_ndx, sample in enumerate(train_reader):
            with torch.enable_grad():
                optimizer.zero_grad()
                x, y = sample
                x = x.float().cuda()
                y = y.float().reshape(y.shape[0], 1).cuda()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                print('epoch', epoch, 'batch', batch_ndx, 'loss', loss)
        
        model.eval()
        score = 0
        for batch_ndx, sample in enumerate(valid_reader):
            with torch.no_grad():
                x, y = sample
                x = x.float().cuda()
                y = y.float().reshape(y.shape[0], 1).cuda()
                out = model(x)
                score += criterion(out, y) * x.shape[0]
        score = score / len(valid_dataset)
        print('epoch', epoch, 'score', score)

        if score < best_score:
            best_score = score
            if score < 0.120:
                torch.save(model.state_dict(), 'output/{}.pt'.format(best_score))

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'output/{}.pt'.format(epoch))


if __name__ == '__main__':
    train(seed=0)