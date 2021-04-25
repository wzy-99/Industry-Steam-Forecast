import torch
import pandas as pd
import numpy as np
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from model import BPModel, SimpleModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def test():
    drop_list = ["V5", "V9", "V11", "V17", "V19", "V20", "V21", "V22", "V28"]

    # data = pd.read_table('./zhengqi_train.txt', sep="\t")
    # if len(drop_list):
    #     data.drop(drop_list, axis=1, inplace=True)
    # train_data, valid_data = train_test_split(data, test_size=0.2, random_state=0)
    # train_dataset = TrainDataset(train_data, normalized=False)

    test_data = pd.read_table('./zhengqi_test.txt', sep="\t")
    if len(drop_list):
        test_data.drop(drop_list, axis=1, inplace=True)
    test_dataset = TestDataset(test_data, normalized=False)
    test_reader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = SimpleModel(len(test_data.columns), 32, 2)
    model.load_state_dict(torch.load('output/i29w32d2s1/bestmodel.pt'))
    model.eval()
    
    result = []
    for batch_ndx, sample in tqdm(enumerate(test_reader)):
        with torch.no_grad():
            x = sample
            x = x.float()
            out = model(x)
            result.append(out)
    
    with open('result.txt', 'w') as f:
        for out in result:
            f.write(str(out.numpy()[0][0]) + '\n')


if __name__ == '__main__':
    test()