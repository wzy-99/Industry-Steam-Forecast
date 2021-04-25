import torch

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, normalized=True):
        self.data_x = df[df.columns[:-1]]
        self.data_y = df[df.columns[-1]]
        self.mean_value = None
        self.std_value = None

        if normalized:
            self.mean_value = self.data_x.mean()
            self.std_value = self.data_x.std()
            self.data_x = (self.data_x - self.mean_value) / (self.std_value + 1e-6)

    def get_normalize_params(self):
        return self.mean_value, self.std_value

    def __getitem__(self, index):
        x = self.data_x[index:index + 1].values[0]
        y = self.data_y[index:index + 1].values[0]
        return x, y

    def __len__(self):
        return len(self.data_y)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, df, normalized=True, normalize_params=None):
        self.data_x = df[df.columns[:-1]]
        self.data_y = df[df.columns[-1]]

        if normalized:
            if normalize_params:
                self.mean_value = normalize_params[0]
                self.std_value = normalize_params[1]
            else:
                self.mean_value = self.data_x.mean()
                self.std_value = self.data_x.std()
            self.data_x = (self.data_x - self.mean_value) / (self.std_value + 1e-6)

    def __getitem__(self, index):
        x = self.data_x[index:index + 1].values[0]
        y = self.data_y[index:index + 1].values[0]
        return x, y

    def __len__(self):
        return len(self.data_y)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, normalized=True, normalize_params=None):
        self.data_x = df

        if normalized:
            if normalize_params:
                self.mean_value = normalize_params[0]
                self.std_value = normalize_params[1]
            else:
                self.mean_value = self.data_x.mean()
                self.std_value = self.data_x.std()
            self.data_x = (self.data_x - self.mean_value) / (self.std_value + 1e-6)

    def __getitem__(self, index):
        x = self.data_x[index:index + 1].values[0]
        return x

    def __len__(self):
        return len(self.data_x)