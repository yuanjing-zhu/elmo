from torch.utils.data import Dataset
import os
import json


class DefaultDataSet(Dataset):
    #TODO: train/test option?
    def __init__(self, path='processed/'):
        super(DefaultDataSet, self).__init__()
        self.path = path
        self.files = os.listdir(path)

    def __getitem__(self, item):
        file = self.path+str(item)
        data = json.load(open(file, 'r'))
        return data['ids']

    def __len__(self):
        return len(self.files)