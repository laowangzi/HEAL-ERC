import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd

class MyDataset(Dataset):

    def __init__(self, path, dtype='train'):
        self.videoText, self.videoSpeakers, self.videoLabels, \
        self.trainVid, self.testVid, self.devVid = pickle.load(open(path, 'rb'), encoding='latin1')
                 
        if dtype == 'train':
            self.keys = [x for x in self.trainVid]
        elif dtype == 'dev':
            self.keys = [x for x in self.devVid]
        else:
            self.keys = [x for x in self.testVid]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.LongTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        #len(dat)== 7;  dat[0]索引的是本batch内所有的videoText; pad_sequence(dat[i]).shape== (110, BS, Dim)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]
    