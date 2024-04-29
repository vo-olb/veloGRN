import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, expression_data, size, data_split=[0.8, 0.2], 
                 flag='train', mode='all', padding=False):
        # expr: (c, g)
        # padding param is True only when velocity computing is needed
        self.expression_data = expression_data
        if padding:
            pad = pd.concat([self.expression_data.iloc[[-1], ]] * size[1], ignore_index=True)
            self.expression_data = pd.concat([self.expression_data, pad], ignore_index=True)
        self.in_len = size[0]
        self.out_len = size[1]
        self.data_split = data_split
        self.mode = mode
        self.split(flag)

    def split(self,flag):
        train_num = int(len(self.expression_data)*self.data_split[0])
        val_num = int(len(self.expression_data) * self.data_split[1])
        test_num = int(len(self.expression_data))
        
        type_map = {'train':0, 'val':1, 'test':2}
        border_start = [0, train_num, 0]
        border_end = [train_num, train_num+val_num, test_num]
        stt, end = border_start[type_map[flag]], border_end[type_map[flag]]

        data = self.expression_data.values
        self.data_x, self.data_y = data[stt:end], data[stt:end]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        seq_x = self.data_x[s_begin:s_end]

        if self.mode == 'encode':
            seq_y = seq_x
        else:
            r_begin = s_end
            r_end = r_begin + self.out_len
            seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.in_len - self.out_len * int(self.mode != 'encode') + 1

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
