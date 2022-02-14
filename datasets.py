# Copyright (c) 2021-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
from sklearn import datasets
import torch 
import numpy as np 
import dataloaders

name2class = {}

def register_class(names):
    
    def _register(cls):
        if names is None:
            local_names = [cls.__name__]
        else:
            local_names = names
        for name in local_names:
            name2class[name] = cls
        return cls

    return _register


@register_class(["twomoons"])
class TwoMoons(Dataset):
    def __init__(self, config, partition):
        self.dataset = torch.FloatTensor(datasets.make_moons(n_samples=12800*50, noise=0.001)[0].astype(np.float32))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        d = self.dataset[idx]
        return d+0.08*torch.randn_like(d)

@register_class(['power', 'gas', 'hepmass', 'miniboone', 'bsds300'])
class UCI(Dataset):
    def __init__(self, config, partition="train"):
        name = config.dataset
        if name == 'bsds300':
            dataset = dataloaders.BSDS300()
        elif name == 'power':
            dataset = dataloaders.POWER()
        elif name == 'gas':
            dataset = dataloaders.GAS()
        elif name == 'hepmass':
            dataset = dataloaders.HEPMASS()
        elif name == 'miniboone':
            dataset = dataloaders.MINIBOONE()

        if partition == "train":
            self.dataset = torch.from_numpy(dataset.trn.x)
        elif partition == 'val':
            self.dataset = torch.from_numpy(dataset.val.x)
        elif partition == "test":
            self.dataset = torch.from_numpy(dataset.tst.x)
        
        self.dataset = self.dataset[:,config.dims]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

