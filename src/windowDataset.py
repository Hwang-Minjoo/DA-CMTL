from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import torch
from src.config import Config as c

    
class windowDataset(Dataset):
    def __init__(self, x, iw=12, ow=6, stride=5):
        L, n_features = x.shape[0], x.shape[1]
        n_samples = (L - iw - ow) // stride + 1
        X = np.zeros([iw, n_samples, n_features]).astype('float32')
        Y = np.zeros([1, n_samples, 1]).astype('float32')
        
        for i in np.arange(n_samples):
            x_idx1 = stride*i; x_idx2 = x_idx1 + iw;
            y_idx = x_idx2 + ow -1
            X[:,i] = x[x_idx1:x_idx2]
            Y[:,i] = x[y_idx, 0]

        X = X.reshape(X.shape[0], X.shape[1], -1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], -1).transpose((1,0,2))
        self.x, self.y, self.len = X, Y, len(X)
        # print(f"X: {X.shape}, Y: {Y.shape}")
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len
    def get_shape(self):
        return self.x.shape, self.y.shape
    
def get_personal_loader(data_X, iw, ow, stride, batch_size, train):
    dataset = windowDataset(data_X, iw=iw, ow=ow, stride=stride)
    if train==True:
        dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last =True)
    else:
        dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last =False)
    return dataloader

def get_multiple_loader(data_X, iw, ow, stride, batch_size, mm_scaler):
    dataset = torch.utils.data.ConcatDataset([windowDataset(mm_scaler.transform(data_X[i]).squeeze(), iw, ow, stride) for i in range(len(data_X))])
    dataloader =  DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    return dataloader

def get_scaler(train_X):
    scaler = MinMaxScaler()
    scaler = scaler.fit(train_X.reshape(-1,2))

    scaler_1d = MinMaxScaler()
    scaler_1d = scaler_1d.fit(train_X[:,:,0].reshape(-1,1))

    scaled_hypo = torch.tensor(scaler_1d.transform(np.array([70]).reshape(-1,1)), dtype=torch.float32).to(c.device)

    return scaler, scaler_1d, scaled_hypo













    
