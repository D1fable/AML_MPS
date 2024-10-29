from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset

class Dataset_NN(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def pre_test(data_train, data_test):
    scalar = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    pca = PCA(n_components=0.95)
    
    data_train = imputer.fit_transform(data_train) 
    data_test = imputer.transform(data_test)
    
    data_train = scalar.fit_transform(data_train)
    data_test = scalar.transform(data_test)

    data_train = pca.fit_transform(data_train)
    data_test = pca.transform(data_test)

    return data_train, data_test