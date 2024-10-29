import pandas as pd
import numpy as np

from preprocessing import Dataset_NN, pre_test
from train import train, Model

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
from torch.utils.data import DataLoader, Dataset



#Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing using device: {device}...\n')


#Import Data frames with panda
X_train_df = pd.read_csv('Data/X_train.csv')
y_test_df = pd.read_csv('Data/y_train.csv')

X_test_df = pd.read_csv('Data/X_test.csv')
print("Finished Importing Data...\n")


#Get numpy Arrays
X_test = X_test_df.drop(columns=['id']).values
X = X_train_df.drop(columns=['id']).values
y = y_test_df['y'].values


#Preprocessing Data
X, X_test = pre_test(X, X_test)
print("Finished preprocessing Data...\n")
print(X.shape)

'''
z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
threshold = 3.1
inliers = (z_scores < threshold).all(axis=1)
X = X[inliers]
y = y[inliers]
'''



#Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)



# Create datasets and data loaders for NN
train_dataset = Dataset_NN(X_train, y_train)
val_dataset = Dataset_NN(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = Model(521).to(device)
train(model, 300, train_loader, val_loader, device)


'''
svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.1)
svr_model.fit(X, y.ravel())
y_pred = svr_model.predict(X_test)
#r2 = r2_score(y_val, y_pred)
#print(f"R2 Score: {r2:.3f}")
'''



#submission = pd.DataFrame({'id': X_test_df['id'], 'target': y_pred})
#submission.to_csv('submission3_mm.csv', index=False)
