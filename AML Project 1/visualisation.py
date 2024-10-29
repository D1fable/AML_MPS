import pandas as pd
import numpy as np
import plotly.express as px

from preprocessing import Dataset_NN, pre_test

X_train_df = pd.read_csv('Data/X_train.csv')

X_raw = X_train_df.drop(columns=['id']).values
_, X_processed = pre_test(X_raw, X_raw)

z_scores = np.abs((X_processed - X_processed.mean(axis=0)) / X_processed.std(axis=0))
threshold = 3.1
inliers = (z_scores < threshold).all(axis=1)

X_z = X_processed[inliers]

a = X_raw.shape
b = X_processed.shape
c = X_z.shape

print(f'Raw shape: {a}')
print(f'Processed shape: {b}')
print(f'Z shape: {c}')


# Create the scatter plot
fig2_r = px.histogram(X_raw[:,0], nbins=100, labels={'value': 'Second Feature Value raw'}, title="Histogram of the Second Feature")
fig2_p = px.histogram(X_processed[:,0], nbins=100, labels={'value': 'Second Feature Value processed'}, title="Histogram of the Second Feature")
fig2_z = px.histogram(X_z[:,0], nbins=100, labels={'value': 'Second Feature Value processed and z'}, title="Histogram of the Second Feature")

fig2_r.write_html("Images/second_feature_plot_r.html")
fig2_p.write_html("Images/second_feature_plot_p.html")
fig2_z.write_html("Images/second_feature_plot_z.html")
