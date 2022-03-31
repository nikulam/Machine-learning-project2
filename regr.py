import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('curry_ordered.csv', delimiter=',')
df = df.drop(columns=['Result', 'T Score', 'OPP'])

scaler = MinMaxScaler()
df[['LAST_N_3PTM', 'LAST_N_PTS', 'SEASON_AVG']] = scaler.fit_transform(df[['LAST_N_3PTM', 'LAST_N_PTS', 'SEASON_AVG']])

print(df.columns.difference(['PTS']))
X = np.array(df[df.columns.difference(['PTS'])])
y = np.array(df['PTS']).reshape(-1, 1)

pca = PCA(n_components=1)
x_pca = pd.DataFrame(pca.fit_transform(df.drop(columns=['PTS'])))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=28)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

lin = LinearRegression()
hub = HuberRegressor(epsilon=1, alpha=0.01)

hub.fit(X_train, y_train.reshape(-1,))

y_pred_train = hub.predict(X_train)
print(mae(y_pred_train, y_train.reshape(-1,)))
y_pred_val = hub.predict(X_val)
print(mae(y_pred_val, y_val.reshape(-1,)))
y_pred_test = hub.predict(X_test)
print(mae(y_pred_test, y_test))

'''
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 
for train_index, val_index in kfold.split(X_train_val):
    #print("Train:", train_index, "Validation:",val_index)
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_train = lin.predict(X_train)
    y_pred_val = lin.predict(X_val)
    tr_error = mean_squared_error(y_train, y_pred_train)
    val_error = mean_squared_error(y_val, y_pred_val)
    
    if tr_error < 76 and val_error < 76: break

print(tr_error, val_error)
'''



