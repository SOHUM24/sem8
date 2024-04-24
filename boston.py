import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
%matplotlib inline

df = pd.read_csv("boston.csv")

df.head()

df.dropna(inplace = True)
df.isnull().sum()

df.info()

correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)

sns.pairplot(df)

X = df.drop('PRICE', axis = 1)
Y = df['PRICE']
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

num_features = X_train.shape[1]

model = Sequential()

model.add(Dense(64, input_dim=num_features, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_test_predict = model.predict(X_test)
mse = mean_squared_error(Y_test, y_test_predict)
rmse = (np.sqrt(mse))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for training set")
print("--------------------------------------")
print('Mean Squared Error (MSE) is {}'.format(mse))
print('Root Mean Squared Error (RMSE) is {}'.format(rmse))
print('R2 score is {}'.format(r2))