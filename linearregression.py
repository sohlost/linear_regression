import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('##YOURTRAININGDATALINKHERE')
df_test = pd.read_csv('##YOURTESTDATALINKHERE')

# Do the one-hot encoding here
df_new = pd.get_dummies(df, columns=['##COLUMS_WHERE_ONE_HOT_ENCODING_IS_NEEDED'])

df_maxmin = pd.DataFrame(columns=['Max', 'Min'])
for i in df_new.columns:
  min = df_new[i].min()
  max = df_new[i].max()
  df_maxmin = df_maxmin._append({'Max': max, 'Min': min}, ignore_index=True)
  df_new[i] = (df_new[i] - min)/(max-min)

y_true = df_new['##Y_AXIS_DATA'].values
X = df_new.drop('##Y_AXIS_DATA', axis=1).values

W = np.random.randn(X.shape[1],1).flatten()
b = np.random.random()
y_pred = np.dot(X,W) + b
print(y_pred)

def mse_loss_fn(y_true, y_pred):
    square_sum =0
    for i in range(len(y_true)):
      square_sum = (y_true[i] - y_pred[i])**2 + square_sum
    L =  square_sum/(2*len(y_true))
    return L

def get_gradients(y_true, y_pred, W, b, X):
    sum_w = 0
    sum_b = 0
    for i in range(len(y_true)):
      sum_w += (y_pred[i] - y_true[i])*X[i]
    dw = sum_w/len(y_pred)
    for i in range(len(y_true)):
      sum_b += y_pred[i] - y_true[i]
    db = sum_b/len(y_pred)
    return dw, db    

def update(weights, bias, gradients_weights, gradients_bias, lr):
    W = weights - lr*gradients_weights
    b = bias - lr*gradients_bias
    return W,b

NUM_EPOCHS = 1000
LEARNING_RATE = 2e-2
losses = []

for epoch in range(NUM_EPOCHS):
     y_pred = (np.dot(X,W) + b).flatten()
     print(W)
     print()
     loss = mse_loss_fn(y_true, y_pred)
     losses.append(loss)
     gradients_weights, gradients_bias = get_gradients(y_true, y_pred, W, b, X)
     W,b = update(W, b, gradients_weights, gradients_bias, LEARNING_RATE)
print(losses)
plt.plot(losses)
plt.show()

df_test = pd.get_dummies(df_test, columns=['##Y_AXIS_DATA'])
j = 0
for col in df_test.columns:
  df_test[col] = (df_test[col] - df_maxmin['Min'][j])/(df_maxmin['Max'][j] - df_maxmin['Min'][j])
  j=j+1
y_test_true = df_test['##Y_AXIS_DATA'].values
X_test = df_test.drop('##Y_AXIS_DATA', axis=1).values
y_test_pred = (np.dot(X_test,W) + b).flatten()
loss_test = mse_loss_fn(y_test_true, y_test_pred)
print(loss_test)










