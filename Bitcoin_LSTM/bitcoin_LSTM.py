import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

# reproducibility
tf.set_random_seed(777)

# train Parameters
seq_length = 60 # 데이터가 분으로 되어있기 때문에 1시간 단위로 예측하고 싶었음.
data_dim = 8 # data의 차원: feature로서 8가지 사용
hidden_dim = 16
output_dim = 1
learning_rate = 0.01
iterations = 500

# diff_24h, diff_per_24h, bid, ask, low, high, volume, last
xy = np.loadtxt('bitcoin_kr.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy_t = xy
xy = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
xy = xy.fit_transform(xy_t)
x = xy
y = xy[:, [-1]]

# build a dataset
dataX = []
dataY = []

# 60분을 하나의 x 데이터값으로 만든 후 y값 예측하는 식으로 만들기.
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 맨 마지막 FC가 위치해 있다.
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

# cost/loss function
loss = tf.reduce_sum(tf.square(Y_pred - Y))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()