import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from load_data import load_data
# 加载模型
from tensorflow.keras.models import load_model


# 生成一个简单的时间序列数据
def generate_time_series(n):
    time = np.arange(0, n)
    data = np.sin(0.1 * time) + 0.1 * np.random.randn(n)
    return data


# 将时间序列转换成具有观察窗口的样本
def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequence = data[i:i+window_size]
        label = data[i+window_size]
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)
def LSTM_predict(x_num):

    # 准备数据
    time_series = load_data()

    window_size = 10
    X, y = create_sequences(time_series, window_size)
    # print(X, y)

    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # # 构建LSTM模型
    # model = Sequential()
    # model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    # model.add(Dense(1))
    #
    # # 编译模型
    # model.compile(optimizer='adam', loss='mse')
    #
    # # 训练模型
    # model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    #
    # model.save('lstm_model.h5')

    loaded_model = load_model('lstm_model.h5')




    print('X_test', X_test)
    print(len(X_test))
    # 在测试集上进行预测
    # y_pred = model.predict(X_test)

    # 使用加载的模型进行预测
    result = loaded_model.predict(X_test)

    print(result)


def LSTM_train():
    # 准备数据
    time_series = load_data()

    window_size = 6
    X, y = create_sequences(time_series, window_size)
    # print(X, y)

    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.save('lstm_model.h5')

# data:[[71 72 73 74 75 76]]
def LSTM_apply(data):
    loaded_model = load_model('lstm_model.h5')
    result = loaded_model.predict(data)
    print(result)
    return result

# LSTM_train()
# LSTM_apply(np.array([[-20, -20, -10, -10, -10, -10]]))


