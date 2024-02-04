import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 生成一个简单的时间序列数据
def generate_time_series(n):
    time = np.arange(0, n)
    data = np.sin(0.1 * time) + 0.1 * np.random.randn(n)
    return data

# 准备数据
n = 1000
time_series = generate_time_series(n)

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

window_size = 10
X, y = create_sequences(time_series, window_size)

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

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='True Data')
plt.plot(np.arange(train_size + window_size, n), y_pred, label='Predictions')
plt.legend()
plt.show()
