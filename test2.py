import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 生成模拟数据
np.random.seed(42)
num_samples = 1000
num_features = 3  # A, B, C 三个输入变量
num_target_variables = 1

# 生成随机时间序列作为输入
A = np.random.random((num_samples, 10, 1))
B = np.random.random((num_samples, 10, 1))
C = np.random.random((num_samples, 10, 1))

# 合并三个输入变量
input_data = np.concatenate([A, B, C], axis=-1)

# 生成对应的目标序列（假设是回归任务）
target_data = np.random.random((num_samples, num_target_variables))

# 划分训练集和测试集
train_size = int(len(input_data) * 0.8)
train_input, test_input = input_data[:train_size], input_data[train_size:]
train_target, test_target = target_data[:train_size], target_data[train_size:]

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(10, num_features)))
model.add(Dense(num_target_variables))

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(train_input, train_target, epochs=10, batch_size=32, validation_data=(test_input, test_target))

# 在测试集上进行预测
predictions = model.predict(test_input)

# 打印模型摘要
model.summary()
