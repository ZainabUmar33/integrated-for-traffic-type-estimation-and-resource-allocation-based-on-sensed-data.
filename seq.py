import math
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal

plt.rcParams['font.sans-serif']=['SimSun']
plt.rcParams['axes.unicode_minus'] = False
config = {
            "font.family": 'serif',
            "font.size": 11,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)

def general_equation(first_x,first_y,second_x,second_y):
    # 斜截式 y = kx + b
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

def envelope_of_traffic(arr):
    peak_id_up, peak_property_up = scipy.signal.find_peaks(arr)

    for i in range(len(peak_id_up) - 1):
        x1 = peak_id_up[i]
        x2 = peak_id_up[i + 1]
        y1 = arr[x1]
        y2 = arr[x2]
        k, b = general_equation(x1, y1, x2, y2)

        j = x1 + 1
        while j < x2:
            arr[j] = k * j + b
            j = j + 1
    return arr

def arr_move_right(arr1):
    arr = np.zeros(len(arr1))
    for i in range(len(arr1) - 1):
        arr[i + 1] = arr1[i]
    return arr

# 构造LSTM模型输入需要的训练数据
def get_dataset1(n_in, n_out, result):
    n_samples = len(result) - n_in - n_out + 1
    X1, X2, y = list(), list(), list()
    for i in range(n_samples):
        # 生成输入序列
        source = result[i : i + n_in]
        # 定义目标序列，这里就是输入序列的前三个数据
        target = result[i + n_in : i + n_in + n_out]
        # 向前偏移一个时间步目标序列
        target_in = arr_move_right(target)
        # 直接使用to_categorical函数进行on_hot编码
        X1.append(source)
        X2.append(target_in)
        y.append(target)
    return array(X1).reshape(len(X1), len(X1[0]), 1), array(X2).reshape(len(X2), len(X2[0]), 1), array(y).reshape(len(y), len(y[0]), 1), n_samples

# 构造LSTM模型输入需要的测试数据
def get_dataset2(n_in, n_out, result):

    n_samples = int(len(result) / (n_in + n_out))
    X1, X2, y = list(), list(), list()
    for i in range(n_samples):
        # 生成输入序列
        j = i * (n_in + n_out)
        source = result[j : j + n_in]
        target = result[j + n_in: j + n_in + n_out]
        X1.append(source)
        y.append(target)
    return array(X1), array(y), n_samples


# 构造Seq2Seq训练模型model, 以及进行新序列预测时需要的的Encoder模型:encoder_model 与Decoder模型:decoder_model
def define_models(n_input, n_output, n_units):
    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # 仅保留编码状态向量
    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # 返回需要的三个模型
    return model, encoder_model, decoder_model


def predict_sequence(infenc, infdec, source, n_steps, n_samples, cardinality):
    # 输入序列编码得到编码状态向量
    state = infenc.predict(source)
    # 输出序列列表
    res = list()
    for i in range(n_samples):
        print(i)
        state1 = state[0][i]
        state1 = state1.reshape(1, len(state1))
        state2 = state[1][i]
        state2 = state2.reshape(1, len(state2))
        state_temp = [state1, state2]
        # 初始目标序列输入：通过开始字符计算目标序列第一个字符，这里是0
        target_seq = np.zeros(shape=(1, 1, cardinality))
        output = list()
        for t in range(n_steps):
            # predict next char
            yhat, h, c = infdec.predict([target_seq] + state_temp)
            # 截取输出序列，取后三个
            output.append(yhat[0, 0, :])
            # 更新状态
            state_temp = [h, c]
            # 更新目标序列(用于下一个词预测的输入)
            target_seq = yhat
        res.append(output)
    return array(res)


# 参数设置
n_steps_in = 4
n_steps_out = 2

# 读取训练数据
#r'C:\\Users\jiani\Desktop\论文实验图\make\1s\u1_traffic1.txt'
train = np.loadtxt(r'C:\\Users\jiani\Desktop\论文实验图\1s\log-tf.txt', delimiter=',')
# train = train[0 : 199]
# train = envelope_of_traffic(train)
train = np.reshape(train, (-1, 1))
sc = MinMaxScaler()
train = sc.fit_transform(train)
train = np.reshape(train, len(train))

# 生成训练数据
X1, X2, y, num = get_dataset1(n_steps_in, n_steps_out, train)
# 定义模型
train, infenc, infdec = define_models(1, 1, 128)
train.compile(optimizer='adam', loss='mean_squared_error')
print(X1.shape, X2.shape, y.shape)

# 训练模型
train.fit([X1, X2], y, epochs=1000)


# 读取预测数据
test = np.loadtxt(r'C:\\Users\jiani\Desktop\论文实验图\1s\log-tf2.txt', delimiter=',')
# test = test[200 : 499]
# test = envelope_of_traffic(test)
test = np.reshape(test, (-1, 1))

sc = MinMaxScaler()
test = sc.fit_transform(test)
test = np.reshape(test, len(test))
# 查看预测结果
X1, _, y, num = get_dataset1(n_steps_in, n_steps_out, test)
target = predict_sequence(infenc, infdec, X1, n_steps_out, num, 1)


for i in range(num):
    X1[i] = sc.inverse_transform(X1[i])
    y[i] = sc.inverse_transform(y[i])
    target[i] = sc.inverse_transform(target[i])
    print('X=%s y=%s, yhat=%s' % (X1[i], y[i], target[i]))


y_pred = np.zeros(len(y) + n_steps_out - 1)
y_test = np.zeros(len(target) + n_steps_out - 1)
index1 = 0
index2 = 0

for i in range(len(y)):
    if i == 0:
        for j in range(n_steps_out):
            y_pred[index1] = y[i][j][0]
            y_test[index2] = target[i][j][0]
            index1 = index1 + 1
            index2 = index2 + 1
    else:
        y_pred[index1] = y[i][n_steps_out - 1][0]
        y_test[index2] = target[i][n_steps_out - 1][0]
        index1 = index1 + 1
        index2 = index2 + 1

y_pred = np.reshape(y_pred, -1)
y_test = np.reshape(y_test, -1)
# 参数评估
mae = np.sum(np.abs(y_pred - y_test)) / len(y)
mse = np.sum((y_pred - y_test) ** 2) / len(y)
rmse = math.sqrt(mse)
mean = np.sum(y_pred) / len(y_pred)

r2 = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_pred - mean) ** 2))#均方误差/方差
mape = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2," mape", mape)

# np.savetxt(r'C:\\Users\jiani\Desktop\论文实验图\100s\y_pred.txt.', y_test, fmt='%d', delimiter=',')
# np.savetxt(r'C:\\Users\jiani\Desktop\论文实验图\100s\y_test.txt', y_pred, fmt='%d', delimiter=',')

y_test = y_test[1 : len(y_test)]
plt.figure(constrained_layout=True, figsize=(6, 2.3))
plt.plot(range(np.array(len(y_pred))), y_pred / 1024, "blue")
plt.plot(range(np.array(len(y_test))), y_test / 1024, "red")
plt.legend(['真实值', '预测值'], fontsize = 11)
plt.xlabel('时间(0.1s)', fontsize = 11)
plt.ylabel('用户流量(KB)', fontsize = 11)
plt.xticks(fontproperties = 'Times New Roman',fontsize=11)
plt.yticks(fontproperties = 'Times New Roman',fontsize=11)
plt.show()