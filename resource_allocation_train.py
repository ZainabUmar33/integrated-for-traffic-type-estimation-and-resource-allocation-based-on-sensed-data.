
"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math
import time
from typing import Optional
import numpy as np
from gym import spaces
from collections import deque
import tensorflow as tf
import tensorlayer as tl
import random
import http.client
from urllib import request, parse
import datetime, pytz

action_num = 100

TTI = 1  # ms
T = 10
K = 100
N = 3
C = 3

x_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# 信号传输模型
GaussianNoise_power = math.pow(10, 0.1 * -27)  # -27dBm
Channel_gain_min = math.pow(10, 0.1 * 1)
Channel_gain_max = math.pow(10, 0.1 * 10)
PRB_max = 50
PRB_min = 5
path_loss_min = math.pow(10, 0.1 * 2)  # 5dB
path_loss_max = math.pow(10, 0.1 * 3)

Transmit_power_min = math.pow(10, 0.1 * 9)  # dBm
Transmit_power_max = math.pow(10, 0.1 * 28)

# 基站计算模型
BaseStation_traffic = np.zeros(shape=(N, K))
BaseStation_traffic[0] = np.random.randint(100000, 300000, (1, K))
BaseStation_traffic[1] = np.random.randint(20000, 50000, (1, K))
BaseStation_traffic[2] = np.random.randint(60000, 900000, (1, K))
BaseStation_traffic = np.transpose(BaseStation_traffic)

BaseStation_traffic_probability = np.random.uniform(0.1, 0.3, (K, N))
CPU_max = 0.99  # %
CPU_min = 0.1
process_CPU = 0.01

# self.latency_max =

# 优化问题中的权重值
latency_weight = 1
packet_loss_rate_weight = 0.2
SE_weight = 0.00002

T_nextState = np.random.randint(5, T + 1)
deque_nextState = deque
path_loss_nextState = np.random.uniform(path_loss_min, path_loss_max, (1, C))
PRB_nextState = np.random.randint(PRB_min, PRB_max, (1, C))
Transmit_power_nextState = np.random.uniform(Transmit_power_min, Transmit_power_max, (1, C))
Channel_gain_nextState = np.random.uniform(Channel_gain_min, Channel_gain_max, (1, C))
CPU_nextState = np.random.uniform(CPU_min, CPU_max)
BaseStation_traffic_nextStation = np.random.randint(80000, 4200000, (5, N))

action_array = np.zeros(shape=(action_num, 2 + 3 * C))

# 记录资源变化情况
rate = np.zeros(shape=(K, N))
latency = np.zeros(shape=(K, N))
PRB_resource = np.zeros(shape=(K, C))
transmit_power_resource = np.zeros(shape=(K, C))
CPU_resource = np.zeros(shape=(K, 1))


# 资源利用率


class ResourceAllocationEnv():

    def __init__(self, goal_velocity=0):
        # 系统QoS建模
        self.deque = []
        for i in range(N):
            self.deque.append(deque())

        self.CurrentTime = 0

        for i in range(action_num):
            t_array_action = np.random.randint(5, T + 1, (1, 1))
            PRB_array_action = np.random.randint(-2, 2, (1, C))
            transmit_power_array_action = np.random.uniform(-1, 1, (1, C))
            channel_gain_array_action = np.random.uniform(-1, 2, (1, C))
            CPU_array_action = np.random.uniform(-0.1, 0.1, (1, 1))
            temp = np.hstack((t_array_action.ravel(), PRB_array_action.ravel(),
                              transmit_power_array_action.ravel(), channel_gain_array_action.ravel(),
                              CPU_array_action.ravel()))
            action_array[i] = temp
        '''

        '''

        self.screen = None
        self.Clock = None
        self.isopen = True

        self.path_loss_array_min = np.random.uniform(path_loss_min, path_loss_min, (1, C))
        self.PRB_array_min = np.random.randint(PRB_min, PRB_min + 1, (1, C))
        Transmit_power_array_min = np.random.uniform(Transmit_power_min, Transmit_power_min, (1, C))
        Channel_gain_array_min = np.random.uniform(Channel_gain_min, Channel_gain_min + 1, (1, C))
        BaseStation_traffic_min = np.random.randint(80000, 80001, (5, N))

        self.min_observation = np.random.uniform(path_loss_min, path_loss_min, (1, C))
        self.min_observation = np.append(self.min_observation, self.PRB_array_min)
        self.min_observation = np.append(self.min_observation, Transmit_power_array_min)
        self.min_observation = np.append(self.min_observation, Channel_gain_array_min)
        self.min_observation = np.append(self.min_observation, CPU_min)
        self.min_observation = np.append(self.min_observation, BaseStation_traffic_min)

        self.path_loss_array_max = np.random.uniform(path_loss_max, path_loss_max, (1, C))
        self.PRB_array_max = np.random.randint(PRB_max, PRB_max + 1, (1, C))
        Transmit_power_array_max = np.random.uniform(Transmit_power_max, Transmit_power_max, (1, C))
        Channel_gain_array_max = np.random.uniform(Channel_gain_max, Channel_gain_max + 1, (1, C))
        BaseStation_traffic_max = np.random.randint(4200000, 4200001, (5, N))

        self.max_observation = np.random.uniform(path_loss_max, path_loss_max, (1, C))
        self.max_observation = np.append(self.max_observation, self.PRB_array_max)
        self.max_observation = np.append(self.max_observation, Transmit_power_array_max)
        self.max_observation = np.append(self.max_observation, Channel_gain_array_max)
        self.max_observation = np.append(self.max_observation, CPU_max)
        self.max_observation = np.append(self.max_observation, BaseStation_traffic_max)

        self.action_space = spaces.Discrete(action_num)
        self.observation_space = spaces.Box(self.min_observation, self.max_observation, dtype=np.float32)

    def step(self, action):

        # 当前状态
        path_loss_now = self.state[0: C]
        PRB_now = self.state[C: 2 * C]
        transmit_power_now = self.state[2 * C: 3 * C]
        channel_gain_now = self.state[3 * C: 4 * C]
        CPU_now = self.state[4 * C]
        BaseStation_traffic_now = self.state[4 * C + 1: 4 * C + 1 + N * 5]

        # 当前动作
        # t_action = int(action[0])
        PRB_action = action[1: C + 1]
        transmit_power_action = action[C + 1: 2 * C + 1]
        channel_gain_action = action[2 * C + 1: 3 * C + 1]
        CPU_action = action[3 * C + 1]

        # 优化问题
        # 计算速率
        compute_process_rate = ((CPU_now + CPU_action) * 3000000) / process_CPU
        # 排队时延

        # 效用函数
        utility_function = 0.0
        QoS = 0.0
        SE = 0.0
        # 计算排队时延
        sum = 0
        latency1 = 0
        if self.CurrentTime < len(BaseStation_traffic):
            CPU_resource[self.CurrentTime][0] = CPU_now + CPU_action
            for j in range(N):
                sum += BaseStation_traffic[self.CurrentTime][j]
        latency1 = sum / compute_process_rate

        # 往deque中防暑数据流
        for i in range(N):
            if self.CurrentTime < len(BaseStation_traffic):
                self.deque[i].append(BaseStation_traffic[self.CurrentTime][i])

        QoS_n_c_t = 0
        SE_n_c_t = 0
        for i in range(N):
            signal_transmission_rate_temp = 0
            signal_transmission_rate_temp_n_c = 0
            deque_now = self.deque[i]
            for j in range(C):
                # 信噪比
                SNR = x_array[i][j] * (
                        (transmit_power_now[j] + transmit_power_action[j]) * path_loss_now[j] * math.pow(
                    (channel_gain_now[j] + channel_gain_action[j]), 2)) / GaussianNoise_power
                signal_transmission_rate_temp_n_c += (PRB_now[j] + PRB_action[j]) * 180000 * math.log(1 + SNR, 2)
                if self.CurrentTime < K:
                    PRB_resource[self.CurrentTime][j] = PRB_now[j] + PRB_action[j]
                    transmit_power_resource[self.CurrentTime][j] = transmit_power_now[j] + transmit_power_action[j]

            # 数据传输速率
            signal_transmission_rate = TTI * 0.001 * signal_transmission_rate_temp_n_c
            if self.CurrentTime < K:
                rate[self.CurrentTime][i] = signal_transmission_rate
            signal_transmission_rate_temp = signal_transmission_rate
            # 传输时延
            latency2 = 0

            while len(deque_now) > 0:
                top_package = deque_now.popleft()
                latency2 += (top_package / signal_transmission_rate_temp) * 0.001

            # 数据包丢失的概率和
            probability = 0
            if self.CurrentTime + i < len(BaseStation_traffic_probability):
                for j in range(N):
                    probability += BaseStation_traffic_probability[self.CurrentTime + i][j]
            probability = probability / N
            QoS_n_c_t += latency_weight * math.exp(
                -(latency1 + latency2) / (TTI * 0.001)) + packet_loss_rate_weight * math.exp(
                -probability / (TTI * 0.001))

            if self.CurrentTime < K:
                latency[self.CurrentTime][i] = latency1 + latency2
            # 频谱效率
            SE_n_c_t += SE_weight * signal_transmission_rate

        QoS += QoS_n_c_t
        SE += SE_n_c_t
        utility_function = QoS + SE

        done = bool(self.CurrentTime >= K)
        reward = utility_function

        # 随机选定一个状态 t, deque_now, path_loss_now, PRB_now, transmit_power_now, CPU_now = self.state

        deque_nextState = self.deque
        # t_nextState = np.random.randint(1, T + 1, (1,1))
        path_loss_nextState = np.random.uniform(path_loss_min, path_loss_max, (1, C))
        while True:
            PRB_nextState = np.random.randint(PRB_min, PRB_max, (1, C))
            sum = 0
            for i in range(C):
                sum += PRB_nextState[0][i]
            if sum <= PRB_max:
                break
        while True:
            Transmit_power_nextState = np.random.uniform(Transmit_power_min, Transmit_power_max, (1, C))
            sum = 0
            for i in range(C):
                sum += Transmit_power_nextState[0][i]
            if sum <= Transmit_power_max:
                break
        Channel_gain_nextState = np.random.uniform(Channel_gain_min, Channel_gain_max, (1, C))
        CPU_nextState = np.random.uniform(CPU_min, CPU_max)
        self.CurrentTime += 1
        BaseStation_traffic_nextStation = np.zeros(shape=(5, N))
        for m in range(5):
            if self.CurrentTime + m < K:
                BaseStation_traffic_nextStation[m] = BaseStation_traffic[self.CurrentTime + m]
        other = np.hstack((path_loss_nextState.ravel(), PRB_nextState.ravel(), Transmit_power_nextState.ravel(),
                           Channel_gain_nextState.ravel(), CPU_nextState, BaseStation_traffic_nextStation.ravel()))

        self.state = other
        return np.array(self.state, dtype=np.float32), reward, done

    # 初始状态
    def reset(self):
        # 把数据流放入队列中
        for i in range(N):
            if self.CurrentTime < len(BaseStation_traffic):
                dequeTemp = self.deque[i]
                dequeTemp.append(BaseStation_traffic[self.CurrentTime][i])

        deque_nextState = self.deque
        while True:
            PRB_nextState = np.random.uniform(PRB_min, PRB_max, (1, C))
            sum = 0
            for i in range(C):
                sum += PRB_nextState[0][i]
            if sum <= PRB_max:
                break
        while True:
            Transmit_power_nextState = np.random.uniform(Transmit_power_min, Transmit_power_max, (1, C))
            sum = 0
            for i in range(C):
                sum += Transmit_power_nextState[0][i]
            if sum <= Transmit_power_max:
                break
        path_loss_nextState = np.random.uniform(path_loss_min, path_loss_max, (1, C))
        Channel_gain_nextState = np.random.uniform(Channel_gain_min, Channel_gain_max, (1, C))
        CPU_nextState = np.random.uniform(CPU_min, CPU_max)
        BaseStation_traffic_nextStation = np.zeros(shape=(5, N))
        for m in range(5):
            if self.CurrentTime + m < K:
                BaseStation_traffic_nextStation[m] = BaseStation_traffic[self.CurrentTime + m]

        other = np.hstack((path_loss_nextState.ravel(), PRB_nextState.ravel(), Transmit_power_nextState.ravel(),
                           Channel_gain_nextState.ravel(), CPU_nextState, BaseStation_traffic_nextStation.ravel()))
        self.state = other

        return np.array(self.state, dtype=np.float32)

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55


class Double_DQN():
    def __init__(self):
        self.env = ResourceAllocationEnv()  # 定义环境
        self.input_dim = self.env.observation_space.shape[0]  # 定义网络的输入形状，这里就是输入S

        # 建立两个网络
        self.Q_network = self.get_model()  # 建立一个Q网络
        self.Q_network.train()  # 在tensorlayer要指定这个网络用于训练。
        self.Target_Q_network = self.get_model()  # 创建一个target_Q网络
        self.Target_Q_network.eval()  # 这个网络指定为不用于更新。

        ## epsilon-greedy相关参数
        self.epsilon = 1.0  # epsilon大小，随机数大于epsilon，则进行开发；否则，进行探索。
        self.epsilon_decay = 0.995  # 减少率：epsilon会随着迭代而更新，每次会乘以0.995
        self.epsilon_min = 0.01  # 小于最小epsilon就不再减少了。

        # 其余超参数
        self.memory = deque(maxlen=200)  # 队列，最大值是2000
        self.batch = 64
        self.gamma = 0.95  # 折扣率
        self.learning_rate = 1e-2  # 学习率
        self.opt = tf.optimizers.Adam(self.learning_rate)  # 优化器

    # dueling DQN只改了网络架构。
    def get_model(self):
        # 第一部分
        input = tl.layers.Input(shape=[None, self.input_dim])
        h1 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(input)
        h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(h1)
        # 第二部分
        svalue = tl.layers.Dense(action_num, )(h2)
        # 第三部分
        avalue = tl.layers.Dense(action_num, )(h2)  # 计算avalue
        mean = tl.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(avalue)  # 用Lambda层，计算avg(a)
        advantage = tl.layers.ElementwiseLambda(lambda x, y: x - y)([avalue, mean])  # a - avg(a)

        output = tl.layers.ElementwiseLambda(lambda x, y: x + y)([svalue, avalue])
        return tl.models.Model(inputs=input, outputs=output)

    def update_epsilon(self):
        '''
        用于更新epsilon
            除非已经epsilon_min还小，否则比每次都乘以减少率epsilon_decay。
        '''
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_Q(self):
        '''
        Q网络学习完之后，需要把参数赋值到target_Q网络
        '''
        for i, target in zip(self.Q_network.trainable_weights, self.Target_Q_network.trainable_weights):
            target.assign(i)

    def remember(self, s, a, s_, r, done):
        '''
        把数据放入到队列中保存。
        '''
        data = (s, a, s_, r, done)
        self.memory.append(data)

    def process_data(self):

        # 从队列中，随机取出一个batch大小的数据。
        data = random.sample(self.memory, self.batch)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        y = self.Q_network(np.array(s, dtype='float32'))
        y = y.numpy()
        Q1 = self.Target_Q_network(np.array(s_, dtype='float32'))
        Q2 = self.Q_network(np.array(s_, dtype='float32'))
        next_action = np.argmax(Q2, axis=1)

        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target = r
            else:
                target = r + self.gamma * Q1[i][next_action[i]]
            target = np.array(target, dtype='float32')

            # y 就是更新目标。
            a_index = self.get_action_index(a)
            y[i][a_index] = target
        return s, y

    def get_action_index(self, a):
        for i in range(action_num):
            for j in range(len(action_array[0])):
                if j == len(action_array[0]) - 1 and math.isclose(a[j], action_array[i][j], rel_tol=0.000001):
                    return i
        return -1

    def update_Q_network(self):

        s, y = self.process_data()
        with tf.GradientTape() as tape:
            Q = self.Q_network(np.array(s, dtype='float32'))
            loss = tl.cost.mean_squared_error(Q, y)  # 最小化target和Q的距离
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.Q_network.trainable_weights))
        return loss

    def get_action(self, s):
        q = self.Q_network(np.array(s, dtype='float32').reshape([-1, 28]))
        a = np.argmax(q)
        return action_array[a]

    ## 开始训练
    def train(self, episode):
        step = 0
        reward = []
        all_reward = []
        all_loss = []
        for ep in range(episode):

            total_loss = []
            total_reward = 0
            loss = 0
            s = self.env.reset()  # 重置初始状态s

            self.env.CurrentTime = 0
            while True:

                # 进行游戏
                a = self.get_action(s)
                s_, r, done = self.env.step(a)
                total_reward += r
                step += 1
                reward.append(r)
                # 保存s, a, s_, r, done
                self.remember(s, a, s_, r, done)
                s = s_

                # 如果数据足够，那么就开始更新
                if len(self.memory) > self.batch:
                    loss = self.update_Q_network()
                    total_loss.append(loss.numpy())
                    if (step + 1) % 5 == 0:
                        self.update_epsilon()
                        self.update_target_Q()

                # 如果到最终状态，就打印一下成绩如何
                if done:
                    print('EP:%i,  total_rewards:%f,   epsilon:%f, loss:%f' % (
                        ep, np.mean(reward), self.epsilon, np.mean(loss)))
                    all_loss.append(np.mean(loss))
                    all_reward.append(np.mean(reward))
                    break
        tl.files.save_npz(self.Q_network.all_weights, name='./resource_allocation_Q.npz')
        tl.files.save_npz(self.Target_Q_network.all_weights, name='./resource_allocation_Target_Q.npz')
        return './resource_allocation_Q.npz', './resource_allocation_Target_Q.npz'

class HTTPSend:
    def send_get(self, url, path, data):
        conn = http.client.HTTPConnection(url)
        conn.request("GET", path)
        r1 = conn.getresponse()
        print(r1.status, r1.reason)

        data1 = r1.read()
        print(data1)  #
        conn.close()

    def send_post(self, url, path, data, header):
        conn = http.client.HTTPConnection(url)
        conn.request("POST", path, data, header)
        r1 = conn.getresponse()
        print(r1.status, r1.reason)

        data1 = r1.read()
        print(data1)  #
        conn.close()

    def send_head(self, url, path, data, header):
        conn = http.client.HTTPConnection(url)
        conn.request("HEAD", path, data, header)
        r1 = conn.getresponse()
        print(r1.status, r1.reason)
        data1 = r1.headers  #
        print(data1)  #
        conn.close()

    def send_put(self, url, path, filedata, header):
        conn = http.client.HTTPConnection(url)
        conn.request("PUT", path, filedata, header)
        r1 = conn.getresponse()
        print(r1.status, r1.reason)

        data1 = r1.read()  #
        print(data1)
        conn.close()


# 开始运行游戏
if __name__ == '__main__':

    httpSend = HTTPSend()
    url = "10.16.200.113:9012"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}

    ddqn = Double_DQN()
    file1, file2 = ddqn.train(5)
    fileMap = {'my post data': str(file1)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)
    tfile = open(file1, mode='rb')
    filedatas = tfile.read()
    fileheaders = {"Content-type": "text/plain", "Accept": "text/plain", \
                   "content-length": str(len(filedatas))}
    httpSend.send_put(url, path="/index", filedata=filedatas, header=fileheaders)

    fileMap = {'my post data': str(file2)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)
    tfile = open(file2, mode='rb')
    filedatas = tfile.read()
    fileheaders = {"Content-type": "text/plain", "Accept": "text/plain", \
                   "content-length": str(len(filedatas))}
    httpSend.send_put(url, path="/index", filedata=filedatas, header=fileheaders)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))




