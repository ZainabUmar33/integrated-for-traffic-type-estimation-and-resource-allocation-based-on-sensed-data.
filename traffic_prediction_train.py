# coding=utf-8
import http.client
from urllib import request, parse
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
import datetime, pytz

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
def getWindowData(fullData, n = 10):
    dataSet = []
    i = n
    while i < len(fullData):
        dataSet.append(fullData[i - n: i])
        i = i + 1
    return np.reshape(np.array(dataSet), (len(fullData) - n, n, 1))

def GRU_Train(dataPath):

    train = pd.read_csv(dataPath)
    data = train.iloc[0, :].values

    X_train = data[1 : len(data) - 1]
    y_train = data[2 : len(data)]

    sc = MinMaxScaler()
    X_train = np.reshape(X_train, (-1, 1))
    y_train = np.reshape(y_train, (-1, 1))
    X_train = sc.fit_transform(X_train)
    y_train = sc.fit_transform(y_train)
    X_train = getWindowData(X_train, 3)
    y_train = getWindowData(y_train, 3)

    model = Sequential()
    model.add(GRU(units=8, activation='relu', input_shape=(None, 1)))
    model.add(Dense(units=1))
    model.summary()

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, batch_size=10, epochs=100)
    model.save('traffic_prediction_model.h5')
    return 'traffic_prediction_model.h5'

if __name__ == '__main__':

    httpSend = HTTPSend()

    url = "localhost:9006"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')

    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    file = GRU_Train('C:/Users/jiani/Desktop/pcap1/log.csv')
    fileMap = {'my post data': str(file)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)
    tfile = open(file, mode='rb')
    filedatas = tfile.read()
    fileheaders = {"Content-type": "text/plain", "Accept": "text/plain", \
                   "content-length": str(len(filedatas))}
    httpSend.send_put(url, path="/index", filedata=filedatas, header=fileheaders)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))