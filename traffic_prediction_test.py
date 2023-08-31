# coding=utf-8
import http.client
from urllib import request, parse
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
import datetime, pytz
import matplotlib.pyplot as plt
import time

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

def GRU_Test(dataPath, modlePath):
    # Importing dataset
    model = keras.models.load_model(modlePath)

    train = pd.read_csv(dataPath)
    data = train.iloc[0, :].values

    X_test = data[1: len(data) - 1]
    y_test = data[2: len(data)]

    sc = MinMaxScaler()
    inputs = X_test
    inputs = np.reshape(inputs, (-1, 1))
    inputs = sc.fit_transform(inputs)
    inputs = getWindowData(inputs, 3)

    '''y_pred = np.zeros(shape=(899, 1))
    for i in range(len(inputs)):
        temp = inputs[i]
        temp = np.reshape(temp, newshape=(1,1,1))
        start = time.perf_counter()
        y_pred[i] = model.predict(temp)
        end = time.perf_counter()
        print('Running time: %s Seconds' % (end - start))'''

    start = time.perf_counter()
    y_pred = model.predict(inputs)
    y_pred = sc.inverse_transform(y_pred)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))


    plt.figure(figsize=(10, 5))
    plt.plot(range(np.array(len(y_test))), y_test, "blue")
    plt.plot(range(np.array(len(y_pred))), y_pred[:, 0], "red")
    plt.legend(['true', 'prediction'])
    plt.xlabel('time')
    plt.ylabel('traffic')
    plt.show()

    return y_pred

if __name__ == '__main__':

    httpSend = HTTPSend()

    url = "localhost:9006"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')

    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    file = GRU_Test('C:/Users/jiani/Desktop/pcap1/log.csv', 'traffic_prediction_model.h5')
    fileMap = {'my post data': str(file)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))



