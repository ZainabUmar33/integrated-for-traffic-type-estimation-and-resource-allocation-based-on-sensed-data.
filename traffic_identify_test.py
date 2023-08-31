# coding=utf-8
import http.client
from urllib import request, parse
import tensorflow as tf
import numpy as np
import pandas as pd
import time, datetime
import pytz

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
        start = time.perf_counter()
        conn.request("POST", path, data, header)
        r1 = conn.getresponse()
        end = time.perf_counter()
        print('run time ----%ssecond'% (end - start))
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

def CNN_Test(dataPath, modelPath):
    model = tf.keras.models.load_model(modelPath)

    start = time.perf_counter()
    test = pd.read_csv(dataPath)
    data = test.to_numpy()
    image = np.array(data[:, 2: len(data[0])])
    label = data[:, 1]

    image = image.astype('float32') / 255
    label = tf.keras.utils.to_categorical(label, 3)

    score = model.predict(image)
    result = np.zeros(shape=(len(image), 2))
    for i in range(len(score)):
        result[i][0] = np.argmax(score[i])
        result[i][1] = label[i][1]

    return result

if __name__ == '__main__':

    httpSend = HTTPSend()

    url = "localhost:9001"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')

    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    file = CNN_Test('dataSet.csv', './traffic_identify_model.h5')
    fileMap = {'my post data': str(file)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))
