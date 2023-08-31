# coding=utf-8
import http.client
from urllib import request, parse
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Reshape, Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
import pandas as pd
import torch
import torchvision
from torchsummary import summary
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

def CNN_Train(dataPath):

    train = pd.read_csv(dataPath)
    data = train.to_numpy()
    image = np.array(data[:, 2: len(data[0])])
    label = data[:, 1]

    image = image.astype('float32') /255
    label = tf.keras.utils.to_categorical(label, 3)

    model = Sequential()
    model.add(Reshape((28,28,1), input_shape=(784,)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    #开始训练
    model.fit(image, label, batch_size=64, epochs=12, verbose=1, validation_data=(image, label))

    model.save('traffic_identify_model.h5')

    device = torch.device('cpu')
    model = torchvision.models.vgg11_bn().to(device)
    summary(model, input_size=(3, 224, 224))
    return 'traffic_identify_model.h5'

if __name__ == '__main__':

    httpSend = HTTPSend()

    url = "10.16.200.112:9008"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')

    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    file = CNN_Train('dataSet.csv')
    fileMap = {'my post data': str(file)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)
    tfile = open(file, mode='rb')
    filedatas = tfile.read()
    fileheaders = {"Content-type": "text/plain", "Accept": "text/plain", \
                   "content-length": str(len(filedatas))}
    httpSend.send_put(url, path="/index", filedata=filedatas, header=fileheaders)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))
