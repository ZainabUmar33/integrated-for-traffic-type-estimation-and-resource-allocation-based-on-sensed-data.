# coding=utf-8
import struct
import time, datetime
import pytz
import numpy as np
import binascii
import os
import pandas as pd
import http.client
from urllib import request, parse

tabel1 = 1
tabel2 = 2
tabel3 = 3


def time_trans(GMTtime):
    # print(GMTtime)
    timeArray = time.localtime(GMTtime)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    return otherStyleTime  # 2013--10--10 23:40:00


class pcap_packet_header:
    def __init__(self):
        self.GMTtime = b'\x00\x00'
        self.MicroTime = b'\x00\x00'
        self.caplen = b'\x00\x00'
        self.lens = b'\x00\x00'
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

def pcap_packet_data(file_name):
    image = []
    start = time.perf_counter()
    fpcap = open(file_name, 'rb')
    # ftxt = open('result.txt', 'w')

    string_data = fpcap.read()
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))

    # pcap文件的数据包解析
    step = 0
    packet_num = 0
    packet_data = []

    pcap_packet_header_list = []
    i = 24

    start = time.perf_counter()
    while (i < len(string_data)):
        # 数据包头各个字段bytes
        GMTtime = string_data[i:i + 4]
        MicroTime = string_data[i + 4:i + 8]
        caplen = string_data[i + 8:i + 12]
        lens = string_data[i + 12:i + 16]
        # 数据包各个字段的正常表示
        packet_GMTtime = struct.unpack('I', GMTtime)[0]
        packet_GMTtime = time_trans(packet_GMTtime)
        packet_MicroTime = struct.unpack('I', MicroTime)[0]
        packet_caplen = struct.unpack('I', caplen)[0]
        packet_len = struct.unpack('I', lens)[0]
        # 数据包头对象
        head = pcap_packet_header()
        head.GMTtime = packet_GMTtime
        head.MicroTime = packet_MicroTime
        head.caplen = packet_caplen
        head.lens = packet_len

        # print(head.MicroTime)
        # print(packet_MicroTime)
        pcap_packet_header_list.append(head)
        # print(packet_len)
        # 写入此包数据
        packet_data.append(binascii.b2a_hex(string_data[i + 16:i + 16 + packet_len]))
        i = i + packet_len + 16
        packet_num += 1
        # a=input()

    for i in range(len(packet_data)):
        tempImage = []
        for j in range(int(len(packet_data[i]) / 2)):
            num = str(packet_data[i][j * 2: j * 2 + 2])
            num = num[2:4]
            tempImage.append(int(num, 16))
        image.append(tempImage)

    fpcap.close()
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    return image

def saveImages(filePath):
    files = os.listdir(filePath)
    dataSetTemp = []

    n = 0
    for file in files:
        if not os.path.isdir(file):
            image = pcap_packet_data(filePath + '/' + file)

            data = []
            i = 0
            while i + 28 < len(image):
                scipyImage = np.zeros(shape=(28, 28))
                for j in range(28):
                    for m in range(28):
                        scipyImage[j][m] = image[i + j][10 + m]
                data.append(scipyImage.reshape(28 * 28))
                i = i + 28
            data = np.array(data)
            if n == 0:
                data = np.insert(data, 0, values=tabel1, axis=1)
            if n == 1:
                data = np.insert(data, 0, values=tabel2, axis=1)
            if n == 2:
                data = np.insert(data, 0, values=tabel3, axis=1)
            dataSetTemp.append(data)
            n = n + 1

    for i in range(len(dataSetTemp)):
        dataSetTemp[i] = np.array(dataSetTemp[i])

    temp = dataSetTemp[0]
    for i in range(len(dataSetTemp) - 1):
        temp = np.vstack((temp, dataSetTemp[i + 1]))

    np.random.shuffle(temp)
    dataSet = pd.DataFrame(temp).to_csv('dataSet.csv')
    return 'dataSet.csv'

if __name__ == '__main__':


    httpSend = HTTPSend()

    url = "10.16.200.110:9011"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')

    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}

    filePath = 'pcap'
    file = saveImages(filePath)
    fileMap = {'my post data': str(file)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)
    tfile = open(file, mode='rb')
    filedatas = tfile.read()
    fileheaders = {"Content-type": "text/plain", "Accept": "text/plain", \
                   "content-length": str(len(filedatas))}
    httpSend.send_put(url, path="/index", filedata=filedatas, header=fileheaders)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))


