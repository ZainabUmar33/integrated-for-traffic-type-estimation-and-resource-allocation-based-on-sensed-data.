import time
import numpy as np
import http.client
from urllib import request, parse
import pandas as pd
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

def saveCSV(logPath):
    traffic = np.loadtxt(logPath, delimiter=',')
    traffic = traffic.reshape(1, len(traffic))
    result = np.zeros(shape=(1, len(traffic[0]) - 1))

    for i in range(len(result[0])):
        result[0][i] = traffic[0][i + 1] - traffic[0][i]
    dataSet = pd.DataFrame(result).to_csv('log.csv')
    return 'log.csv'

if __name__ == '__main__':

    httpSend = HTTPSend()

    url = "10.16.200.110:9010"
    data = {
        'my post data': 'I am client , hello world',
    }
    datas = parse.urlencode(data).encode('utf-8')
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    file = saveCSV('log.txt')
    fileMap = {'my post data': str(file)}
    filetemp = parse.urlencode(fileMap).encode('utf-8')
    httpSend.send_post(url, "/index", filetemp, headers)

    tfile = open(file, mode='rb')
    filedatas = tfile.read()
    fileheaders = {"Content-type": "text/plain", "Accept": "text/plain", \
                   "content-length": str(len(filedatas))}
    httpSend.send_put(url, path="/index", filedata=filedatas, header=fileheaders)

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')))


