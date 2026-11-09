import base64
import hashlib
import json
import os
import random
import time
import requests
from urllib import request

proxies = {
    'https': 'http://127.0.0.1:8001',  # v2只提供了http的端口，这里https也是用http代理
    'http': 'http://127.0.0.1:8001'
}

# url1='https://www.google.com'
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36 Edg/98.0.1108.42'
# }
# proxies = {
#     'https': 'http://127.0.0.1:8001',  # v2只提供了http的端口，这里https也是用http代理
#     'http': 'http://127.0.0.1:8001'
# }
# res=requests.get(url = url1,proxies = proxies,headers=headers)


class HttpUtils(object):
    def __init__(self):
        self.client_ip = ''

    @staticmethod
    def get_md5(md5str):
        m1 = hashlib.md5()
        m1.update(md5str.encode("utf-8"))
        return m1.hexdigest()

    @staticmethod
    def get_filename(path):
        return os.path.basename(path)

    @staticmethod
    def to_request_for_code(url, method='POST', request_param=None):
        try:
            print(url)
            print(request_param)
            response = requests.request(method, url=url, data=request_param, params=request_param, timeout=15)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            print(repr(e))
            return False, repr(e)

    @staticmethod
    def get(url, request_param=None, timeout=30, headers=None):
        try:
            response = requests.get(url=url, params=request_param, timeout=timeout, headers=headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    @staticmethod
    def delete(url, request_param=None, timeout=30, headers=None):
        try:
            response = requests.delete(url=url, params=request_param, timeout=timeout, headers=headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    @staticmethod
    def post_form(url, request_param=None, timeout=30, headers=None):
        try:
            post_headers = {"Content-Type": "application/x-www-form-urlencoded"}
            if None is not headers:
                post_headers.update(headers)
            response = requests.post(url=url, data=request_param, timeout=timeout, headers=post_headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    @staticmethod
    def post_json(url, request_param=None, timeout=2400000, headers=None):
        try:
            post_headers = {"Content-Type": "application/x-www-form-urlencoded","Connection":"keep-alive"}

            # heads = ([('Host', None),
            #                     ('Connection', 'keep-alive'),
            #                     ('Upgrade-Insecure-Requests', '1'),
            #                     ('User-Agent',
            #                     'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.78 Safari/537.36'),
            #                     ('Accept',
            #                     'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
            #                     ('Accept-Language', 'en-US,en;q=0.9'),
            #                     ('Accept-Encoding', 'gzip, deflate')])

            if None is not headers:
                post_headers.update(headers)
            response = requests.post(url=url, json=request_param, timeout=timeout, headers=post_headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    @staticmethod
    def post_files(url, files=None, request_param=None, timeout=30, headers=None):
        """
        :param url:
        :param files: files={"myfile1": open('D:\全力以富\1.docx', 'rb'),"myfile2": open('D:\全力以富\2docx', 'rb')})
        :param request_param:
        :param timeout:
        :param headers:
        :return:

        """
        try:
            post_headers = {}
            # post_headers = {"Content-Type": "multipart/form-data"}
            if None is not headers:
                post_headers.update(headers)
            response = requests.post(url=url, files=files, data=request_param, timeout=timeout, headers=post_headers, verify=False)
            return json.loads(response.text)
            # return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return repr(e)



    @staticmethod
    def post_xml(url, request_param=None, timeout=30, headers=None):
        try:
            post_headers = {"Content-Type": "text/xml; charset=utf-8",
                            "SOAPAction": "http://WebXml.com.cn/getSupportCityString"},

            #             request_param_demo = '''<?xml version="1.0" encoding="utf-8"?>
            # <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            #   <soap:Body>
            #     <getSupportCityString xmlns="http://WebXml.com.cn/">
            #       <theRegionCode>string</theRegionCode>
            #     </getSupportCityString>
            #   </soap:Body>
            # </soap:Envelope>'''
            if None is not headers:
                post_headers.update(headers)
            response = requests.post(url=url, data=request_param, timeout=timeout, headers=post_headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    @staticmethod
    def put_form(url, request_param=None, timeout=30, headers=None):
        try:
            post_headers = {"Content-Type": "application/x-www-form-urlencoded"}
            if None is not headers:
                post_headers.update(headers)
            response = requests.put(url=url, data=request_param, timeout=timeout, headers=post_headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    @staticmethod
    def put_json(url, request_param=None, timeout=30, headers=None):
        try:
            post_headers = {"Content-Type": "application/json"}
            if None is not headers:
                post_headers.update(headers)
            response = requests.put(url=url, json=request_param, timeout=timeout, headers=post_headers)
            return (True, response.json()) if response.status_code == 200 else (False, repr(response))
        except Exception as e:
            return False, repr(e)

    def get_real_ip(self):
        """
        获取公网ip
        :return:
        """
        try:
            if self.client_ip == '':
                b_ip = request.urlopen('http://ip.42.pl/raw').read()
                self.client_ip = str(b_ip, 'utf-8')
        except:
            pass
        try:
            if self.client_ip == '':
                self.client_ip = requests.get('http://www.jsonip.com').json().get('ip', '')
        except:
            pass
        try:
            if self.client_ip == '':
                self.client_ip = requests.get('http://caiwu.leesrobots.com:8384/ip.php').json().get('ip', '')
        except:
            pass
        return self.client_ip
