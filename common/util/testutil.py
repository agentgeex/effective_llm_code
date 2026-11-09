# coding: utf-8
# modify_response.py

import os
import re
from common.util.http_utils import HttpUtils
from common.util import utils
#import mitmproxy

url = "http://127.0.0.1:5000"


# 验证get请求
def testhttpget():
    test_get_parm = {'id': '123'}
    res = HttpUtils.get(url + "/test/get", test_get_parm)
    print(res)

#验证POST JSON 请求
def testhttppostjson():
    test_post_json_parm = {'id': 'aaa'}
    res = HttpUtils.post_json(url + "/test/post/json", test_post_json_parm)
    print(res)

def testimage2base64(filename):

    image_root_dir = "/path/to/your/project/poolguarder/train_sample/train_data/testcase/8.7_chengdu/obj_train_data"
    image_base64 = utils.image2Base64(f"{image_root_dir}/{filename}.jpg")

    # fileo = open(file, 'r')
    # filelines = fileo.readlines()
    # logger.debug(f" === 标签总个数   {len(filelines)} ")
    # fileo.close()

    parserfile = f"{image_root_dir}/{filename}.txt"
    file1 = open(parserfile, 'r')
    content1 = file1.read()


    with open(parserfile, "r") as fileo:
        filelines = fileo.readlines()

    print(f" 标注文件总目标数为：{len(filelines)} ")
    # for item in filelines:
    #     print(item)
    print(content1)
    # res conf_thres=0.2, iou_thres=0.2
    test_post_json_parm = {'model_id': '8.7_chengdu_m.onnx','b64':image_base64,'conf_thres':0.2,'iou_thres':0.2 }
    netresult, res = HttpUtils.post_json(os.environ.get("DETECT_API_URL", "http://<your-detect-service>/detect/"), test_post_json_parm)

    print(res)
    check_mark_list = res['result']
    print("\n\n")
    print(f" 模型识别出的目标什么为：  {len(check_mark_list)}")
    for item1 in check_mark_list:
        print(item1)

    print(res)




#testimage2base64("2023-08-08-15-36-44_6")
# TP2 =
#print( [0 for i in range(7)])

a = {'a':1,'b':2}
print(len(a))
