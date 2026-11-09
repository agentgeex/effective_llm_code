import os
import numpy as np


# IOU计算函数，表征两个检测框的重合度
def iou_cal(RecA, RecB):
    xA = max(RecA[0], RecB[0])
    yA = max(RecA[1], RecB[1])
    xB = min(RecA[2], RecB[2])
    yB = min(RecA[3], RecB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    RecA_Area = (RecA[2] - RecA[0] + 1) * (RecA[3] - RecA[1] + 1)
    RecB_Area = (RecB[2] - RecB[0] + 1) * (RecB[3] - RecB[1] + 1)
    iou = interArea / float(RecA_Area + RecB_Area - interArea)
    return iou


# 检测框格式转换，中心点转角点
def yolo_coords_to_corner(coords):
    centerx, centery, w, h = coords
    right = int(centerx + 0.5 * w)
    left = int(max(0, centerx - 0.5 * w))
    top = int(max(0, centery - 0.5 * h))
    bottom = int(centery + 0.5 * h)
    result = [left, top, right, bottom]
    return result


# 根据框的大小，返回权重。越大的框，权重越高。
def area_to_weight(area, is_fp):
    if is_fp:
        if area > 100 ** 2:
            return 1
        elif area > 50 ** 2:
            return 0.9
        elif area > 30 ** 2:
            return 0.8
        elif area > 20 ** 2:
            return 0.7
        elif area > 10 ** 2:
            return 0.2
        elif area > 0.1:
            return 0
    else:
        if area > 100 ** 2:
            return 1
        if area > 80 ** 2:
            return 0.9
        if area > 50 ** 2:
            return 0.1
        if area > 30 ** 2:
            return 0.1
        if area > 0:
            return 0.1


# # 标注txt文件所在的目录
# label_path = "./obj_train_data/"
#
# # 算法识别txt文件所在的目录
# result_path = "./result/"
#
# # 文件列表
# labeled_txt_list = os.listdir(label_path)
# result_txt_list = os.listdir(result_path)

# # TP、FP、FN 指标
# TP2 = [0 for i in range(7)]
# FP2 = [0 for i in range(7)]
# FN2 = [0 for i in range(7)]
# FN4 = [0 for i in range(7)]
#
# # 遍历所有标注txt文件
# for label_file in labeled_txt_list:
#     with open(label_path + label_file, "r") as file:
#         if label_file in result_txt_list:
#             with open(result_path + label_file, "r") as r_file:
#                 label_boxes = []
#                 result_boxes = []
#                 line = True
#                 # 读标注txt文件，每一行代表一个标注框
#                 while line:
#                     line = file.readline()
#                     line2 = line.strip().split(" ")
#                     if len(line2) == 5:
#                         line2[0] = int(line2[0])
#                         line2[1] = int(float(line2[1]) * 1920)
#                         line2[3] = int(float(line2[3]) * 1920)
#                         line2[2] = int(float(line2[2]) * 1080)
#                         line2[4] = int(float(line2[4]) * 1080)
#                         label_boxes.append(line2)
#                 line = True
#                 # 读算法识别txt文件，每一行代表一个检测框
#                 while line:
#                     line = r_file.readline()
#                     line2 = line.strip().split(" ")
#                     if len(line2) == 5:
#                         line2[0] = int(line2[0])
#                         line2[1] = int(float(line2[1]) * 1920)
#                         line2[3] = int(float(line2[3]) * 1920)
#                         line2[2] = int(float(line2[2]) * 1080)
#                         line2[4] = int(float(line2[4]) * 1080)
#                         result_boxes.append(line2)
#
#                 # 遍历所有标注框，搜索匹配对应的检测框
#                 for label_box in label_boxes:
#                     found = False
#                     area = label_box[3] * label_box[4]
#                     # 浮板框的IOU大于0.5时，即认为匹配成功
#                     if label_box[0] == 6:
#                         weight = area_to_weight(area, is_fp=True)
#                         iou_st = 0.5
#                     # 其他标签框的IOU大于0.7时，即认为匹配成功
#                     else:
#                         weight = area_to_weight(area, is_fp=False)
#                         iou_st = 0.7
#                     # 搜索匹配对应的检测框
#                     for r_box in result_boxes:
#                         xyxy_lb = yolo_coords_to_corner(label_box[1:])
#                         xyxy_rb = yolo_coords_to_corner(r_box[1:])
#                         iou = iou_cal(xyxy_lb, xyxy_rb)
#                         if iou > iou_st:
#                             found = True
#                             if label_box[0] == r_box[0]:
#                                 # 标签正确的，增加TP指标
#                                 TP2[label_box[0]] += weight
#                             else:
#                                 # 标签错误的，增加FP和FN指标
#                                 FP2[r_box[0]] += weight
#                                 FN2[label_box[0]] += weight
#                             break
#                             # 如果标注框未搜索到对应的检测框，增加FN指标（相对于识别错误，没有找到目标的情况下对识别精度的影响力较小）
#                     if not found:
#                         if label_box[0] == 6:
#                             FN4[label_box[0]] += weight * 0.1
#                         else:
#                             FN4[label_box[0]] += weight * 0.8
#
# # 准确率Acc = TP/(TP+FP+FN)
# #  sum(FN2)
# print("TP=",TP2)
# print("FP2",FP2)
# print("FN2",FN2)
# print("FN4",FN4)
# acc = sum(TP2) / (sum(TP2) +sum(FN2)  + sum(FN4))
# print("acc = ", acc)
# acc1 = sum(TP2) / (sum(TP2) + sum(FN4))
# print("acc1 =", acc1)
#



