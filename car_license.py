import cv2
import os
import sys
import numpy as np
import tensorflow as tf
from my_cv.car_func import *

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']

if __name__ == '__main__':
    cur_dir = sys.path[0]
    car_plate_w,car_plate_h = 136,36   #车牌分类模型的输入
    char_w,char_h = 20,20              #字符分类模型的输入
    plate_model_path = os.path.join(cur_dir+'./model/plate_recongnize/model.ckpt-510.meta')
    char_model_path = os.path.join(cur_dir+'./model/char_recongnize/model.ckpt-720.meta')
    way='C:/Users/asus/Desktop/car_license/code/pictures/'+sys.argv[1]+'.jpg'
    img = cv2.imread(way)
    # 预处理
    pred_img = pre_process(img)
    
    # 车牌定位
    car_plate_list = locate_carPlate(img,pred_img,car_plate_w,car_plate_h)
    
    # CNN车牌过滤
    ret,car_plate = cnn_select_carPlate(car_plate_list,plate_model_path)
    if ret == False:
        print("未检测到车牌")
        sys.exit(-1)

    # 字符提取
    char_img_list = extract_char(car_plate,char_w,char_h)

    # CNN字符识别
    text = cnn_recongnize_char(char_img_list,char_model_path)
    number=len(text)
    text_1=' '
    for i in range(number):
        text_1=text_1+text[i]
        if i==1:
            text_1=text_1+' '
    print(text_1)
