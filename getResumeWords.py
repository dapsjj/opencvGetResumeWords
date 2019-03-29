#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import re
import pandas as pd


img_root = r'root'+os.sep #原始文件路径
save_img_dir = r'save_img'+os.sep #要保存的文件路径
suffix = r'.jpg' #图片后缀
image_format = r'%04d' #要保存的图片的格式化的名称

min_width = 15 #轮廓最小宽度
min_height = 15 #轮廓最小高度
pytesseract.pytesseract.tesseract_cmd = r"E:\tesseract-ocr\tesseract.exe"

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
current_file_path = os.path.join(os.path.dirname(__file__)) #当前文件的路径
all_txt_list = []

name_df =  pd.read_csv( current_file_path + os.sep +'Chinese_Family_Name.csv' ) #关于姓氏的DataFrame
all_name = name_df.NameB.values.tolist() #所有姓氏

native_df =  pd.read_csv( current_file_path + os.sep +'jiguan.csv' ) #关于籍贯的DataFrame
all_native_province = native_df.JG4.values.tolist() #所有籍贯的省份
all_native_city = native_df.JG5.values.tolist() #所有籍贯的城市

major_df =  pd.read_csv( current_file_path + os.sep +'zhuanye.csv' ) #关于专业的DataFrame
all_major = major_df.major.values.tolist() #所有专业


def save_resume_txt():
    count_num = 1 #要保存的图片计数
    root_image = os.listdir(img_root)

    for name in root_image:
        just_image_name = name #保存简历图片用的名字
        name = current_file_path + os.sep + img_root + name
        readImg = cv2.imread(name) #读图片
        # readImg[np.where((readImg >= [128, 150, 162]).all(axis=2))] = [255, 255, 255]  # 原图被改变
        current_person_small_img_list = []
        current_person_word_list = []

        if readImg is not None:  # 判断图片是否读入
            HSV = cv2.cvtColor(readImg, cv2.COLOR_BGR2HSV)  # 把BGR图像转换为HSV格式
            Lower = np.array([26, 43, 46])  # 要识别颜色的下限
            Upper = np.array([34, 255, 255])  # 要识别的颜色的上限
            # mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
            mask = cv2.inRange(HSV, Lower, Upper)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

            # 将滤波后的图像变成二值图像放在binary中
            ret, binary = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)

            # 在binary中发现轮廓，轮廓按照面积从小到大排列
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not os.path.exists(save_img_dir):  # 文件夹不存在则创建
                os.makedirs(save_img_dir)

            for contour in contours:  # 遍历所有的轮廓
                x, y, w, h = cv2.boundingRect(contour)  # 将轮廓分解为识别对象的左上角坐标和宽、高
                if w < min_width or h < min_height:
                    continue
                rect = cv2.minAreaRect(contour)  # 最小外接矩形
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                left =box[1][0]#最小外接矩形的左上角
                top =box[1][1]#最小外接矩形的左上角
                width = box[2][0]-box[1][0]#最小外接矩形的宽
                height = box[0][1]-box[1][1]#最小外接矩形的高
                try:
                    part_of_src_img = readImg[y - 2:y + h + 2, x - 2:x + w + 2]
                    # part_of_src_img = readImg[top - 2:top + width + 2, left - 2:left + width + 2]
                    part_of_src_img[np.where((part_of_src_img>=[0,100,100]).all(axis=2))] = [255,255,255] #原图被改变
                    deal_img = contrast_brightness(part_of_src_img, 1.5, 5)
                    gray = cv2.cvtColor(deal_img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
                    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    h,w = binary.shape #获取高和宽
                    mul = 1.5
                    resize_img = cv2.resize(binary, (int(w * mul), int(h * mul)))
                    resize_img = cv2.bilateralFilter(resize_img, 5, 105, 105)  # 双边滤波
                    resize_img = cv2.morphologyEx(resize_img, cv2.MORPH_OPEN, kernel1)

                    # cv2.imwrite(current_file_path + os.sep + save_img_dir  + image_format % count_num + suffix, resize_img)  # 保存图片指定区域

                    current_person_small_img_list.append(resize_img)
                    count_num += 1
                except Exception as ex:
                    continue

            for img in current_person_small_img_list:
                key_word_list = pytesseract.image_to_string(img, lang='chi_sim').split('\n')
                for key_word in key_word_list:
                    if key_word:
                        current_person_word_list.append(key_word)
            #获取人名,如果能识别到人名则copy一份原图,命名成'人名.jpg',识别不到人名,则仍然以原图片命名
            rename_current_people_image , nameFlag = getImageNameInResume(just_image_name,current_person_word_list)
            # print(rename_current_people_image , nameFlag)

            copy_img_name = current_file_path + os.sep + save_img_dir + rename_current_people_image
            copy_img_name = copy_img_name.replace("\\","/") #在windows上使用os.sep时，变成'\\'，所以要替换一下，不然这个路径的图片打不开
            current_person_word_list.append([copy_img_name,nameFlag])
            copyReadImg = cv2.imread(name)  # 读图片
            cv2.imencode(suffix, copyReadImg)[1].tofile(copy_img_name) #中文图片名称处理
            current_person_word_list_no_need_last_column = []
            for item in current_person_word_list[0:-1]:
                current_person_word_list_no_need_last_column.append(removeBlank(item))
            current_person_word_list_no_need_last_column.append(current_person_word_list[-1]) #最后一列是列表,因此没办法去掉空白,所以单独append
            all_txt_list.append(current_person_word_list_no_need_last_column)

    return all_txt_list




def contrast_brightness(image,c,b):
    '''
    #增加对比度
    :param image: 要增加对比对的图片
    :param c: 对比度
    :param b: 亮度
    :return: 增强会后的图片
    '''
    blank=np.zeros_like(image,image.dtype)
    dst=cv2.addWeighted(image,c,blank,1-c,b)
    return dst


def getPhoneLength(paraStr):
    '''
    返回数字类型的字符串的长度
    :param paraStr: 含有数字的字符串
    :return: 符串中数字的个数
    '''
    list_number = re.findall(r'\d', paraStr)
    return len(list_number)


def getImageNameInResume( paraNameOfOriginalImage,paraThisPeopleWordslist ):
    '''
    获取简历中的人名
    :param paraNameOfOriginalImage: 当前循环的人的原始图片名称
    :param paraThisPeopleWordslist: 当前循环的人的words的list
    :return: 原本的照片名或者人名.jpg
    '''

    if paraNameOfOriginalImage and paraThisPeopleWordslist: #如果图片名字和关键词列表不是空
        for word in paraThisPeopleWordslist:
            if len(word)<5: #名字长度小于5
                for first_name in all_name: #循环姓氏列表
                    if word.startswith(first_name):
                        return word + suffix,'haveName' #能找到姓氏就返回名字.jpg
        return paraNameOfOriginalImage,'noName' #找不到姓氏就返回原图名字
    else:#如果图片名字或关键词列表是空,因为图片名字一定不是空,那么就是paraThisPeopleWordslist是空
        return paraNameOfOriginalImage, 'noName'  # 找不到姓氏就返回原图名字



def setWordsToCSV(paraAllTxtList):
    '''
    图片信息保存到csv,生成的csv的格式是姓名,籍贯,专业,学校,电话,原始简历图片在服务器的地址
    :param paraAllTxtList: 要处理的列表
    :return: 无
    '''
    print(paraAllTxtList)


def generateWordsList(paraAllTxtList):
    '''
    生成列表的格式是姓名,籍贯,专业,学校,电话,原始简历图片在服务器的地址
    :param paraAllTxtList: 要处理的列表
    :return: 要写进csv的列表
    '''
    pass


def removeBlank(MyString):
    '''
    :param MyString: 要替换空白的字符串
    :return: 去掉空白后的字符串
    '''
    try:
        MyString = re.sub('[\s+]', '', MyString)
        return MyString
    except Exception as ex:
        raise ex



if __name__ == '__main__':
    all_txt_list = save_resume_txt()
    generateWordsList(all_txt_list)
    setWordsToCSV(all_txt_list)









