'''
    ssd-pytorch/voc_annotation.py

https://github.com/bubbliiiing/ssd-pytorch/blob/master/voc_annotation.py
'''
import os,time, sys
import random
import xml.etree.ElementTree as ET

import numpy as np
import json, pathlib, argparse

from libs.utils import voc_labels#get_classes

sys.path.append('./libs')
from logger_setup import *
import lib_misc

strabspath=os.path.abspath(sys.argv[0])
strdirname=os.path.dirname(strabspath)
str_split=os.path.split(strdirname)
prevdirname=str_split[0]
dirnamelog=os.path.join(strdirname,"logs")
dirname_test_wav= os.path.join(strdirname,"test_wav")
dirnametest= os.path.join(strdirname,"test")

def est_timer(start_time):
    time_consumption, h, m, s= lib_misc.format_time(time.time() - start_time)         
    msg = 'Time Consumption: {}.'.format( time_consumption)#msg = 'Time duration: {:.2f} seconds.'
    logger.info(msg)


def convert_annotation(year, image_id, list_file):
    xml_file = os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id))
    logger.info(f"xml_file: {xml_file}")
    
    in_file = open(xml_file, encoding='utf-8')

    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), 
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML_Pytorch_23-4_SSD_voc_annotation')
    parser.add_argument('--conf', type=str, default='config.json', help='Config json')
    args = parser.parse_args()
    logger_set(strdirname)
    
    t0 = time.time()
    local_time = time.localtime(t0)
    msg = 'Start Time is {}/{}/{} {}:{}:{}'
    logger.info(msg.format( local_time.tm_year,local_time.tm_mon,local_time.tm_mday,\
                            local_time.tm_hour,local_time.tm_min,local_time.tm_sec))
    
    opt_verbose = 'On'
    json_file= args.conf
    json_path_file = pathlib.Path(strdirname)/json_file
    
    if (not os.path.isfile(json_file))  :
        raise Exception( f'Please check json file: {json_file}  if exist!!! ')
    json_data = json.load(json_path_file.open())

    #--------------------------------------------------------------------------------------------------------------------------------#
    #   annotation_mode用于指定该文件运行时计算的内容
    #   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
    #   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
    #   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
    #--------------------------------------------------------------------------------------------------------------------------------#
    annotation_mode     = json_data["Config"][6]["anno_mode"]#0
    #-------------------------------------------------------------------#
    #   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
    #   与训练和预测所用的classes_path一致即可
    #   如果生成的2007_train.txt里面没有目标信息
    #   那么就是因为classes没有设定正确
    #   仅在annotation_mode为0和2的时候有效
    #-------------------------------------------------------------------#
    #classes_path        = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------------------------------------------------#
    #   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
    #   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
    #   仅在annotation_mode为0和1的时候有效
    #--------------------------------------------------------------------------------------------------------------------------------#
    trainval_percent    = json_data["Config"][6]["trainval_percent"]#0.9
    train_percent       = json_data["Config"][6]["train_percent"]#0.9
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    home = os.path.expanduser("~")       
    VOCdevkit_path  = f'{home}/projects/VOCDetection/2007/trainval/VOCdevkit'
    VOCdevkit_test_path  = f'{home}/projects/VOCDetection/2007/test/VOCdevkit'
    VOCdevkit_webdav_path  = f'{home}/infinicloud/2007_trainval/VOCdevkit'

    VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
    classes      = [k for _, k in enumerate(voc_labels)]#get_classes(classes_path)

    #-------------------------------------------------------#
    #   统计目标数量
    #-------------------------------------------------------#
    photo_nums  = np.zeros(len(VOCdevkit_sets))
    nums        = np.zeros(len(classes))
    logger.info(f"photo_nums: {photo_nums}")
    logger.info(f"nums: {nums}")

    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        logger.info("Generate txt in ImageSets.")

        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        saveBasePath_test    = os.path.join(VOCdevkit_test_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml: 
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        #logger.info(f"list: {list}" )
        logger.info(f"trainval: {trainval}")
        logger.info(f"train: {train}")
        logger.info(f"train and val size: {tv}")
        logger.info(f"train size: {tr}")

        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath_test,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        logger.info("Generate txt in ImageSets done.")
    
    if annotation_mode == 0 or annotation_mode == 2:
        #logger.info("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        if os.path.isdir(VOCdevkit_webdav_path):
            imgsets_path = VOCdevkit_webdav_path 
        else:
            imgsets_path = VOCdevkit_path
        logger.info(f"imgsets_path: {imgsets_path}")

        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(imgsets_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        logger.info("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            logger.info("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            logger.info("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            logger.info("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            logger.info("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            logger.info("（重要的事情说三遍）。")
    
    est_timer(t0)