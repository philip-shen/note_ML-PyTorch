'''
    pytorch_advanced/2_objectdetection/2-8_SSD_inference.ipynb

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/2-8_SSD_inference.ipynb

    
    pytorch_advanced/2_objectdetection/2-8_SSD_inference_appendix.ipynb

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/2-8_SSD_inference_appendix.ipynb
'''
import cv2  # OpenCVライブラリ
import matplotlib.pyplot as plt 
import numpy as np
import torch
import os, sys , time
import json, argparse, pathlib

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

from libs.ssd_model import SSD
from libs.ssd_model import DataTransform

# 画像に対する予測
from libs.ssd_predict_show import SSDPredictShow
from libs.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn
from libs.ssd_model import SSD

def est_timer(start_time):
    time_consumption, h, m, s= lib_misc.format_time(time.time() - start_time)         
    msg = 'Time Consumption: {}.'.format( time_consumption)#msg = 'Time duration: {:.2f} seconds.'
    logger.info(msg)


                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_22-5_SSD_inference')
    parser.add_argument('--conf', type=str, default='config.json', help='Config json')
    args = parser.parse_args()

    t0 = time.time()
    local_time = time.localtime(t0)
    msg = 'Start Time is {}/{}/{} {}:{}:{}'

    logger_set(strdirname)
    opt_verbose = 'On'
    json_file= args.conf
    json_path_file = pathlib.Path(strdirname)/json_file
    
    if (not os.path.isfile(json_file))  :
        raise Exception( f'Please check json file: {json_file}  if exist!!! ')
    json_data = json.load(json_path_file.open())

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス： {device}")
    
    # ファイルパスのリストを作成
    home = os.path.expanduser("~")    
    rootpath= f'{home}/infinicloud/{json_data["Config"][2]["rootpath_trainval"]}/'
    if not os.path.isdir(rootpath):
        raise Exception(f"rootpath: {rootpath} not found!")
    
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    # SSD300の設定
    ssd_cfg = {
        'num_classes': 21,  # 背景クラスを含めた合計クラス数
        'input_size': 300,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # SSDネットワークモデル
    net = SSD(phase="inference", cfg=ssd_cfg)

    path_weights = f'{home}/infinicloud/weights/ssd300_50.pth'
    # SSDの学習済みの重みを設定
    if os.path.isfile(path_weights):
        path_weights = path_weights
    else:
        path_weights = './weights/ssd300_50.pth'
        
    #net_weights = torch.load(path_weights, map_location={'cuda:0': 'cpu'})        
    net_weights = torch.load(path_weights, map_location = device)        

    #net_weights = torch.load(f'{home}/infinicloud/weights/ssd300_mAP_77.43_v2.pth',
    #                         map_location={'cuda:0': 'cpu'})

    net.load_state_dict(net_weights)

    logger.info('ネットワーク設定完了：学習済みの重みをロードしました')

    # 1. 画像読み込み
    image_file_path = "./images/cowboy-757575_640.jpg"
    img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
    height, width, channels = img.shape  # 画像のサイズを取得

    # 2. 元画像の表示
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # 3. 前処理クラスの作成
    color_mean = (104, 117, 123)  # (BGR)の色の平均値
    input_size = 300  # 画像のinputサイズを300×300にする
    transform = DataTransform(input_size, color_mean)

    # 4. 前処理
    phase = "val"
    img_transformed, boxes, labels = transform(
        img, phase, "", "")  # アノテーションはないので、""にする
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)
    logger.info(f'img: {img}')
    
    # 5. SSDで予測
    net.eval()  # ネットワークを推論モードへ

    '''
    img_x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
    logger.info(f'img_x: {img_x}')
    detections = net(img_x)

    #print(detections.shape)
    #print(detections)
    logger.info(f'detections.shape: {detections.shape}')
    logger.info(f'detections: {detections}')
    '''

    # output : torch.Size([batch_num, 21, 200, 5])
    #  =（batch_num、クラス、confのtop200、規格化されたBBoxの情報）
    #   規格化されたBBoxの情報（確信度、xmin, ymin, xmax, ymax）
    
    # 予測と、予測結果を画像で描画する
    ssd = SSDPredictShow(eval_categories=voc_classes, net=net)
    ssd.show(image_file_path, data_confidence_level=0.6)
    
    est_timer(t0)
