'''
pytorch_advanced/3_semantic_segmentation/3-2_DataLoader.ipynb

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/3_semantic_segmentation/3-2_DataLoader.ipynb
'''

# パッケージのimport
import os.path as osp
from PIL import Image
import cv2

import torch.utils.data as data
# データ処理のクラスとデータオーギュメンテーションのクラスをimportする
from libs.data_augumentation import Compose_Ogawa, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor

import numpy as np
import matplotlib.pyplot as plt

import os, sys, time
import json, argparse, pathlib
import libs.utils

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

def make_datapath_list(rootpath, opt_verbose='OFF'):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose_Ogawa([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose_Ogawa([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)

class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        # 3. 前処理を実施
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img

def chk_train_imgs():
    # 実行するたびに変わります

    # 画像データの読み込み
    index = 0
    imges, anno_class_imges = train_dataset.__getitem__(index)

    # 画像の表示
    img_val = imges
    img_val = img_val.numpy().transpose((1, 2, 0))
    plt.imshow(img_val)
    plt.show()

    # アノテーション画像の表示
    anno_file_path = train_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
    p_palette = anno_class_img.getpalette()

    anno_class_img_val = anno_class_imges.numpy()
    anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
    anno_class_img_val.putpalette(p_palette)
    plt.imshow(anno_class_img_val)
    plt.show()

def chk_val_imgs():
    # 画像データの読み込み
    index = 0
    imges, anno_class_imges = val_dataset.__getitem__(index)

    # 画像の表示
    img_val = imges
    img_val = img_val.numpy().transpose((1, 2, 0))
    plt.imshow(img_val)
    plt.show()

    # アノテーション画像の表示
    anno_file_path = train_anno_list[0]
    anno_class_img = Image.open(anno_file_path)   # [高さ][幅][色RGB]
    p_palette = anno_class_img.getpalette()

    anno_class_img_val = anno_class_imges.numpy()
    anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
    anno_class_img_val.putpalette(p_palette)
    plt.imshow(anno_class_img_val)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_25-1_dataloader')
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

    # 動作確認 ファイルパスのリストを取得
    if sys.platform.lower() == 'linux':
        home = os.path.expanduser('~')
        VOCdevkit_path  = f'{home}/projects/VOCDetection/2012/trainval/VOCdevkit'
        VOCdevkit_test_path  = f'{home}/projects/VOCDetection/2012/test/VOCdevkit'
        VOCdevkit_webdav_path  = f'{home}/infinicloud/2012_trainval/VOCdevkit'
    elif sys.platform.lower() == 'win32':
        home = 'z:'
        VOCdevkit_webdav_path  = f'{home}/2012_trainval/VOCdevkit'

    if os.path.isdir(VOCdevkit_webdav_path): 
        rootpath = f"{VOCdevkit_webdav_path}/VOC2012/"
    else:
        rootpath = f"{VOCdevkit_path}/VOC2012/"

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath, opt_verbose)

    logger.info(f"len of train_img_list: {train_img_list.__len__()}")
    logger.info(f"len of train_anno_list: { train_anno_list.__len__()}")
    logger.info(f"len of val_img_list: {val_img_list.__len__()}")
    logger.info(f"len of val_anno_list: {val_anno_list.__len__()}")

    # 動作確認

    # DataTranfrom 動作の確認
    # 1. 画像読み込み
    image_file_path = train_img_list[-1]
    anno_file_path = train_anno_list[-1]
    img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
    img_anno = cv2.imread(anno_file_path) 
    height, width, channels = img.shape  # 画像のサイズを取得
    logger.info(f'\nimage_file_path: {image_file_path};\nanno_file_path: {anno_file_path}')

    # 3. 元画像の表示
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(img_anno, cv2.COLOR_BGR2RGB))
    plt.show()

    # (RGB)の色の平均値と標準偏差
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    input_size=475
    transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

    # train画像の表示
    '''
    phase = "train"
    img_transformed, img_anno_transformed = transform(
        phase, img, img_anno )
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()
    '''

    # データセット作成
    train_dataset = VOCDataset(train_img_list, train_anno_list, 
                               phase="train", 
                               transform=transform)

    val_dataset = VOCDataset(val_img_list, val_anno_list, 
                             phase="val", 
                             transform=transform)

    # データの取り出し例
    logger.info(f"train_dataset.__getitem__(0)[0].shape: {train_dataset.__getitem__(0)[0].shape}")
    logger.info(f"train_dataset.__getitem__(0)[1].shape: {train_dataset.__getitem__(0)[1].shape}")
    logger.info(f"train_dataset.__getitem__(0): {train_dataset.__getitem__(0)}")
    logger.info(f"val_dataset.__getitem__(0)[0].shape: {val_dataset.__getitem__(0)[0].shape}")
    logger.info(f"val_dataset.__getitem__(0)[1].shape: {val_dataset.__getitem__(0)[1].shape}")
    logger.info(f"val_dataset.__getitem__(0): {val_dataset.__getitem__(0)}")

    # データローダーの作成
    batch_size = 8

    train_dataloader = data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
                        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 動作の確認
    batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
    imges, anno_class_imges = next(batch_iterator)  # 1番目の要素を取り出す
    logger.info(imges.size())  # torch.Size([8, 3, 475, 475])
    logger.info(anno_class_imges.size())  # torch.Size([8, 3, 475, 475])
    
    est_timer(t0)