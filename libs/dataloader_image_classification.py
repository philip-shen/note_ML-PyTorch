import glob
import os.path as osp
import torch.utils.data as data
from torchvision import models, transforms
from PIL import Image
from logger_setup import *


class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """
    """
    DAY16：Pytorch transforms（上） 
    https://ithelp.ithome.com.tw/articles/10275928

    隨機長寬比裁剪：transforms.RandomResizedCrop

    隨機大小及隨機長寬比裁剪原始圖片，最後再resize到設定好的size。
        size：要剪裁的圖片大小。
        scale：例如scale=(0.2, 1.0)，則會隨機從0.2到1.0中，選一個倍數裁剪，如0.2則裁剪原圖的0.2倍。
        ratio：長寬比區間，隨機選取。
    """
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),  # データオーギュメンテーション
                transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)


def make_datapath_list(rootpath, phase="train", opt_verbose='OFF'):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------
    phase : 'train' or 'val'
        訓練データか検証データかを指定する

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    #rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    #print(target_path)
    logger.info(f'target_path: {target_path}')

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        if opt_verbose.lower() == 'on':
            logger.info(f'path: {path}')
        path_list.append(path)

    return path_list


class HymenopteraDataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        '''
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]
        '''
        '''
        INFO: target_path: /home/philphoenix/projects/hymenoptera_data/train/**/*.jpg
        INFO: path: /home/philphoenix/projects/hymenoptera_data/train/ants/226951206_d6bf946504.jpg
        
        INFO: target_path: /home/philphoenix/projects/hymenoptera_data/val/**/*.jpg
        INFO: path: /home/philphoenix/projects/hymenoptera_data/val/ants/1073564163_225a64f170.jpg
        '''
        label = img_path.split('/')[-2].lower()

        # ラベルを数値に変更する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label
