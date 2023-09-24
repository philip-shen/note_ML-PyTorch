'''
pytorch_advanced/1_image_classification/1-3_transfer_learning.ipynb

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/1_image_classification/1-3_transfer_learning.ipynb
'''
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys, time
import pathlib
import urllib.request
import zipfile
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

sys.path.append('./libs')

import dataloader_image_classification 
from logger_setup import *
import lib_misc

strabspath=os.path.abspath(sys.argv[0])
strdirname=os.path.dirname(strabspath)
str_split=os.path.split(strdirname)
prevdirname=str_split[0]
dirnamelog=os.path.join(strdirname,"logs")
dirname_test_wav= os.path.join(strdirname,"test_wav")
dirnametest= os.path.join(strdirname,"test")

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def data_download():
    home = os.path.expanduser("~")

    # フォルダ「data」が存在しない場合は作成する
    data_dir =  pathlib.Path(f'{home}/projects')#"./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # ImageNetのclass_indexをダウンロードする
    # Kerasで用意されているものです
    # https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py

    
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    save_path = os.path.join(data_dir, "imagenet_class_index.json")

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    
    # 1.3節で使用するアリとハチの画像データをダウンロードし解凍します
    # PyTorchのチュートリアルで用意されているものです
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    save_path = os.path.join(data_dir, "hymenoptera_data.zip")

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

        # ZIPファイルを読み込み
        zip = zipfile.ZipFile(save_path)
        zip.extractall(data_dir)  # ZIPを解凍
        zip.close()  # ZIPファイルをクローズ

        # ZIPファイルを消去
        os.remove(save_path)

'''
ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with LibreSSL 2.8.3

https://stackoverflow.com/questions/76187256/importerror-urllib3-v2-0-only-supports-openssl-1-1-1-currently-the-ssl-modu

pip install urllib3==1.26.6
'''
def preview_img(image_file_path):
    # 訓練時の画像前処理の動作を確認
    # 実行するたびに処理結果の画像が変わる

    # 1. 画像読み込み
    #image_file_path = './data/goldenretriever-3724972_640.jpg'
    img = Image.open(image_file_path)   # [高さ][幅][色RGB]

    # 2. 元の画像の表示
    #plt.imshow(img)
    #plt.show()

    # 3. 画像の前処理と処理済み画像の表示
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = dataloader_image_classification.ImageTransform(size, mean, std)
    img_transformed = transform(img, phase="train")  # torch.Size([3, 224, 224])

    # (色、高さ、幅)を (高さ、幅、色)に変換し、0-1に値を制限して表示
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    #plt.imshow(img_transformed)
    
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=({"xticks":(), "yticks":()}))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img);
    ax[1].imshow(img_transformed);    

    plt.tight_layout()
    plt.show()

def make_dataset(opt_verbose='OFF'):
    home = os.path.expanduser("~")    
    rootpath= f'{home}/infinicloud/hymenoptera_data/'#f'{home}/projects/hymenoptera_data/',

    if not os.path.isdir(rootpath):
        raise Exception(f"rootpath: {rootpath} not found!")
    
    # 実行
    train_list = dataloader_image_classification.make_datapath_list(rootpath,phase="train")
    val_list = dataloader_image_classification.make_datapath_list(rootpath,phase="val")
    if opt_verbose.lower() == 'on':
        logger.info(f'len of train_list: {len(train_list) }')
        logger.info(f'len of val_list: {len(val_list)}')

    # 実行
    train_dataset = dataloader_image_classification.HymenopteraDataset(
                                    file_list=train_list, 
                                    transform=dataloader_image_classification.ImageTransform(size, mean, std), 
                                    phase='train')

    val_dataset = dataloader_image_classification.HymenopteraDataset(
                                    file_list=val_list, 
                                    transform=dataloader_image_classification.ImageTransform(size, mean, std), 
                                    phase='val')

    # 動作確認
    if opt_verbose.lower() == 'on':
        index = 242
        logger.info(f'train_dataset.__len__(): {train_dataset.__len__()}')
        logger.info(f'\ntrain_dataset.__getitem__({index})[1]: {train_dataset.__getitem__(index)[1]}')
        logger.info(f'\ntrain_dataset.__getitem__({index})[0].size(): {train_dataset.__getitem__(index)[0].size()}')
    
        index = 0
        logger.info(f'val_dataset.__len__(): {val_dataset.__len__()}')
        logger.info(f'\nval_dataset.__getitem__({index})[1]: {val_dataset.__getitem__(index)[1]}')
        logger.info(f'\nval_dataset.__getitem__({index})[0].size(): {val_dataset.__getitem__(index)[0].size()}')

    return train_dataset, val_dataset

def make_dataloader(opt_verbose='OFF'):
    
    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 動作確認
    batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
    inputs, labels = next(
        batch_iterator)  # 1番目の要素を取り出す
    
    if opt_verbose.lower() == 'on':
        logger.info(f'inputs.size(): {inputs.size()}')
        logger.info(f'labels: {labels}')
        #logger.info(f'size of batch_iterator: {batch_iterator.}')

    return dataloaders_dict
'''
The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. 
You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
'''
def make_model():
    # 学習済みのVGG-16モデルをロード
    # VGG-16モデルのインスタンスを生成
    #use_pretrained = True  # 学習済みのパラメータを使用
    #models.VGG16_Weights.DEFAULT
    net = models.vgg16(weights = None)

    # VGG16の最後の出力層の出力ユニットをアリとハチの2つに付け替える
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # 訓練モードに設定
    net.train()

    logger.info('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

    return net

def update_model(net):
    # 転移学習で学習させるパラメータを、変数params_to_updateに格納する
    params_to_update = []

    # 学習させるパラメータ名
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]

    # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            #print(name)
            logger.info(f'update_param_names: {name}')

        else:
            param.requires_grad = False

    # params_to_updateの中身を確認
    #print("-----------")
    logger.info(f'params_to_update: {params_to_update}')

    return net, params_to_update

# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # epochのループ
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        #print('-------------')

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測
                    
  
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)  
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                record_train_loss.append(epoch_loss)
                record_train_acc.append(epoch_acc)
            else:
                record_val_loss.append(epoch_loss)
                record_val_acc.append(epoch_acc)

    return record_train_loss, record_train_acc, record_val_loss, record_val_acc

def plotdiagrm_loss(rec_train, rec_val):
    plt.plot(range(1, len(rec_train)+1, 1), rec_train, label="Train")
    plt.plot(range(1, len(rec_val)+1, 1), rec_val, label="Val")
    plt.legend()

    plt.title("Error Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

def plotdiagrm_acc(rec_train, rec_val):
    plt.plot(1, range(len(rec_train)+1, 1), rec_train, label="Train")
    plt.plot(1, range(len(rec_val)+1, 1), rec_val, label="Val")
    plt.legend()

    plt.title("Acc Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.show()

def plotdiagrm_loss_acc(rec_loss_train, rec_loss_val, rec_acc_train, rec_acc_val):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
   
    ax[0].plot(range(1, len(rec_loss_train)+1, 1), rec_loss_train, label="Train")
    ax[0].plot(range(1, len(rec_loss_val)+1, 1), rec_loss_val, label="Val")
    ax[0].set_title("Error")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Error rate")
    #ax[0].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax[0].legend()
    
    ax[1].plot(range(1, len(rec_acc_train)+1, 1), rec_acc_train, label="Train")
    ax[1].plot(range(1, len(rec_acc_val)+1, 1), rec_acc_val, label="Val")
    ax[1].set_title("Acc")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Acc rate")
    #ax[1].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax[1].legend()
    
    plt.tight_layout()    
    plt.show()

# 入力画像の前処理のクラス
class BaseTransform():
    """
    画像のサイズをリサイズし、色を標準化する。

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
            transforms.CenterCrop(resize),  # 画像中央をresize × resizeで切り取り
            transforms.ToTensor(),  # Torchテンソルに変換
            transforms.Normalize(mean, std)  # 色情報の標準化
        ])

    def __call__(self, img):
        return self.base_transform(img)
        
'''
The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. 
You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
'''
def load_vgg16():
    # 学習済みのVGG-16モデルをロード
    # 初めて実行する際は、学習済みパラメータをダウンロードするため、実行に時間がかかります

    # VGG-16モデルのインスタンスを生成
    #use_pretrained = True  # 学習済みのパラメータを使用
    net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    net.eval()  # 推論モードに設定

    # モデルのネットワーク構成を出力
    #print(net)
    logger.info(f'\nnet: {net}')

    return net 

# 出力結果からラベルを予測する後処理クラス
class ILSVRCPredictor():
    """
    ILSVRCデータに対するモデルの出力からラベルを求める。

    Attributes
    ----------
    class_index : dictionary
            クラスindexとラベル名を対応させた辞書型変数。
    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        """
        確率最大のILSVRCのラベル名を取得する。

        Parameters
        ----------
        out : torch.Size([1, 1000])
            Netからの出力。

        Returns
        -------
        predicted_label_name : str
            最も予測確率が高いラベルの名前
        """
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

def predict_by_ILSVRC():
    net = load_vgg16()

    # ILSVRCのラベル情報をロードし辞意書型変数を生成します
    ILSVRC_class_index = json.load(open('./libs/imagenet_class_index.json', 'r'))

    # ILSVRCPredictorのインスタンスを生成します
    predictor = ILSVRCPredictor(ILSVRC_class_index)

    # 入力画像を読み込む
    image_file_path = './data/goldenretriever-3724972_640.jpg'
    img = Image.open(image_file_path)  # [高さ][幅][色RGB]

    # 前処理の後、バッチサイズの次元を追加する
    transform = BaseTransform(resize, mean, std)  # 前処理クラス作成
    img_transformed = transform(img)  # torch.Size([3, 224, 224])
    inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])

    # モデルに入力し、モデル出力をラベルに変換する
    out = net(inputs)  # torch.Size([1, 1000])
    result = predictor.predict_max(out)

    # 予測結果を出力する
    logger.info(f"入力画像の予測結果： {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_21-1_image_classification')
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
    
    #preview_img(image_file_path = './data/goldenretriever-3724972_640.jpg')
    
    json_data = json.load(json_path_file.open())

    # 3. 画像の前処理と処理済み画像の表示
    size, resize = 224, 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # 
    predict_by_ILSVRC()

    train_dataset, val_dataset = make_dataset(opt_verbose)

    # ミニバッチのサイズを指定
    batch_size = json_data["Config"][1]["batch_size"];#32
    dataloaders_dict = make_dataloader(opt_verbose)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    net = make_model()
    update_net, params_to_update = update_model(net)

    # 最適化手法の設定
    optimizer = optim.SGD(params=params_to_update, 
                          lr = json_data["Config"][1]["learning_rate"], 
                          momentum = json_data["Config"][1]["momentum"])

    # loss and acc のログ
    record_train_loss, record_train_acc = [], []
    record_val_loss, record_val_acc = [], []
    
    # 学習・検証を実行する
    num_epochs = json_data["Config"][1]["num_epochs"];#

    record_train_loss, record_train_acc, record_val_loss, record_val_acc = \
        train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    
    #plotdiagrm_acc(record_train_acc, record_val_acc)
    #plotdiagrm_loss(record_train_loss, record_val_loss)
    plotdiagrm_loss_acc(record_train_acc, record_val_acc,
                        record_train_loss, record_val_loss)
    
    time_consumption, h, m, s= lib_misc.format_time(time.time() - t0) 
    msg = 'Time consumption: {}.'
    logger.info(msg.format(time_consumption ))