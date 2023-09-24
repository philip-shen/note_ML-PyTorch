''''
pytorch_advanced/1_image_classification/1-5_fine_tuning.ipynb

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/1_image_classification/1-5_fine_tuning.ipynb
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
def load_vgg16():
    # 学習済みのVGG-16モデルをロード
    # 初めて実行する際は、学習済みパラメータをダウンロードするため、実行に時間がかかります

    # VGG-16モデルのインスタンスを生成
    #use_pretrained = True  # 学習済みのパラメータを使用
    net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    net.eval()  # 推論モードに設定

    # モデルのネットワーク構成を出力
    logger.info(f'\nnet: {net}')

    return net 

'''
The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. 
You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
'''
def make_model():
    # 学習済みのVGG-16モデルをロード
    # VGG-16モデルのインスタンスを生成
    #use_pretrained = True  # 学習済みのパラメータを使用
    #models.VGG16_Weights.DEFAULT
    #net = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
    net = models.vgg16(weights = None)

    # VGG16の最後の出力層の出力ユニットをアリとハチの2つに付け替える
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # 訓練モードに設定
    net.train()

    logger.info('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

    return net

def update_model_params(net):
    # 学習させる層のパラメータ名を指定
    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight",
                            "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    # パラメータごとに各リストに格納する
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
            logger.info(f"params_to_update_1に格納： {name}")

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
            logger.info(f"params_to_update_2に格納： {name}")

        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
            logger.info(f"params_to_update_3に格納： {name}")

        else:
            param.requires_grad = False
            logger.info(f"勾配計算なし。学習しない： {name}")

    return net, params_to_update_1, params_to_update_2, params_to_update_3 

# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス： {device}")

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        #print('-------------')

        # epochごとの訓練と検証のループ
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

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

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

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示inputs
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                            phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                record_train_loss.append(epoch_loss)
                record_train_acc.append(epoch_acc.cpu())
            else:
                record_val_loss.append(epoch_loss)
                record_val_acc.append(epoch_acc.cpu())

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
    plt.plot(range(1, len(rec_train)+1, 1), rec_train, label="Train")
    plt.plot(range(1, len(rec_val)+1, 1), rec_val, label="Val")
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
    json_data = json.load(json_path_file.open())

    # 3. 画像の前処理と処理済み画像の表示
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset, val_dataset = make_dataset(opt_verbose)

    # ミニバッチのサイズを指定
    batch_size = json_data["Config"][1]["batch_size"];#32
    dataloaders_dict = make_dataloader(opt_verbose)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()
    #net = load_vgg16()
    net = make_model()

    # ファインチューニングで学習させるパラメータを、変数params_to_updateの1～3に格納する
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []
    
    net, params_to_update_1, params_to_update_2, params_to_update_3 = update_model_params(net)

    # モデルのネットワーク構成を出力
    logger.info(f'\nnet of update_model_params: {net}')

    # 最適化手法の設定
    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3}
        ], 
        momentum = json_data["Config"][1]["momentum"])

    # loss and acc のログ
    record_train_loss, record_train_acc = [], []
    record_val_loss, record_val_acc = [], []

    # 学習・検証を実行する
    num_epochs = json_data["Config"][1]["num_epochs"];#
    
    record_train_loss, record_train_acc, record_val_loss, record_val_acc = \
        train_model(net, 
                dataloaders_dict, 
                criterion, 
                optimizer, 
                num_epochs=json_data["Config"][1]["num_epochs"])

    logger.info(f'\nrecord_train_loss: {record_train_loss}')
    logger.info(f'\nrecord_train_acc: {record_train_acc}')
    logger.info(f'\nrecord_val_loss: {record_val_loss}')
    logger.info(f'\nrecord_val_acc: {record_val_acc}')
    '''    
    record_train_loss= [0.4290879941526264, 0.1299130065144335]
    record_train_acc= [0.7901, 0.9465]
    record_val_loss= [0.6217924761616327, 0.17848939132067113, 0.12634789237594293]
    record_val_acc= [0.6667, 0.9412, 0.9542]
    '''
    #plotdiagrm_acc(record_train_acc, record_val_acc)
    #plotdiagrm_loss(record_train_loss, record_val_loss)
    plotdiagrm_loss_acc(record_train_acc, record_val_acc,
                        record_train_loss, record_val_loss)
    
    # PyTorchのネットワークパラメータの保存
    save_path = './weights_fine_tuning.pth'
    torch.save(net.state_dict(), save_path)

    '''
    # PyTorchのネットワークパラメータのロード
    load_path = './weights_fine_tuning.pth'
    load_weights = torch.load(load_path)
    net.load_state_dict(load_weights)

    # GPU上で保存された重みをCPU上でロードする場合
    load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
    net.load_state_dict(load_weights)
    '''
    time_consumption, h, m, s= lib_misc.format_time(time.time() - t0) 
    msg = 'Time consumption: {}.'
    logger.info(msg.format(time_consumption ))