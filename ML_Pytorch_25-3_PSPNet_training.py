'''
pytorch_advanced/3_semantic_segmentation/3-7_PSPNet_training.ipynb

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/3_semantic_segmentation/3-7_PSPNet_training.ipynb

本ファイルでは、PSPNetの学習と検証の実施を行います。AWSのGPUマシンで計算します。
p2.xlargeで約12時間かかります。
'''

# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from libs.dataloader import make_datapath_list, DataTransform, VOCDataset
from nets.pspnet_Ogawa import PSPNet

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

# 初期設定
# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def est_timer(start_time):
    time_consumption, h, m, s= lib_misc.format_time(time.time() - start_time)         
    msg = 'Time Consumption: {}.'.format( time_consumption)#msg = 'Time duration: {:.2f} seconds.'
    logger.info(msg)

def make_dataloader(rootpath):
    # ファイルパスリスト作成
    #rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath=rootpath)

    # Dataset作成
    # (RGB)の色の平均値と標準偏差
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=475, color_mean=color_mean, color_std=color_std))

    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=475, color_mean=color_mean, color_std=color_std))

    # DataLoader作成
    batch_size = 8

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict

def make_network_model(weights):
    # ファインチューニングでPSPNetを作成
    # ADE20Kデータセットの学習済みモデルを使用、ADE20Kはクラス数が150です
    net = PSPNet(n_classes=150)

    # ADE20K学習済みパラメータをロード
    state_dict = torch.load(weights)
    net.load_state_dict(state_dict)

    # 分類用の畳み込み層を、出力数21のものにつけかえる
    n_classes = 21
    net.decode_feature.classification = nn.Conv2d(
        in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    net.aux.classification = nn.Conv2d(
        in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    # 付け替えた畳み込み層を初期化する。活性化関数がシグモイド関数なのでXavierを使用する。

    return net

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)
# 損失関数の設定
class PSPLoss(nn.Module):
    """PSPNetの損失関数のクラスです。"""

    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight  # aux_lossの重み

    def forward(self, outputs, targets):
        """
        損失関数の計算。

        Parameters
        ----------
        outputs : PSPNetの出力(tuple)
            (output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))。

        targets : [num_batch, 475, 475]
            正解のアノテーション情報

        Returns
        -------
        loss : テンソル
            損失の値
        """

        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss+self.aux_weight*loss_aux

# スケジューラーの設定
def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow((1-epoch/max_epoch), 0.9)

# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Torch using device: {device}" )

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # multiple minibatch
    batch_multiplier = 3

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        logger.info('-------------')
        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        logger.info('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                scheduler.step()  # 最適化schedulerの更新
                optimizer.zero_grad()
                logger.info('（train）')

            else:
                if((epoch+1) % 5 == 0):
                    net.eval()   # モデルを検証モードに
                    logger.info('-------------')
                    logger.info('（val）')
                else:
                    # 検証は5回に1回だけ行う
                    continue

            # データローダーからminibatchずつ取り出すループ
            count = 0  # multiple minibatch
            for imges, anno_class_imges in dataloaders_dict[phase]:
                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
                # issue #186より不要なのでコメントアウト
                # if imges.size()[0] == 1:
                #     continue

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)
                anno_class_imges = anno_class_imges.to(device)

                
                # multiple minibatchでのパラメータの更新
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imges)
                    loss = criterion(
                        outputs, anno_class_imges.long()) / batch_multiplier

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()  # 勾配の計算
                        count -= 1  # multiple minibatch

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            logger.info('iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size*batch_multiplier, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        logger.info('-------------')
        logger.info('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        logger.info('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss /
                     num_train_imgs, 'val_loss': epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

    # 最後のネットワークを保存する
    torch.save(net.state_dict(), 'weights/pspnet50_' +
               str(epoch+1) + '.pth')
    logger.info(f'output weights/pspnet50_{str(epoch+1)}.pth')    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML_Pytorch_25-3_PSPNet_training')
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

    dataloaders_dict = make_dataloader(rootpath)

    weights = "./weights/pspnet50_ADE20K.pth"
    net = make_network_model(weights)
    net.decode_feature.classification.apply(weights_init)
    net.aux.classification.apply(weights_init)

    logger.info('ネットワーク設定完了：学習済みの重みをロードしました')

    criterion = PSPLoss(aux_weight=0.4)

    # ファインチューニングなので、学習率は小さく
    optimizer = optim.SGD([
        {'params': net.feature_conv.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': net.decode_feature.parameters(), 'lr': 1e-2},
        {'params': net.aux.parameters(), 'lr': 1e-2},
    ], momentum = json_data["Config"][8]["momentum"], 
        weight_decay = json_data["Config"][8]["weight_decay"])

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    # 学習・検証を実行する
    num_epochs = json_data["Config"][8]["num_epochs"]
    train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=num_epochs)

    est_timer(t0)