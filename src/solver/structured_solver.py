from typing import List

import pandas as pd
import torch
import torch.nn as nn

import time

from data_loader.structured_data_loader import get_data_loader
from models.mlp import MLP

from pickle_function import dumpPickle, loadPickle

import pdb

def get_model_class(model_name: str):
    model_class_dict = {
        "MLP" : MLP
    }
    model_class = model_class_dict[model_name]

    return model_class

def get_loss(predict_task: str):
    loss_dict = {
        "Classification" : nn.CrossEntropyLoss(),
        "Regression" : nn.MSELoss()
    }

    criterion = loss_dict[predict_task]

    return criterion


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, training_start_time: int, data_name: str, model_name:str, patience: int=10, verbose: bool=False):
        self.data_name = data_name
        self.model_name = model_name
        self.epoch = 0 # 監視中のエポック数のカウンターを初期
        self.best_valid_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience
        self.verbose = verbose
        self.training_start_time = training_start_time

        self.early_stop = False

    def __call__(self, model, valid_loss):
        if self.best_valid_loss < valid_loss: # 前エポックの損失より大きくなった場合
            self.epoch += 1 # カウンターを1増やす

            if self.epoch > self.patience: # 監視回数の上限に達した場合
                if self.verbose:  # 早期終了のフラグが1の場合
                    print('early stopping')
                self.early_stop = True

        else: # 前エポックの損失以下の場合
            self.epoch = 0 # カウンターを0に戻す
            self.best_valid_loss = valid_loss # 損失の値を更新する
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        model_save_path = "model_checkpoints/"
        dumpPickle(model_save_path + "data_name=" + self.data_name + "_model_name=" + self.model_name + "_" + str(int(self.training_start_time)) + ".pickle", model)
        print("saving model...")

class StructuredSolver:
    """
    構造化データのためのソルバー。モデルの最適化、保存を行う。
    """
    def __init__(self,
                 data_name: str,
                 feature_names: List,
                 target_name: str,
                 predict_task: str = "Classification",
                 model_name: str="MLP",
                 epoch_num: int = 5,
                 batch_size: int = 256,
                 lr: float=0.1
                 ):
        self.data_name = data_name
        self.feature_names = feature_names
        self.target_name = target_name
        self.predict_task = predict_task
        self.model_name = model_name
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr

        self.build_model()


    def build_model(self):
        self.training_start_time = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # モデル
        model_class = get_model_class(self.model_name)
        self.model = model_class(
            feature_names=self.feature_names
            ).to(device)
        # 最適化ソルバー
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # 損失関数
        self.criterion = get_loss(self.predict_task)


    def train(self, train_data, valid_data):
        print("============================= TRAIN MODE ===============================")
        self.training_start_time = time.time()

        train_loader = get_data_loader(
            data=train_data,
            feature_names=self.feature_names,
            target_name=self.target_name,
            loader_type="Structured",
            mode="train",
            batch_size=self.batch_size
            )

        valid_loader = get_data_loader(
            data=valid_data,
            feature_names=self.feature_names,
            target_name=self.target_name,
            loader_type="Structured",
            mode="valid",
            batch_size=self.batch_size
            )

        self.model.train_loss_list = []
        self.model.valid_loss_list = []

        early_stopping = EarlyStopping(
            training_start_time=self.training_start_time,
            model_name=self.model_name,
            data_name=self.data_name,
            patience=3,
            verbose=True
            )
        for epoch in range(self.epoch_num):
            train_loss, valid_loss = 0, 0

            self.model.train()
            for i, (X, y) in enumerate(train_loader):
                self.optimizer.zero_grad() # 勾配をリセット
                outputs = self.model(X) # 順伝播の計算
                loss = self.criterion(outputs, y) # lossの計算
                train_loss += loss.item() # train_lossに結果を蓄積
                loss.backward() # 誤差逆伝播
                self.optimizer.step() # 重みの計算
            mean_train_loss = train_loss / len(train_loader.dataset)

            self.model.eval()
            with torch.no_grad(): # この間は勾配の計算ストップ（計算時間短縮）
                for i, (X, y) in enumerate(valid_loader):
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    valid_loss += loss.item()
            mean_valid_loss = valid_loss / len(valid_loader.dataset)

            # lossのlogの表示
            print(
                "Epoch [{}/{}] train loss:{train_loss:.4f} valid loss:{valid_loss:.4f}".format(
                    epoch+1, self.epoch_num, train_loss=mean_train_loss, valid_loss=mean_valid_loss
                    )
                )

            self.model.train_loss_list.append(mean_train_loss)
            self.model.valid_loss_list.append(mean_valid_loss)

            early_stopping(self.model, mean_valid_loss)

