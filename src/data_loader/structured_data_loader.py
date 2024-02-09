from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import pdb

class StructuredDataLoader(DataLoader):
    """
    構造化データに関するデータローダーを扱うクラス
    """
    def __init__(self, data: pd.DataFrame, feature_names: str, target_name: str):
        self.feature_names = feature_names
        self.target_name = target_name
        self.features = data[feature_names].values
        self.target = data[target_name].values

    def __len__(self):
        """
        データ全体のサイズ
        """

        return len(self.features)

    def __getitem__(self, idx: int):
        """
        一回のサンプルで返す値

        Parameters
        ----------
        idx : int
            データのindex

        Returns
        -------
        features_i : np.array
            一回のサンプルで返す特徴量
        target_i : np.array
            一回のサンプルで返すラベル
        """
        # DataLoaderは特定のサポートされている型で返す必要がある。
        features_i = torch.tensor(np.float32(self.features[idx]))
        target_i = torch.tensor(self.target[idx]).to(torch.int64)

        return features_i, target_i






def get_data_loader(data:pd.DataFrame,
                    feature_names: List,
                    target_name: str,
                    loader_type: str,
                    mode: str,
                    batch_size: int
                    ) -> torch.utils.data.dataloader.DataLoader:
    """
    data_loaderを取得する。

    Parameters
    ----------
    data : pd.DataFrame
        データ全体
    feature_names : List
        特徴量名のリスト
    target_name : str
        目的変数の名前
    loader_type : str
        データの種類。今のところ、Sttucturedのみ。
    mode : str
        "train", "valid", "test"
    batch_size : int
        バッチサイズ

    Returns
    -------
    data_loader : torch.utils.data.dataloader.DataLoader
        _description_
    """
    if loader_type == "Structured":
        dataset = StructuredDataLoader(data, feature_names, target_name)
    else:
        print(loader_type + " is not defined")

    if mode != "train":
        shuffle = False
    else:
        shuffle = True

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return data_loader