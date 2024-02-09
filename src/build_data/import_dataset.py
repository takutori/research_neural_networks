from dataclasses import dataclass
import pandas as pd


def get_dataset_obj(filename: str):
    """
    データに対応するデータクラスをインスタンス化して取得する。

    Parameters
    ----------
    filename : str
        importしたいデータのファイル名

    Returns
    -------
    data_dict[filename] : dataclasss
    """
    data_dict = {
        "apple_quality" : AppleQuality()
    }

    return data_dict[filename]

@dataclass
class AppleQuality:
    """
    apple qualityのデータセットの表すクラス
    """
    data: pd.DataFrame

    def __init__(self):
        self.data_name = "apple_quality"
        file_path = "data/raw/" + self.data_name + ".csv"
        self.data = pd.read_csv(file_path)
        self.data = self.data.iloc[:len(self.data)-1]
        self.encording_target_label()

    def encording_target_label(self):
        """
        保持しているデータセットの目的変数をエンコーディングする。
        """
        self.target_label = {
            0 : "bad",
            1 : "good"
        }

        self.data.replace({"bad" : 0, "good" : 1}, inplace=True)

