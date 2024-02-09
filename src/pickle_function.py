import pickle

def dumpPickle(file_name: str, obj):
    """
    Pythonのオブジェクトを保存する。

    Parameters
    ----------
    file_name : str
        オブジェクトの保存先
    obj : _type_
        保存したいオブジェクト
    """
    with open(file_name, mode="wb") as f:
        pickle.dump(obj, f)

def loadPickle(file_name: str):
    """
    pickleファイルをオブジェクトとして読み込む

    Parameters
    ----------
    file_name : str
        読み込みたいpickleファイルの名前

    Returns
    -------
    obj : object
        読み込んだオブジェクト
    """
    with open(file_name, mode="rb") as f:
        return pickle.load(f)