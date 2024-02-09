import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from build_data.import_dataset import get_dataset_obj
from solver.structured_solver import StructuredSolver

import pdb


parser = argparse.ArgumentParser(description="")

parser.add_argument('-f', '--filename', type=str, help="file name for training")
parser.add_argument('-m', '--modelname', type=str, help="model name of training")

args = parser.parse_args()


def main():

    data_obj = get_dataset_obj(args.filename)
    data = data_obj.data

    target_name = "Quality"
    feature_names = [col for col in data.columns if col != target_name]

    # 訓練データとテストデータに分ける
    X_train, X_test, y_train, y_test = train_test_split(data[feature_names], data[target_name], test_size=0.2, random_state=42, stratify=data[target_name])
    train_data = pd.DataFrame(X_train, columns=feature_names)
    train_data[target_name] = y_train
    test_data = pd.DataFrame(X_test, columns=feature_names)
    test_data[target_name] = y_test
    # 訓練データを訓練データと検証データに分ける
    X_train, X_valid, y_train, y_valid = train_test_split(train_data[feature_names], train_data[target_name], test_size=0.2, random_state=42, stratify=train_data[target_name])
    train_data = pd.DataFrame(X_train, columns=feature_names)
    train_data[target_name] = y_train
    valid_data = pd.DataFrame(X_valid, columns=feature_names)
    valid_data[target_name] = y_valid

    solver = StructuredSolver(
        data_name=data_obj.data_name,
        feature_names=feature_names,
        target_name=target_name,
        predict_task="Classification",
        model_name=args.modelname,
        epoch_num=10,
        batch_size=32,
        lr=0.1
        )
    solver.train(train_data=train_data, valid_data=valid_data)

if __name__ == "__main__":
    main()