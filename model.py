#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, sys
import pandas as ps
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score

#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def train_model(data, seed = None):
    np.random.seed(seed)
    data = data.iloc[np.random.permutation(len(data))]
    pos_data = data[data["similar"] == 1.0]
    neg_data = data[data["similar"] == 0.0][:len(pos_data)]
    data = ps.concat([pos_data, neg_data])
    print("Размер выборки: {}".format(len(data)))

    X = data.drop(["id1", "id2", "similar"], axis=1)
    y = data["similar"]
    
    model = LogisticRegression(penalty='l1', tol=0.05)
    model.fit(X, y)
    return model

def validate_model(model, data):
    X = data.drop(["id1", "id2", "similar"], axis=1)
    y = data["similar"]
    p = model.predict(X)

    score = (p == y).mean()
    res = ps.DataFrame({"y": y, "p": p}, index=None)
    true_pos = res[res["y"] == 1]["p"].mean()
    true_neg = 1 - res[res["y"] == 0]["p"].mean()

    print("Попадания: {}".format(score))
    print("True positives: {}".format(true_pos))
    print("True negatives: {}".format(true_neg))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="загрузить модель из файла", metavar="FILE")
    parser.add_argument("-t", "--train", help="обучить модель на данных", metavar="FILE")
    parser.add_argument("-o", "--output", help="записать модель в файл", metavar="FILE")
    parser.add_argument("-v", "--validate", help="проверить производительность модели на выборке", metavar="FILE")
    parser.add_argument("-p", "--predict", help="предсказать значения для выборки", nargs=2, metavar=("INFILE", "OUTFILE"))
    parser.add_argument("-r", "--random", help="семя генератора псевдослучайных чисел", metavar="SEED", type=int)
    args = parser.parse_args()

    if args.input:
        model = joblib.load(args.input)
    elif args.train:
        print("Читаю данные из файла...", end=' ')
        sys.stdout.flush()
        data = ps.read_csv(args.train)
        print("ОК.")
        model = train_model(data, args.random)
    else:
        return print("Не указано, откуда брать модель (ключ -i или -t)")

    if args.validate:
        test = ps.read_csv(args.validate)
        validate_model(model, test)

    if args.predict:
        infile, outfile = args.predict
        to_predict = ps.read_csv(infile)
        predicted = predict_values(model, to_predict)
        predicted.to_csv(outfile)

    if args.output:
        joblib.dump(model)

if __name__ == "__main__":
    main()