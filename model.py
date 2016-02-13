#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, sys, pickle
import pandas as ps
import numpy as np

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def all_models(seed = None):
    models = []
    for pen in ("l1", "l2"):
        for tol in range(1, 31):
            tol *= 0.01
            models.append({"model": LogisticRegression(penalty=pen, tol=tol, random_state=seed),
                "txt": "LogReg (pen: {}, tol: {})".format(pen, tol)})
    for C in range(1, 21):
        C *= 0.1
        for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
            models.append({"model": SVC(C=C, kernel=kernel, random_state=seed),
                "txt": "SVC (C: {}, kernel: {})".format(C, kernel)})
    for weights in ("uniform", "distance"):
        for n in range(1, 60, 3):
            models.append({"model": KNeighborsClassifier(n, weights),
                "txt": "KNeighbors (n: {}, wieghts: {})".format(n, weights)})
    for n in range(3, 61, 3):
        for depth in tuple(range(1, 8)) + (None,):
            for crit in ("gini", "entropy"):
                models.append({"model": RandomForestClassifier(n, crit, depth, random_state=seed),
                    "txt": "RFC (n: {}, crit: {}, depth: {})".format(
                        n, crit, depth)})
    return models

def compete_models(train, test, seed = None):
    pos_data = train[train["similar"] == 1.0]
    neg_data = train[train["similar"] == 0.0][:len(pos_data)]
    train = ps.concat([pos_data, neg_data])
    print("Размер учебной выборки: {}".format(len(train)))

    X1 = train.drop(["id1", "id2", "similar"], axis=1)
    y1 = train["similar"]
    X2 = test.drop(["id1", "id2", "similar"], axis=1)
    y2 = test["similar"]

    models = all_models(seed)
    for mdl in models:
        mdl["model"].fit(X1, y1)
        y = mdl["model"].predict(X2)
        
        mdl["score"] = (y == y2).mean()

        res = ps.DataFrame({"y": y2, "p": y}, index=None)
        mdl["true_pos"] = res[res["y"] == 1]["p"].mean()
        mdl["true_neg"] = 1 - res[res["y"] == 0]["p"].mean()

        print(mdl["txt"] + ": {:.4f}/{:.4f}/{:.4f}".format(
            mdl["score"], mdl["true_pos"], mdl["true_neg"]))

    models = [value for value in models if value["score"] > 0.8 and value["true_pos"] > 0.7]
    sc = max(models, key=lambda x: x["score"])
    pos = max(models, key=lambda x: x["true_pos"])
    neg = max(models, key=lambda x: x["true_neg"])
    bal = max(models, key=lambda x: (x["true_pos"]+x["true_neg"])/2)

    print("Best total score: {} ({})".format(sc["txt"], sc["score"]))
    print("Best true positives: {} ({})".format(pos["txt"], pos["true_pos"]))
    print("Best true negatives: {} ({})".format(neg["txt"], neg["true_neg"]))
    print("Best balanced score: {} ({}/{})".format(bal["txt"], bal["true_pos"], bal["true_neg"]))

def train_model(data, seed = None):
    if seed is not None:
        np.random.seed(seed)
        data = data.iloc[np.random.permutation(len(data))]

    pos_data = data[data["similar"] == 1.0]
    neg_data = data[data["similar"] == 0.0][:len(pos_data)]
    data = ps.concat([pos_data, neg_data])
    print("Размер учебной выборки: {}".format(len(data)))

    X = data.drop(["id1", "id2", "similar"], axis=1)
    y = data["similar"]
    
    model = LogisticRegression(penalty='l1', tol=0.01)
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

    print("Размер тестовой выборки: {}".format(len(data)))
    print("Попадания: {}".format(score))
    print("True positives: {}".format(true_pos))
    print("True negatives: {}".format(true_neg))

def predict_values(model, data):
    if "similar" in data.columns:
        data = data.drop(["similar"], axis=1)
    X = data.drop(["id1", "id2"], axis=1)
    y = model.predict(X)
    return ps.concat([data, ps.DataFrame({"similar": y}, index=None)], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="загрузить модель из файла", metavar="FILE")
    parser.add_argument("-t", "--train", help="обучить модель на данных", metavar="FILE")
    parser.add_argument("-c", "--compete", help="проверить производительность моделей", nargs=2, metavar=("TRAIN", "TEST"))
    parser.add_argument("-o", "--output", help="записать модель в файл", metavar="FILE")
    parser.add_argument("-v", "--validate", help="проверить производительность модели на выборке", metavar="FILE")
    parser.add_argument("-p", "--predict", help="предсказать значения для выборки", nargs=2, metavar=("INFILE", "OUTFILE"))
    parser.add_argument("-r", "--random", help="семя генератора псевдослучайных чисел", metavar="SEED", type=int)
    args = parser.parse_args()

    if args.input:
        model = pickle.load(args.input)
    elif args.train:
        print("Читаю тренировочные данные...", end=' ')
        sys.stdout.flush()
        data = ps.read_csv(args.train)
        print("ОК.")
        model = train_model(data, args.random)
    elif args.compete:
        print("Читаю обучающую выборку...", end=' ')
        sys.stdout.flush()
        train = ps.read_csv(args.compete[0])
        print("ОК.")
        print("Читаю тестовую выборку...", end=' ')
        sys.stdout.flush()
        test = ps.read_csv(args.compete[1])
        print("ОК.")
        compete_models(train, test)
    else:
        return print("Не указано, откуда брать модель (ключи -i, -t, -c)")

    if args.validate:
        print("Читаю тестовые данные...", end=' ')
        sys.stdout.flush()
        test = ps.read_csv(args.validate)
        print("ОК.")
        validate_model(model, test)

    if args.predict:
        infile, outfile = args.predict
        to_predict = ps.read_csv(infile)
        predicted = predict_values(model, to_predict)
        predicted.to_csv(outfile)

    if args.output:
        pickle.dump(model, args.output)

if __name__ == "__main__":
    main()