#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, sys, pickle
from itertools import combinations
import pandas as ps
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

'''
    Модуль, осуществляющий обучение и использование модели.

    Использование:
        -t TRAINFILE - обучить модель на выборке
        -o OUTFILE - сохранить обученную модель в файл
        -i INFILE - прочитать модель из файла
        -v TESTFILE - оценить производительность модели на выборке
        -p INFILE OUTFILE - предсказать значения целевой функции
            для выборки INFILE и записать результат в OUTFILE
        -c TRAINFILE TESTFILE - оценить производительность всех возможных
            моделей, обучив их на выборке из TRAINFILE и проверив точность
            на выборке из TESTFILE
        -r SEED - задать семя генератора псевдослучайных чисел

    Пример использования:
        model.py -t prepared-500-s300.csv -v prepared-500-s700.csv
'''

def draw_plots(data, seed = None):
    # Перемешать выборку:
    if seed is not None:
        np.random.seed(seed)
        data = data.iloc[np.random.permutation(len(data))]

    # Выбрать все записи, значение целевой функции которых равно 1,
    # и ровно столько же записей со значением целевой функции 0:
    pos_data = data[data["similar"] == 1.0]
    neg_data = data[data["similar"] == 0.0][:len(pos_data)]
    data = ps.concat([pos_data, neg_data])

    features = ["S", "NAME"]
    d0 = data[data["similar"] == 0]
    d1 = data[data["similar"] == 1]
    for f1, f2 in combinations(features, 2):
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.scatter(d0[f1], d0[f2], color="red", alpha=0.25)
        plt.scatter(d1[f1], d1[f2], color="green", alpha=0.25)
        plt.show()

def draw_graph(data):
    graph = nx.Graph()
    ids = set()
    for ind, row in data.iterrows():
        ids.add(row["id1"])
        ids.add(row["id2"])
    nums = {id: i for i, id in enumerate(sorted(ids))}
    for ind, row in data.iterrows():
        id1, id2 = nums[row["id1"]], nums[row["id2"]]
        graph.add_node(id1)
        graph.add_node(id2)
        if row["similar"]:
            graph.add_edge(id1, id2)
    nx.draw_networkx(graph)
    plt.show()

def all_models(seed = None):
    '''
        Сгенерировать все возможные типы моделей.
        Аргументы:
            seed - семя генератора псевдослучайных чисел
        Возвращает: список моделей (не обученных)
    '''
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
    '''
        Осуществить оценку моделей по 4 показателям:
            - Общая точность
            - Процент верно предсказанных 0
            - Процент верно предсказанных 1
            - Среднее арифметическое двух предыдущих значений
        Аргументы:
            train - DataFrame с обучающей выборкой
            test - DataFrame с тестовой выборкой (должно быть указано
                значение целевой функции)
            seed - семя генератора псевдослучайных чисел
        Возвращает: None
    '''

    # Перемешать обучающую выборку:
    if seed is not None:
        np.random.seed(seed)
        train = train.iloc[np.random.permutation(len(train))]

    # Выбрать все записи, значение целевой функции которых равно 1,
    # и ровно столько же записей со значением целевой функции 0:
    pos_data = train[train["similar"] == 1.0]
    neg_data = train[train["similar"] == 0.0][:len(pos_data)]
    train = ps.concat([pos_data, neg_data])
    print("Размер учебной выборки: {}".format(len(train)))

    # Выделить данные и значения целевой функции:
    X1 = train.drop(["id1", "id2", "similar"], axis=1)
    y1 = train["similar"]
    X2 = test.drop(["id1", "id2", "similar"], axis=1)
    y2 = test["similar"]

    # Провести оценку каждой модели:
    models = all_models(seed)
    for mdl in models:
        mdl["model"].fit(X1, y1)
        y = mdl["model"].predict(X2)
        res = ps.DataFrame({"y": y2, "p": y}, index=None)
        
        mdl["prec"] = res[res["p"] == 1]["y"].mean()
        mdl["recall"] = res[res["y"] == 1]["p"].mean()
        mdl["f1"] = 2/(1/mdl["prec"] + 1/mdl["recall"])

        print("{}: f1 {:.4f}, prec {:.4f}, recall {:.4f}".format(
            mdl["txt"], mdl["f1"], mdl["prec"], mdl["recall"]))
    
    # Выбираем лучшую модель по f1 score:
    best = max(models, key=lambda x: x["f1"])
    print("Лучшая модель: {} (f1 {:.4f}, prec {:.4f}, recall {:.4f})".format(
        best["txt"], best["f1"], best["prec"], best["recall"]))

def train_model(data, seed = None):
    '''
        Осуществить обучение модели на выборке.
        Аргументы:
            data - DataFrame с обучающей выборкой
            seed - семя генератора псевдослучайных чисел
        Возвращает: обученную модель
    '''

    # Перемешать выборку:
    if seed is not None:
        np.random.seed(seed)
        data = data.iloc[np.random.permutation(len(data))]

    # Выбрать все записи, значение целевой функции которых равно 1,
    # и ровно столько же записей со значением целевой функции 0:
    pos_data = data[data["similar"] == 1.0]
    neg_data = data[data["similar"] == 0.0][:len(pos_data)]
    data = ps.concat([pos_data, neg_data])
    print("Размер учебной выборки: {}".format(len(data)))

    # Выделить данные и значения целевой функции:
    X = data.drop(["id1", "id2", "similar"], axis=1)
    y = data["similar"]
    
    # Обучить модель:
    #model = LogisticRegression(penalty='l1', tol=0.28, random_state=seed)
    #model = RandomForestClassifier(60, 'entropy', 7, random_state=seed)
    #model = KNeighborsClassifier(28, "uniform")
    model = SVC(C=0.1, kernel='poly', random_state=seed)
    model.fit(X, y)
    #print(X.columns, '\n', model.feature_importances_)
    return model

def validate_model(model, data, visualize):
    '''
        Проверить производительность модели на выборке.
        Аргументы:
            model - обученная модель
            data - DataFrame с тестовыми данными (в данных должны быть
                указаны значения целевой функции)
        Возвращает: None
    '''

    # Отделить данные, фактические значения функции и предсказанные:
    X = data.drop(["id1", "id2", "similar"], axis=1)
    y = data["similar"]
    p = model.predict(X)
    res = ps.DataFrame({"y": y, "p": p}, index=None)

    # Визуализировать графы:
    if visualize:
        ids = ps.DataFrame({"id1": data["id1"], "id2": data["id2"]})
        draw_graph(ps.concat([ids, ps.DataFrame({"similar": y})], axis=1))
        draw_graph(ps.concat([ids, ps.DataFrame({"similar": p})], axis=1))

    # Оценить производительность модели:
    score = (p == y).mean()
    report = classification_report(y, p)
    p_vs_o = ps.DataFrame({"predicted 0": [
            len(res[(res["p"] == 0) & (res["y"] == 0)]),
            len(res[(res["p"] == 0) & (res["y"] == 1)]),
        ],
        "predicted 1": [
            len(res[(res["p"] == 1) & (res["y"] == 0)]),
            len(res[(res["p"] == 1) & (res["y"] == 1)]),
        ]}, index=["observed 0", "observed 1"])

    print("Размер тестовой выборки: {}".format(len(data)))
    print('\n', report, '\n', p_vs_o)

def predict_values(model, data):
    '''
        Предсказать значения целевой функции на данной выборке.
        Аргументы:
            model - обученная модель
            data - DataFrame с данными (наличие или отсутствие
                значений целевой функции не играет роли)
        Возвращает: исходные данные с предсказанными значениями
            целевой функции
    '''
    if "similar" in data.columns:
        X = data.drop(["id1", "id2", "similar"], axis=1)
    else:
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
    parser.add_argument("-g", "--graph", help="визуализировать графы (работает только с -v)", action="store_true")
    parser.add_argument("-d", "--draw", help="нарисовать графики для данных", metavar="FILE")
    parser.add_argument("-p", "--predict", help="предсказать значения для выборки", nargs=2, metavar=("INFILE", "OUTFILE"))
    parser.add_argument("-r", "--random", help="семя генератора псевдослучайных чисел", metavar="SEED", type=int)
    args = parser.parse_args()

    if args.draw:
        print("Читаю обучающую выборку...", end=' ')
        sys.stdout.flush()
        data = ps.read_csv(args.draw)
        print("ОК.")
        draw_plots(data, args.random)

    if args.input:
        model = pickle.load(open(args.input, "rb"))
    elif args.train:
        print("Читаю обучающую выборку...", end=' ')
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
        compete_models(train, test, args.random)
    else:
        return print("Не указано, откуда брать модель (ключи -i, -t, -c)")

    if not (args.validate or args.predict or args.output):
        return print("Не указано, что делать с моделью (ключи -v, -p, -o, -c)")

    if args.validate:
        print("Читаю тестовые данные...", end=' ')
        sys.stdout.flush()
        test = ps.read_csv(args.validate)
        print("ОК.")
        validate_model(model, test, args.graph)

    if args.predict:
        infile, outfile = args.predict
        to_predict = ps.read_csv(infile)
        predicted = predict_values(model, to_predict)
        predicted.to_csv(outfile)

    if args.output:
        pickle.dump(model, open(args.output, "wb"))

if __name__ == "__main__":
    main()