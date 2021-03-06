#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, sys
import pandas as ps
import numpy as np
from itertools import combinations
from subprocess import Popen, PIPE

from porter import Porter

'''
    Модуль, осуществляющий преобразование сырых текстов
    в матрицу признаков.
    Использование:
        prepare.py INFILE -o OUTFILE
        INFILE - csv-файл со значениями, разделёнными точкой с запятой,
            и без заголовка;
        OUTFILE - имя файла, в который записать матрицу признаков.
    Дополнительные флаги:
        -n NUMBER - обработать только первые NUMBER сообщений;
        -s NUMBER - разделить реультат на два файла, в первый поместить
            NUMBER строк, во второй все остальные;
        -r SEED - случайно перемешать строки входного файла, использовав
            SEED в качестве семени генератора псевдослучайных чисел.
'''

MYSTEM_GRAMMEMS = ["NAME", "A", "ADV", "ADVPRO", "ANUM", "APRO", "COM",
    "CONJ", "INTJ", "NUM", "PART", "PR", "S", "SPRO", "V"]

def choose_with_cluster(data, num):
    '''
        Выбрать из входных данных ровно num сообщений,
        при этом сообщения выбираются целыми кластерами.
        Аргументы:
            data - DataFrame с исходными сообщениями;
            num - количество сообщений, которое надо выбрать.
        Возвращает: DataFrame с выбранными сообщениями.
    '''
    data = filter_by_cluster(data)
    result = []
    ln = 0
    for cluster in data["cluster"].unique():
        result.append(data[data["cluster"] == cluster])
        ln += len(result[-1])
        if ln > num:
            return ps.concat(result)[:num]

def filter_by_cluster(data):
    '''
        Отфильтровать сообщения, оставив лишь принадлежащие
        непустым кластерам. Параметры:
            data - DataFrame с сообщениями;
        Аргументы: DataFrame с сообщениями, у каждого из которых
        в кластере есть как минимум одно другое сообщение.
    '''
    data = data[data["cluster"] != "-"]
    count = {}
    for cluster in data["cluster"]:
        count[cluster] = count.get(cluster, 0) + 1
    allow = set(cluster for cluster, k in count.items() if k > 1)
    return data[data["cluster"].isin(allow)]

def levenstein_distance(a, b):
    '''
        Вернуть расстояние Левенштейна между строками a и b.
        Аргументы:
            a, b - строки;
        Возвращает: целое число - расстояние Левенштейна.
    '''
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current_row = range(n+1)
    for i in range(1, m+1):
        previous_row, current_row = current_row, [i]+[0]*n
        for j in range(1,n+1):
            add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]
            if a[j-1] != b[i-1]:
                change += 1
            current_row[j] = min(add, delete, change)
    return current_row[n]

def mystem_parse(texts):
    '''
        Прогнать список текстов через mystem, вернув для каждого слова
        его нормальную форму и граммему.
        Аргументы:
            texts - список строковых значений;
        Возвращает: список наборов, где каждый набор соответствует
        входному тексту и содержит кортежи (нормальная форма, граммема)
    '''
    result = []
    # В качестве разделителя сообщений используем волшебное слово.
    # Грязно, но работает:
    text = "\nDEADBEEF\n".join(texts)

    print("Calling mystem...", end=' ')
    sys.stdout.flush()
    pipe = Popen(["mystem", "-lni"], stdout=PIPE, stdin=PIPE)
    raw = pipe.communicate(text.encode("utf-8"))[0].decode("utf-8")
    print("Done.")

    msg = set()
    for line in raw.split():
        if "DEADBEEF" in line:
            result.append(msg)
            msg = set()
            continue
        if line[-1] == '?':
            # Если mystem не опознал слово, трактуем его
            # как имя собственное:
            norm = line
            gramm = "NAME"
        else:
            norm = line[:line.find("=")]
            gramm = None
            line = line.split("|")[0][line.find("=")+1:]
            for trait in ("гео", "имя", "фам", "отч"):
                if trait in line:
                    gramm = "NAME"
                    break
            if gramm is None:
                gramm = line.split('=')[0].split(',')[0]
        assert gramm in MYSTEM_GRAMMEMS
        norm = norm.strip('?')
        msg.add((norm, gramm))
    result.append(msg)
    return result

def prepare(data, titles = False, porter = False):
    '''
        Составить матрицу признаков для данных сообщений.
        Аргументы:
            data - DataFrame с исходными сообщениями.
        Возвращает: DataFrame - матрицу признаков.
    '''
    cluster_nums = {cluster: i for i, cluster
                    in enumerate(data["cluster"].unique())}
    messages = [{"id": ind,
        "text": ((row["title"]+" ") if titles else "") + row["text"],
        "cluster": cluster_nums[row["cluster"]]}
        for ind, row in data.iterrows()]
    tokens = mystem_parse([m["text"] for m in messages])
    assert len(tokens) == len(messages)
    for mes, tok in zip(messages, tokens):
        mes["tokens"] = tok
        if porter:
            mes["lemmas"] = set(Porter.stem(norm) for norm, _ in tok)
    rows = []
    for i, (m1, m2) in enumerate(combinations(messages, 2)):
        print("Прогресс в сочетаниях: {}/{}".format(i+1,
            len(messages)*(len(messages)-1)//2), end='\r')
        row = {'id1': m1['id'], 'id2': m2['id'],
            'similar': int(m1["cluster"] == m2["cluster"])}
        row.update({grm: 0 for grm in MYSTEM_GRAMMEMS})
        row.update(texts_comparison(m1, m2))
        rows.append(row)
    print("")
    return ps.DataFrame(rows, columns=['id1','id2'] + MYSTEM_GRAMMEMS \
        + ["semantic_repeats", "similar"])

def texts_comparison(t1, t2):
    '''
        Осуществить сравнение двух текстов и составить одну
        строку признаков.
        Аргументы:
            t1, t2 - словари, у которых ключ 'tokens' - это набор
                кортежей (нормальная форма слова, граммема слова)
        Возвращает: словарь с результатом сравнения данных текстов
    '''
    result = {}

    # Ищем явные повторы:
    common = t1["tokens"] & t2["tokens"]
    for norm, grammema in common:
        result[grammema] = result.get(grammema, 0) + 1

    # Ищем неявные повторы:
    if "lemmas" in t1:
        # Если есть леммы - значит, использован алгоритм Портера
        result["semantic_repeats"] = len(t1["lemmas"] & t2["lemmas"]) - len(common)
    else:
        tokens_left = t1["tokens"] - common
        tokens_right = t2["tokens"] - common
        left_matches, right_matches = set(), set()
        for w1, l1 in tokens_left:
            for w2, l2 in tokens_right:
                if levenstein_distance(w1, w2) < max((len(w1), len(w2)))//2:
                    left_matches.add(w1)
                    right_matches.add(w2)
        result["semantic_repeats"] = min((len(left_matches), len(right_matches)))
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv file with data")
    parser.add_argument("-o", "--output", help="выходной файл",
        metavar="FILE", default="prepared.csv")
    parser.add_argument("-s", "--split", help="number of rows to place in the first split",
        metavar="NUMBER", default=0, type=int)
    parser.add_argument("-n", "--number", help="number of rows to process",
        metavar="NUMBER", default=0, type=int)
    parser.add_argument("-t", "--title", help="конкатенировать заголовок", action="store_true")
    parser.add_argument("-p", "--porter", help="использовать стеммер Портера", action="store_true")
    parser.add_argument("-r", "--random", help="семя генератора псевдослучайных чисел", metavar="SEED", type=int)
    args = parser.parse_args()

    data = ps.read_csv(args.file, sep=';', header=None, index_col=0,
        names=['id','title','text','cluster','time','publisher'])

    if args.random:
        np.random.seed(args.random)
        data = data.iloc[np.random.permutation(len(data))]

    if args.number > 0:
        data = choose_with_cluster(data, args.number)

    if args.split > 0:
        chunks = [data[:args.split], data[args.split:]]
    else:
        chunks = [data]

    for chunk in chunks:
        prep = prepare(chunk, args.title, args.porter)
        if args.split > 0:
            dot = args.output.rfind(".")
            filename = args.output[:dot] + "-s" + str(len(chunk)) \
                + args.output[dot:]
        else:
            filename = args.output
        prep.to_csv(filename, index=False)

if __name__ == "__main__":
    main()