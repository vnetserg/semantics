#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, sys
import pandas as ps
import networkx as nx
import numpy as np
from itertools import combinations
from subprocess import Popen, PIPE

MYSTEM_GRAMMEMS = ["NAME", "A", "ADV", "ADVPRO", "ANUM", "APRO", "COM",
    "CONJ", "INTJ", "NUM", "PART", "PR", "S", "SPRO", "V"]

def filter_by_cluster(data):
    data = data[data["cluster"] != "-"]
    count = {}
    for cluster in data["cluster"]:
        count[cluster] = count.get(cluster, 0) + 1
    allow = set(cluster for cluster, k in count.items() if k > 1)
    return data[data["cluster"].isin(allow)]

def levenstein_distance(a, b):
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
    result = []
    text = "\n#!#DEADBEEF#!#\n".join(texts)

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
            norm = line.strip('?')
            lemma = "NAME"
        else:
            norm = line[:line.find("=")]
            lemma = None
            line = line.split("|")[0][line.find("=")+1:]
            for trait in ("гео", "имя", "фам", "отч"):
                if trait in line:
                    lemma = "NAME"
                    break
            if lemma is None:
                lemma = line.split('=')[0].split(',')[0]
        assert lemma in MYSTEM_GRAMMEMS
        msg.add((norm, lemma))
    result.append(msg)
    return result

def prepare(data):
    data = filter_by_cluster(data)
    cluster_nums = {cluster: i for i, cluster
                    in enumerate(data["cluster"].unique())}
    messages = [{"id": ind, "text": row["text"],
                 "cluster": cluster_nums[row["cluster"]]}
                for ind, row in data.iterrows()]
    tokens = mystem_parse([m["text"] for m in messages])
    assert len(tokens) == len(messages)
    for mes, tok in zip(messages, tokens):
        mes["tokens"] = tok
    rows = []
    for i, (m1, m2) in enumerate(combinations(messages, 2)):
        print("Combinations progress: {}/{}".format(i+1,
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
    result = {}

    common = t1["tokens"] & t2["tokens"]
    for norm, lemma in common:
        result[lemma] = result.get(lemma, 0) + 1

    graph = nx.Graph()
    tokens_left = t1["tokens"] - common
    tokens_right = t2["tokens"] - common
    for w1, l1 in tokens_left:
        for w2, l2 in tokens_right:
            if levenstein_distance(w1, w2) < max((len(w1), len(w2)))//2:
                graph.add_edge(w1, w2)
    result["semantic_repeats"] = len(nx.maximal_matching(graph))

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
    parser.add_argument("-r", "--random", help="семя генератора псевдослучайных чисел", metavar="SEED", type=int)
    args = parser.parse_args()

    data = ps.read_csv(args.file, sep=';', header=None, index_col=0,
        names=['id','title','text','cluster','time','publisher'])

    if args.random:
        np.random.seed(args.random)
        data = data.iloc[np.random.permutation(len(data))]

    if args.number > 0:
        data = data[:args.number]

    if args.split > 0:
        chunks = [data[:args.split], data[args.split:]]
    else:
        chunks = [data]

    for chunk in chunks:
        prep = prepare(chunk)
        if args.split > 0:
            dot = args.output.rfind(".")
            filename = args.output[:dot] + "-s" + str(len(chunk)) \
                + args.output[dot:]
        else:
            filename = args.output
        prep.to_csv(filename, index=False)

if __name__ == "__main__":
    main()