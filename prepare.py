#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, sys
import pandas as ps
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

def mystem_parse(texts):
    result = []
    text = "\nDEADBEEF\n".join(texts)
    print("Calling mystem...", end=' ')
    sys.stdout.flush()
    pipe = Popen(["mystem", "-lni"], stdout=PIPE, stdin=PIPE)
    raw = pipe.communicate(text.encode("utf-8"))[0].decode("utf-8")
    print("Done.")
    msg = {"norms": set(), "lemma": {}}
    for line in raw.split():
        if "DEADBEEF" in line:
            result.append(msg)
            msg = {"norms": set(), "lemma": {}}
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
        msg["norms"].add(norm)
        msg["lemma"][norm] = lemma
    result.append(msg)
    return result

def prepare(data):
    data = filter_by_cluster(data)
    cluster_nums = {cluster: i for i, cluster
                    in enumerate(data["cluster"].unique())}
    messages = [{"id": ind, "text": row["text"],
                 "cluster": cluster_nums[row["cluster"]]}
                for ind, row in data.iterrows()]
    mystems = mystem_parse([m["text"] for m in messages])
    assert len(mystems) == len(messages)
    for mes, mys in zip(messages, mystems):
        mes.update(mys)
    #prepared_data = ps.DataFrame(columns=['id1','id2']+MYSTEM_GRAMMEMS+["similar"])
    rows = []
    for i, (m1, m2) in enumerate(combinations(messages, 2)):
        print("Combinations progress: {}/{}".format(i,
            len(messages)*(len(messages)-1)//2), end='\r')
        row = {'id1': m1['id'], 'id2': m2['id'],
            'similar': int(m1["cluster"] == m2["cluster"])}
        row.update({grm: 0 for grm in MYSTEM_GRAMMEMS})
        row.update(texts_comparison(m1, m2))
        rows.append(row)
        #if len(rows) > 1000000 or i == len(m1)*len(m2)//2-1:
        #    new_df = ps.DataFrame(rows, columns=['id1','id2']+MYSTEM_GRAMMEMS+["similar"])
        #    prepared_data = ps.concat([prepared_data, new_df])
        #    rows = []
    #return prepared_data
    return ps.DataFrame(rows, columns=['id1','id2']+MYSTEM_GRAMMEMS+["similar"])

def texts_comparison(t1, t2):
    result = {}
    common = set(word for word in t1["norms"] if word in t2["norms"])
    for word in common:
        lemma = t1["lemma"][word]
        result[lemma] = result.get(lemma, 0) + 1
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv file with data")
    parser.add_argument("-o", "--output", help="output file",
        metavar="FILE", default="prepared.csv")
    parser.add_argument("-n", "--number", help="number of rows to process",
        metavar="NUMBER", default=0, type=int)
    args = parser.parse_args()
    data = ps.read_csv(args.file, sep=';', header=None, index_col=0,
        names=['id','title','text','cluster','time','publisher'])
    if args.number > 0:
        data = data[:args.number]
    data = prepare(data)
    data.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()