#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import argparse, requests, json
import pandas as ps

from prepare import filter_by_cluster

'''
    Модуль проверки орфографии сообщений с помощью сервиса Яндекс.Спеллер.

    Использование:
        speller.py INFILE -o OUTFILE

        INFILE - csv-файл со значениями, разделёнными точкой с запятой,
            и без заголовка;
        OUTFILE - имя файла, в который записать исправленные тексты.

    Дополнительные флаги:
        -l LOGFILE - записать лог исправлений в указанный файл
        -n NUMBER - обработать только первые NUMBER записей
        -f - обрабатывать только записи, принадлежащие непустому кластеру
'''

def yandex_speller(texts):
    '''
        Запросить исправление орфографии у Яндекс.Спеллера. Аргументы:
            texts - список строк, которые необходимо исправить
        Возвращает: список исправленных строк
    '''
    res = []
    for i in range(0, len(texts), 5):
        print("Прогресс: {}/{}".format(i+5, len(texts)), end='\r')
        params = {'text': texts[i:i+5], 'lang': 'ru'}
        resp = requests.get('http://speller.yandex.net/services/spellservice.json/checkTexts', params=params)
        res += json.loads(resp.text)
    return res

def edit_orphography(data):
    '''
        Осуществить исправление орфографии. Аргументы:
            data - DataFrame, в поле 'text' которого находятся сообщения
        Возвращает: копию исходного DataFrame с исправленными текстами
    '''
    log = ""
    data = data.copy()
    edits_all = yandex_speller(list(data["text"]))
    for i, (edits, text) in enumerate(zip(edits_all, data["text"])):
        for edit in edits:
            if edit["s"]:
                # Если слово начинается с заглавной буквы,
                # игнорируем исправление:
                if edit["word"][0].upper() == edit["word"][0]:
                    continue
                log += "{} -> {}\n".format(edit["word"], edit["s"][0])
                pos = edit["pos"]
                ln = edit["len"]
                text = text[:pos] + edit["s"][0] + text[pos+ln:]
            data.iloc[i]["text"] = text
    return data, log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv файл с данными")
    parser.add_argument("-o", "--output", help="выходной файл",
        metavar="FILE", default="spelled.csv")
    parser.add_argument("-l", "--log", help="файл логов",
        metavar="FILE", default=None)
    parser.add_argument("-n", "--number", help="число записей для обработки",
        metavar="NUMBER", default=0, type=int)
    parser.add_argument("-f", "--filter", help="отфильтровать записи без кластера", action="store_true")
    args = parser.parse_args()

    data = ps.read_csv(args.file, sep=';', header=None, index_col=0,
        names=['id','title','text','cluster','time','publisher'])
    if args.filter:
        data = filter_by_cluster(data)
    if args.number > 0:
        data = data[:args.number]
    data, log = edit_orphography(data)
    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            f.write(log)
    data.to_csv(args.output, sep=';', header=None)
    print("\nDone.")

if __name__ == "__main__":
    main()