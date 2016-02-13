# Поиск семантический дубликатов
В даном репозитории представлено решение задачи поиска семантических дубликатов. Суть задачи в том, чтобы построить модель, способную сказать, являются ли два данных текста семантическими дубликатами.

## Алгоритм решения задачи
1. Все сообщения проходят проверку на наличие опечаток через сервис Яндекс.Speller. При этом игнорируются исправления во всех словах, начинающихся с заглавной буквы - иначе спеллер делает очень много исправлений в именах собственных;
2. Все слова приводятся к нормальной форме и "получают" граммему с помощью Mystem;
3. Для каждой пары сообщений:
  1. Производится поиск явных повторов: для каждой пары {нормальная форма, граммема} в первом сообщении ищется такая же пара во втором сообщении. Если совпадение найдено, для данной граммемы счётчик совпадений увеличивается на единицу;
  2. Производится поиск неявных повторов: строится двудольный граф, в котором каждая вершина - это нормальная форма слова, у которой нет явного совпадения в другом сообщении, а ребро между вершинами означает, что расстояние Левенштейна между двумя данными нормальными формами меньше или равно длинне наибольшей из двух нормальных форм попалам (рёбрами соединяются только нормальные формы слов из разных сообщений). Для данного графа вычисляется максимальное паросочетание, его размер записывается в атрибут "неявные повторы";
4. Для обучения модели отбираются некоторые пары сообщений, являющиеся семантическими повторами, и ещё столько же случайно выбранных пар сообщений, семантическими повторами не являющиеся;
5. Производительность модели проверяется на всех оставшихся парах сообщений, не участвовавших в обучении модели.

## Практические ограничений алгоритма
1. Сервис Яндекс.Спеллер имеет ограничение на количество обрабатываемых от одного клиента сообщений, и все сообщения за раз проверить на наличие ошибок нельзя. Поэтому при проверке имеет смысл отбросить сообщения, кластер которых не определён - тогда возможно проверить всю выборку за раз;
2. Скорость преобразования сырых текстов сообщений в таблицу признаков - порядка нескольких сотен пар сообщений в секунду. В разумные сроки можно составить таблицу признаков примерно для 1000 сообщений, выбранных из исходной выборки случайным образом.

## Компоненты решения
Решение задачи разбито на следующие независимые модули:
* speller - осуществляет исправление в исходных текстах орфографических ошибок с использованием сервиса Яндекс.Speller;
* prepare - осуществляет преобразование исходных текстов в матрицу признаков для классификатора;
* model - осуществляет обучение модели, её запись на диск и чтение с диска, оценку производительности.

## Интерфейсы модулей
### speller.py
Основной способ взаимодействия с модулем:

```speller.py INFILE -o OUTFILE```

* INFILE - csv-файл с входными данными (значения разделены точкой с запятой, без заголовка, кодировка utf-8)
* OUTFILE - файл, в который будет записан результат.

Поддерживаются следующие флаги:
* -l FILE - записать лог преобразований орфографии в файл
* -n NUMBER - обработать только первые NUMBER записей
* -f - отфильтровать те записи, у которых не указан кластер, либо в кластере всего одна запись

### prepare.py
Основной способ взаимодействия с модулем:

```prepare.py INFILE -o OUTFILE```

* INFILE - csv-файл с входными данными (значения разделены точкой с запятой, без заголовка, кодировка utf-8)
* OUTFILE - файл, в который будет записан результат. Здесь уже значения разделены запятой и заголовок присутствует (умолчания для pandas)

Поддерживаются следующие флаги:
* -n NUMBER - обработать только первые NUMBER записей
* -s NUMBER - разделить результат на два независимых файла, в первом будет NUMBER записей, во втором все остальные
* -r SEED - случайно перемешать строки во входной матрице, используя SEED как семя генератора псевдослучайных чисел


### model.py
Данный модуль имеет несколько сценариев взеимодействия.

Способы задания источника модели:
* -t TRAINFILE - обучить модель на указанном файле с обучающей выборкой
* -i INFILE - загрузить ранее сохранённую модель из файла

Сохранение модели в файл:
* -o OUTFILE

Использование модели:
* -v TESTFILE - проверить, насколько хорошо модель предскажет значение целевой функции для выборки в данном файле (в выборке должны присутствовать значения целевой функции)
* -p INFILE OUTFILE - предсказать значения целевой функции для выборки из INFILE и записать результат в OUTFILE

Другое:
* -r SEED - задать значение семени генератора псевдослучайных чисел

Пример использования: `model.py -t train.csv -v test.csv -o model.mdl -r 42`

## Пример получения готовой модели из сырых данных

```
speller.py 120k-utf8.csv -o spelled-f.csv -l log.txt -f
prepare.py spelled-f.csv -o prepared-1000.csv -n 1000 -s 300 -r 42
model.py -t prepared-1000-s300.csv -v prepared-1000-s700.csv -o model.mdl -r 42
```
