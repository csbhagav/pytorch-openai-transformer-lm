import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        ids = []
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                ids.append(line[0])
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return ids, st, ct1, ct2, y


def write_to_file(lst, filename):
    with open(filename, "w") as f:
        for item in lst:
            f.write(item)
            f.write("\n")
    f.close()

def rocstories(data_dir, n_train=1497, n_valid=374):
    ids, storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'valid.csv'))
    teIds, teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'test.csv'))
    tr_ids, va_ids, tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, \
    va_ys = \
        train_test_split(ids, storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    write_to_file(tr_ids, "./train_ids.lst")
    write_to_file(va_ids, "./valid_ids.lst")
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)


def _anli(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        ids = []
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join([line[2]])
                c1 = ' '.join([line[7], line[6]])
                c2 = ' '.join([line[8], line[6]])
                ids.append(line[0])
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return ids, st, ct1, ct2, y


# def _anli(path):
#     with open(path, encoding='utf_8') as f:
#         f = csv.reader(f)
#         ids = []
#         st = []
#         ct1 = []
#         ct2 = []
#         y = []
#         for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
#             if i > 0:
#                 s = ' '.join([line[2], line[6]])
#                 c1 = ' '.join([line[7]])
#                 c2 = ' '.join([line[8]])
#                 ids.append(line[0])
#                 st.append(s)
#                 ct1.append(c1)
#                 ct2.append(c2)
#                 y.append(int(line[-1])-1)
#         return ids, st, ct1, ct2, y


def write_to_file(lst, filename):
    with open(filename, "w") as f:
        for item in lst:
            f.write(item)
            f.write("\n")
    f.close()


def anli(data_dir, n_train=1497, n_valid=500):
    ids, storys, comps1, comps2, ys = _anli(os.path.join(data_dir, 'train.csv'))
    teIds, teX1, teX2, teX3, _ = _anli(os.path.join(data_dir, 'test.csv'))
    tr_ids, va_ids, tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, \
    va_ys = \
        train_test_split(ids, storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    write_to_file(tr_ids, "./anli_train_ids.lst")
    write_to_file(va_ids, "./anli_valid_ids.lst")
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)