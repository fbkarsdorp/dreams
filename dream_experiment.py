import os
from collections import defaultdict, Counter
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

from IRSystem import IRSystem, ML_IRSystem
from metrics import mean_average_precision
from utils import flatten, unique_everseen
import re


def recurse_labels(labels):
    label = labels[0]
    xml = '<label name="%s">' % label
    if len(labels) > 1:
        xml += recurse_labels(labels[1])
    xml += '</label>'
    return xml

def label_hierarchy(label):
    return (label[0], label_hierarchy(label[1:])) if len(label) > 1 else (label[0], )

def read_dreams(filename, lemmata=True):
    label = None
    text = []
    for line in open(filename):
        line = line.strip()
        if line:
            if 'QQQ' in line:
                yield label, ' '.join(text)
                label = re.search('QQQ[0-9]+\.', line).group()
                text = []
            else:
                if line.startswith('<br>'):
                    continue
                line = line.split()
                #if line[1].startswith('V'):
                text.append(line[2 if lemmata else 0])


def read_labels(filename):
    for line in open(filename):
        line = line.strip()
        if line:
            fields = line.split('\t')
            if len(fields) > 2:
                do_ic, _, labels = line.split('\t')
                labels = list(set([l for l in labels.split()]))
                yield do_ic, labels


def match_labels_documents(documents, labels):
    for doc_id, label in labels:
        yield label, documents[doc_id]


if __name__ == '__main__':
    for labelfile in ("labels.hcnorms_misgoodfortune.all.txt",
                      "labels.hcnorms_char.all.txt",
                      "dream_acts.txt", 'dream_sets.txt', "all_labels.txt"):
    # First we'll do a regular IR experiment with BM25
        documents = {doc_id: text for doc_id, text in read_dreams("data/dreambank.en.stanford.out")}
        labels = list(read_labels("data/" + labelfile))
        y, X = zip(*match_labels_documents(documents, labels))
        y, X = np.array(y), np.array(X)
        kf = KFold(len(y), n_folds=10, shuffle=True, random_state=1)
        rank_scores = np.zeros(10)
        for i, (train, test) in enumerate(kf):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            labels = Counter(flatten(list(y_train)))
            labels = [label for label, count in labels.items() if count >= 1]
            model = IRSystem(k1=1.2, b=0.75, cutoff=0)
            model.fit_raw(X_train, y_train, ngram_range=(1, 1), stop_words='english', min_df=2)
            ranking = model.rank_labels(X_test, raw=True)
            ranking = ranking.tolist()
            ranking = map(lambda r: list(unique_everseen(r)), map(flatten, ranking))
            ranking, y_test = zip(*[(r, y_) for r, y_ in zip(ranking, y_test) if any(l in labels for l in y_)])
            rank_scores[i] = mean_average_precision(ranking, y_test)
        print 'IR: (%s)' % (labelfile), rank_scores.mean(), rank_scores.std()

        # Next, we'll do an IR experiment with Big Documents
        documents = {doc_id: text for doc_id, text in read_dreams("data/dreambank.en.stanford.out")}
        labels = list(read_labels("data/" + labelfile))
        y, X = zip(*match_labels_documents(documents, labels))
        y, X = np.array(y), np.array(X)
        kf = KFold(len(y), n_folds=10, shuffle=True, random_state=1)
        rank_scores = np.zeros(10)
        for i, (train, test) in enumerate(kf):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            big_docs = defaultdict(str)
            for (labels, doc) in zip(y_train, X_train):
                for label in labels:
                    big_docs[label] += " " + doc
            y_train, X_train = zip(*big_docs.items())
            labels = Counter(flatten(list(y_train)))
            labels = [label for label, count in labels.items() if count >= 1]
            model = IRSystem(k1=1.2, b=0.75, cutoff=0)
            model.fit_raw(X_train, y_train, ngram_range=(1, 1), stop_words='english', min_df=2)
            ranking = model.rank_labels(X_test, raw=True)
            ranking, y_test = zip(*[(r, y_) for r, y_ in zip(ranking, y_test) if any(l in labels for l in y_)])
            rank_scores[i] = mean_average_precision(ranking, y_test)
        print 'Big Doc IR (%s):' % labelfile, rank_scores.mean(), rank_scores.std()

        # Finally we'll do an experiment with a ML IR system.
        documents = {doc_id: text for doc_id, text in read_dreams("data/dreambank.en.stanford.out")}
        labels = list(read_labels("data/" + labelfile))
        y, X = zip(*match_labels_documents(documents, labels))
        y, X = np.array(y), np.array(X)
        kf = KFold(len(y), n_folds=10, shuffle=True, random_state=1)
        rank_scores = np.zeros(10)
        for i, (train, test) in enumerate(kf):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            labels = Counter(flatten(list(y_train)))
            labels = [label for label, count in labels.items() if count >= 1]
            model = ML_IRSystem(num_neighbors=101, smooth=1., k1=1.2, b=0.75, cutoff=0)
            model.fit_raw(X_train, y_train, ngram_range=(1, 1), stop_words='english', min_df=2)
            ranking = model.rank(X_test, raw=True)
            ranking = [sorted(ranks, key=ranks.__getitem__, reverse=True) for ranks in ranking]
            ranking, y_test = zip(*[(r, y_) for r, y_ in zip(ranking, y_test) if any(l in labels for l in y_)])
            rank_scores[i] = mean_average_precision(ranking, y_test)
        print 'ML IR (%s):' % (labelfile), rank_scores.mean(), rank_scores.std()
