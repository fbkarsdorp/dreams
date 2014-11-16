from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

from utils import flatten
from metrics import mean_reciprocal_rank, mean_average_precision


class IRSystem(BaseEstimator):
    """Implementation of a simple IR system which uses Okapi BM25
    to rank documents given a query. The interface has been modeled
    so that it matches to API of sklearn."""
    def __init__(self, k1=1.2, b=0.75, cutoff=0):
        """Initialize a new IR Sytem. The cutoff argument is used
        to filter terms that have an idf-score smaller that the number
        provided. The default is 0, which basically means that it will
        remove all terms that occur in more than 50 percent of the
        documents."""
        self.k1 = k1
        self.b = b
        self.cutoff = cutoff

    def fit(self, X, y, copy=False):
        """Use this method to create an index for a collection of documents
        X. X has to be a n by m (sparse) matrix, where n is the number of
        documents and m the number of features. sklearn's CountVectorizer
        can provide such a datastructure easily (see the method fit_raw)"""
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)
        n_samples, n_features = X.shape
        # compute the document frequency of each term in X
        df = np.bincount(X.indices, minlength=n_features)
        # compute the idf values for each term in X
        idf = np.log((float(n_samples) - df + 0.5) / (df + 0.5))
        # store the idf values in the spdiag matrix
        self.idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        # store the length of each document
        doc_lengths = np.array(X.sum(axis=1))
        # the average document length
        avgN = doc_lengths.sum() / n_samples
        # precompute the denominator
        self.denom = np.array(self.k1 * (1 - self.b + (self.b * doc_lengths / avgN))).flatten()
        self._values = X * (self.k1 + 1)
        # loop over all nonzero entries
        for i in range(self._values.shape[0]):
            begin_col_index = self._values.indptr[i]
            end_col_index = self._values.indptr[i+1]
            for index in range(begin_col_index, end_col_index):
                self._values.data[index] /= (X.data[index] + self.denom[i])
        # compute final scores
        self._values = self._values * self.idf_diag
        self._values.data[self._values.data<self.cutoff] = 0
        self._target = np.array(y)
        self._training_instances = X
        return self

    def index(self, X, y):
        "Same as fit, only here to provide a method with a more intuitive name..."
        return self.fit(X, y)

    def _transform_raw(self, raw_documents):
        return self.vectorizer.transform(raw_documents)

    def fit_raw(self, raw_documents, labels, analyzer='word', stop_words=None, ngram_range=(1, 1), min_df=1, max_df=1.0):
        """Fit or index a collection of raw documents. This method
        will call the CountVectorizer object from sklearn."""
        self.vectorizer = CountVectorizer(min_df=min_df, max_df=max_df,
                                          analyzer=analyzer, ngram_range=ngram_range)
        return self.fit(self.vectorizer.fit_transform(raw_documents), labels)

    def scores(self, X, raw=False):
        """Given a query or a collection of queries X, return
        the scores against the indexed collection. Again, X has to
        be a (sparse) matrix with n documents and m features."""
        if raw: X = self._transform_raw(X)
        scores = (X * self._values.T)
        return scores

    def rank_scores(self, X, raw=False, k=10, norm=False):
        if raw: X = self._transform_raw(X)
        scores = self.scores(X).toarray()
        if norm:
            scores = scores / np.linalg.norm(scores)
        rankings = scores.argsort(1)
        return [zip(self._target[ranking[::-1][:k]], scores[i][ranking[::-1][:k]]) for i, ranking in enumerate(rankings)]

    def rank(self, X, raw=False):
        """Return a ranking of all documents in the index against
        the query of queries X."""
        if raw: X = self._transform_raw(X)
        rankings = self.scores(X).toarray().argsort(1)
        return np.array([ranking[::-1] for ranking in rankings])

    def rank_labels(self, X, raw=False):
        if raw: X = self._transform_raw(X)
        return self._target[self.rank(X)]

    def predict(self, X, raw=False):
        "For each query in X, return the top 1 document in the index."
        if raw: X = self._transform_raw(X)
        return [self._target[ranking[-1]] for ranking in self.rank(X)]

    def score(self, X, y):
        return mean_reciprocal_rank(y, self.predict(X))


class ML_IRSystem(IRSystem):
    def __init__(self, num_neighbors=5, smooth=1.0, k1=1.2, b=0.75, cutoff=0):
        IRSystem.__init__(self, k1, b, cutoff)
        self.num_neighbors = num_neighbors
        self.smooth = float(smooth)
        self.priors = {}
        self.n_priors = {}
        self.conds = defaultdict(dict)
        self.n_conds = defaultdict(dict)

    def fit(self, X, y, copy=False):
        IRSystem.fit(self, X, y, copy)
        self._target = list(self._target)
        self._labels = tuple(set(flatten(self._target)))
        self._label_dist = Counter(flatten(self._target))
        self._target = np.array(self._target)
        self.compute_prior()
        self.compute_conditional()

    def compute_prior(self):
        if not hasattr(self, '_target'):
            raise ValueError("Model not fitted.")
        denom = self.smooth * len(self._labels) + sum(self._label_dist.values())
        for label in self._labels:
            label_count = self._label_dist[label]
            self.priors[label] = float(self.smooth + label_count) / denom
            self.n_priors[label] = 1.0 - self.priors[label]

    def compute_conditional(self):
        if not hasattr(self, '_training_instances'):
            raise ValueError("Model not fitted.")
        tmp_Ci = defaultdict(lambda: defaultdict(int))
        tmp_NCi = defaultdict(lambda: defaultdict(int))
        for i, ranking in enumerate(IRSystem.rank(self, self._training_instances)):
            neighbors = [n for n in ranking[:self.num_neighbors+1] if n != i]
            for label in self._labels:
                aces = sum([1 for labels in self._target[neighbors] if label in labels])
                if label in self._target[i]:
                    tmp_Ci[label][aces] += 1.0
                else:
                    tmp_NCi[label][aces] += 1.0
        s = self.smooth
        nn = self.num_neighbors
        for label in self._labels:
            tmp1, tmp2 = 0.0, 0.0
            for n in range(self.num_neighbors+1):
                tmp1 += tmp_Ci[label][n]
                tmp2 += tmp_NCi[label][n]
            for n in range(self.num_neighbors+1):
                self.conds[label][n] = (s + tmp_Ci[label][n]) / (s * (nn + 1) + tmp1)
                self.n_conds[label][n] = (s + tmp_NCi[label][n]) / (s * (nn + 1) + tmp2)

    def rank(self, X, raw=False):
        rankings = IRSystem.rank(self, X, raw)
        prob_rankings = []
        for i, ranking in enumerate(rankings):
            probabilities = {}
            neighbors = ranking[:self.num_neighbors]
            for label in self._labels:
                aces = sum(1 for labels in self._target[neighbors] if label in labels)
                prob_in = self.priors[label] * self.conds[label][aces]
                prob_out = self.n_priors[label] * self.n_conds[label][aces]
                probabilities[label] = prob_in / (prob_in + prob_out)
            prob_rankings.append(probabilities)
        return prob_rankings

    def predict(self, X, raw=False, t=0.5):
        rankings = self.rank(X, raw=raw)
        for ranking in rankings:
            yield [label for label, probability in ranking.items() if probability >= t]

if __name__ == '__main__':
    documents = ['boek schrijver inkt de de had een',
                 'boek boef kaars schrijver schrijver lezer de de een',
                 'kasteel zwaard heer een de gekke',
                 'kasteel dame jonkvrouw een de hond']
    labels = [['boek', 'humanities'], ['boek'],
              ['middeleeuwen'], ['middeleeuwen', 'vrouw']]
    queries = ['schrijver inkt', 'dame zwaard']
    model = ML_IRSystem(num_neighbors=2, smooth=1.0, k1=1.2, b=0.75, cutoff=0)
    model.fit_raw(documents, labels)
    print(model.rank(queries, raw=True))
