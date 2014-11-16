from functools import partial
import numpy as np


def reciprocal_rank(ranking, references, atk=None):
    """Compute the reciprocal rank of a ranked list against
    some ground truth."""
    for k, prediction in enumerate(ranking[:atk], 1):
        if prediction in references:
            return 1.0 / k
    return 0.0

def average_precision(ranking, references, atk=None):
    """Compute the average precision of a ranked list against
    some ground truth"""
    total, num_correct = 0.0, 0.0
    for k, prediction in enumerate(ranking[:atk], 1):
        if prediction in references:
            num_correct += 1
            total += num_correct / k
    return total / num_correct if total > 0 else 0.0

def _mean_score(rankings, references, fn):
    return sum(fn(ranking, reference) for ranking, reference in zip(rankings, references)) / len(rankings)

def mean_average_precision(rankings, references, atk=None):
    """Compute the mean average precision. Input should be a list of
    prediction rankings and a list of ground truth rankings."""
    return _mean_score(rankings, references, partial(average_precision, atk=atk))

def mean_reciprocal_rank(rankings, references, atk=None):
    """Compute the mean reciprocal precision. Input should be a list of
    prediction rankings and a list of ground truth rankings."""
    return _mean_score(rankings, references, partial(reciprocal_rank, atk=atk))

def one_error(ranking, references):
    "Compute the one error score or AP@1"
    return average_precision(ranking, references, atk=1)

def is_error(ranking, references):
    """Return 1 if the predicted ranking is not perfect."""
    return 1 if average_precision(ranking, references) < 1 else 0

def margin(ranking, references):
    """Return the margin, or absolute difference between the highest
    irrelevant item and the lowest relevant one."""
    lowest_relevant, highest_irrelevant = 0, 0
    for k, prediction in enumerate(ranking, 1):
        if prediction not in references and highest_irrelevant is 0:
            highest_irrelevant = k
        if prediction in references and k > lowest_relevant:
            lowest_relevant = k
    return abs(lowest_relevant - highest_irrelevant)

def dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)

def ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.
    return dcg(relevances, rank) / best_dcg
