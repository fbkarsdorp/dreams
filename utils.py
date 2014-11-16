from itertools import ifilterfalse

def flatten(lst):
    if lst:
        car, cdr = lst[0], lst[1:]
        if isinstance(car, (list, tuple)):
            if cdr:
                return flatten(car) + flatten(cdr)
            return flatten(car)
        if cdr:
            return [car] + flatten(cdr)
        return [car]

def flatten_labels(container):
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten(i):
                yield j
        else:
            yield i

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
