""" LSH module
"""
from utils import get_hash_family, jaccard_on_idxs
from itertools import combinations
import json

def display_lsh_curve(z, until=30):
    """ Print table of possible values
        for given z : |Signatures|
    """
    r = [_ for _ in range(1, z)]
    b = [int(z/_) for _ in r]
    s = [round((1/b[_j])**(1/r[_j]), 3) for _j, _ in enumerate(r)]
    tab_row = " | {} \t | {} \t | {} \t |"
    print(tab_row.format("r", "b", "s"))
    print('-'*30)
    for _r, _b, _s in zip(r, b, s):
        if _r*_b != z: continue
        print(tab_row.format(str(_r), str(_b), str(_s)))
        if _r >= until:
            break

def lsh(rat_minh, b, r):
    """ Implementation of LSH for MinHash signatures
    """
    # hashing of locally sensitive
    def hash_local(t):
        k, x = t
        _bands = []
        for j in range(b):
            _band = ''.join([
                str(v) \
                for v in x[r*j: r*(j+1)]])
            _bands.append(_band)
        return [(_hb, k) for _hb in _bands]
    # return candidates
    return rat_minh.flatMap(hash_local)\
                    .groupByKey()\
                    .mapValues(set)\
                    .filter(lambda x: len(x[1]) > 1)

def filter_valid(candidates, cache_feat, jacc_th, serialize=True):
    """ Filter valid candidates with a 
        Jaccard similarity >= TH
    """
    valids = candidates\
                .map(lambda x: x[1])\
                .flatMap(lambda x: combinations(x, 2))\
                .map(lambda x: (x,1))\
                .groupByKey().keys()\
                .map(lambda c: (
                    c[0], c[1], 
                    jaccard_on_idxs(
                        cache_feat[c[0]],
                        cache_feat[c[1]],
                    )
                ))\
                .filter(lambda c: c[2] >= jacc_th)
    if serialize:
        return valids\
                .map(lambda z: json.dumps({
                        "b1": z[0],
                        "b2": z[1],
                        "sim": z[2]
                    })
                )
    return valids