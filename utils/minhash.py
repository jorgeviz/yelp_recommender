""" MinHash Module
"""
from utils import *
import math


def minhash(rmat, num_shingles, num_signatures):
    """ Apply Min Hash per document

        Return: 
        -----
        pyspark.rdd
            [("business_id", {rating indexes}),..] -> i.e. [("ieuhg", {2, 3}]
    """
    # Get hash functions
    hash_fns = get_hash_family(num_signatures, num_shingles)
    # Apply minhash
    def apply_mh(rs):
        sign = []
        for h in hash_fns: # iterate over hash functions
            _minh = math.inf
            for r in rs:  # iterate over rating indexes
                tmp = h(r)
                if tmp < _minh:
                    _minh = tmp
            sign.append(_minh)
        assert sign.__len__() == len(hash_fns), "Error in shape"
        return sign
    # Apply minhash over matrix
    return rmat.mapValues(apply_mh)


