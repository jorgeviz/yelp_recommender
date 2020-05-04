""" INF 553 Assignment 3
    Task 3: Collaborative Filtering Recommendation System [TRAIN]

    Execution:
    python task3train.py ../../data/project/test_review_ratings.json userCFRS.model user_based
    python task3train.py ../../data/project/test_review_ratings.json itemCFRS.model item_based

    In this task, you will build collaborative filtering recommendation systems 
    with train reviews and use the models to predict the ratings for a pair of user 
    and business. You are required to implement 2 cases:

    • Case 1: Item-based CF recommendation system (2pts)
        In Case 1, during the training process, you will build a model by computing 
        the Pearson correlation for the business pairs that have at least 
        three co-rated users. During the predicting process, you will use the model 
        to predict the rating for a given pair of user and business. 
        You must use at most N business neighbors that are most similar to the 
        target business for prediction (you can try various N, e.g., 3 or 5).

    • Case 2: User-based CF recommendation system with Min-Hash LSH (2pts)
        In Case 2, during the training process, since the number of potential 
        user pairs might be too large to compute, you should combine the Min-Hash 
        and LSH algorithms in your user-based CF recommendation system. 
        You need to (1) identify user pairs who are similar using their co-rated 
        businesses without considering their rating scores (similar to Task 1). 
        This process reduces the number of user pairs you need to compare for the 
        final Pearson correlation score. (2) compute the Pearson correlation for 
        the user pair candidates that have Jaccard similarity >= 0.01 and at least 
        three co-rated businesses. The predicting process is similar to Case 1.
"""
import sys
import math
import re
import time
from pprint import pprint
import json
from pathlib import Path
from collections import OrderedDict, Counter
from operator import add
from pyspark import SparkContext, SparkConf
from minhash import minhash
from lsh import lsh, filter_valid
from utils import pearson_correlation
from collections import namedtuple
import ipdb

# Debugger
# import ipdb
DEBUG = False
#

# Params
N_SIGNATURES = 512
LSH_BANDS = 256
LSH_ROWS = N_SIGNATURES // LSH_BANDS
MIN_JACC_SIM = 0.01
MAX_PART_SIZE = 10 * (1024**2)
# Minimum number of Co-rated businesses to consider similar
MIN_CO_RATED = 2


def fetch_arg(_pos):
    """ Fetch arg from sys args
    """
    if len(sys.argv) <= _pos:
        raise Exception("Missing arguments!")
    return sys.argv[_pos]

def parse_args():
    """ Parse arguments
    """
    args = OrderedDict({
        "train_file": str, "model_file": str,
        "cf_type": str
    })
    for i, a in enumerate(args):
        args[a] = args[a](fetch_arg(i + 1))
    return args

def create_spark():
    """ Method to create Spark Context

        Returns:
        -----
        sc : pyspark.SparkContext
    """
    conf = SparkConf()\
        .setAppName("Task3")\
        .setMaster("local[3]")\
        .set("spark.executor.memory","4g")\
        .set("spark.driver.cores", "3")\
        .set("spark.driver.memory", "3g")
    sc = SparkContext(conf=conf)
    return sc

def read_file(sc, fpath):
    """ Read a file
    """
    _fsize = Path(fpath).stat().st_size
    return sc.textFile(fpath, _fsize // MAX_PART_SIZE )

def read_json(sc, fpath):
    """ Read JSON-rowed file parsed in to RDD
    """
    data = read_file(sc, fpath)\
            .map(lambda x: json.loads(x))
    return data

def read_csv(sc, fpath, with_heads=False):
    """ Read and parse CSV into RDD
    """
    def filter_heads(z): return z[1] > 0
    data = read_file(sc, fpath)\
        .zipWithIndex()\
        .filter(lambda z: True if with_heads else filter_heads(z))\
        .map(lambda z: tuple(z[0].split(',')))
    return data

def log(*msg, level="INFO"):
    """ Log message with visual help
    """
    print("-"*50)
    print("[{}]".format(level), end=" ")
    for m in msg:
        print("{}".format(m), end=" ")
    print("\n" + "-"*50)

def compute_w_log(msg, fn):
    """ Compute Spark action with 
        logged message and timestamp
    """
    if not DEBUG:
        return 
    t1 = time.time()
    z = fn()
    log(msg, z, 'in', time.time() - t1)
    ipdb.set_trace()

# ------------ Item Based CF ------------------------
def get_biz_ratings(data):
    """ Get business ratings 
    """
    log("Incoming business", data.map(lambda x: x['business_id']).distinct().count())
    biz_ratings = data\
        .map(lambda x: (x['business_id'], (x['user_id'], x['stars'])))\
        .groupByKey()
    biz_ratings.cache()
    #compute_w_log("Biz ratings:", biz_ratings.count)
    return biz_ratings, biz_ratings.sortByKey().keys()

def get_joined_biz_candidates(biz_data, biz_candids):
    """ Compute Joined RDD with candidates
        key and rating values

        biz_data: (business_id, (user1, user2))

        biz_candids: (business_id1, business_id2)

        (business_id1, business_id2, (val1, ...), (val2,...))
    """
    # join columns using cache of unique users
    BCache = namedtuple('BCache', ('set', 'dict'))
    biz_cache = biz_data\
        .mapValues(lambda x: BCache(set([j[0] for j in x]), dict(x)) )\
        .collectAsMap()
    joined_cands = biz_candids.map(lambda x: (
            (x[0], x[1]), (biz_cache[x[0]].set, biz_cache[x[1]].set)
        ) 
    )
    # filter the ones with less than Min Co-rated
    joined_cands = joined_cands\
        .mapValues(lambda v: v[0].intersection(v[1]))\
        .filter(lambda s: s[1].__len__() >= MIN_CO_RATED)
    compute_w_log("Joined Filtered:", joined_cands.count)
    # compute intersection
    def get_ratings_inters(x):
        (b1, b2), inters = x
        return {_k: (biz_cache[b1].dict[_k], biz_cache[b2].dict[_k])\
                for _k in inters}

    filtered_cands = joined_cands\
            .map(lambda x: (x[0], get_ratings_inters(x)) )
    compute_w_log("Intersection Candidates", filtered_cands.count)
    return filtered_cands

def get_item_based_cf(data):
    """ Get similar business pairs 
        and respective Pearson weights
    """
    # Fetch business ratings rdd
    biz_rdd, biz_keys = get_biz_ratings(data)
    # Generate candidates
    biz_candids = biz_keys.cartesian(biz_keys)\
                    .filter(lambda x: x[0] < x[1])
    #compute_w_log("Candidates", biz_candids.count)
    # Generate joined canddates rdd 
    biz_filtered = get_joined_biz_candidates(biz_rdd, biz_candids)
    # filter non-min co-rated pairs
    # biz_filtered = biz_joined\
    #         .filter(lambda x: x[1][2].__len__() >= MIN_CO_RATED)
    biz_filtered.cache()
    compute_w_log("Filtered possible Cands", biz_filtered.count)
    # Compute Pearson Correlation
    biz_corr = biz_filtered\
            .map(lambda x: (x[0], pearson_correlation(x[1])))
    compute_w_log("Candidates Corr", biz_corr.count)
    _biz_correls =  biz_corr.collect()
    log("Candidates Pairs:", len(_biz_correls))
    return _biz_correls

# ------------ End of Item Based CF -----------------

# ------------ User Based CF ------------------------

def get_rating_shingles(data):
    """ Map from user id to the Index of the 
        business at which gave a review
    """
    biz_map = dict(data.flatMapValues(lambda x: x)
                    .map(lambda x: x[1][0])
                    .distinct()
                    .zipWithIndex()
                    .collect())
    # group by user_id and reduce unique user indexes   
    user_ratings = data\
        .flatMapValues(lambda x: x)\
        .map(lambda x: (x[0], x[1][0]))\
        .groupByKey()\
        .mapValues(lambda x: set(biz_map[_k] for _k in set(x)))
    return user_ratings, biz_map

def minhash_lsh_candidates(data):
    """ Compute boolean minhash lsh to 
        yield candidates over ratings matrix
    """
    # get shingles
    user_ratings, biz_map = get_rating_shingles(data)
    compute_w_log("Rating Shingles", user_ratings.count)
    # minhash signatures
    user_signs = minhash(user_ratings, len(biz_map), N_SIGNATURES)
    compute_w_log("Minhash Signatures", user_signs.count)
    # compute LSH buckets
    user_candidates = lsh(user_signs, LSH_BANDS, LSH_ROWS)
    compute_w_log("LSH Candidates", user_candidates.count)
    # Join with ratings and compute jaccard sim
    cache_vects = dict(user_candidates\
        .map(lambda x: x[1])\
        .flatMap(lambda x: [(x_i, 1) for x_i in x ])\
        .join(user_ratings)\
        .map(lambda x: (x[0], x[1][1]))\
        .collect()
    )
    valid_user_cands = filter_valid(user_candidates, 
            cache_vects,
            MIN_JACC_SIM,
            serialize=False
        )
    valid_user_cands.cache()
    compute_w_log("Valid Candidates", valid_user_cands.count)
    # del cache_vects
    return valid_user_cands

def get_user_ratings(data):
    """ Get user's ratings 
    """
    user_ratings = data\
        .map(lambda x: (x['user_id'], (x['business_id'], x['stars'])))\
        .groupByKey()\
        .filter(lambda x: len(x[1]) >= MIN_CO_RATED)
    user_ratings.cache()
    compute_w_log("User ratings:", user_ratings.count)
    return user_ratings

def get_joined_user_candidates(user_data, user_cands):
    """ Join candidates with features
    """
    user_cache = dict(user_data.collect())
    joined_cands = user_cands.map(lambda x: (
            (x[0], x[1]), (user_cache[x[0]], user_cache[x[1]])
        ) 
    )
    # compute intersection
    def get_ratings_inters(x):
        v1, v2 = x; inters = {}
        v1 = dict(v1); v2 = dict(v2)
        for _k, _v1 in v1.items():
            if _k in v2:
                inters[_k] = (_v1, v2[_k])
        return v1, v2, inters

    joined_cands = joined_cands\
            .map(lambda x: (x[0], get_ratings_inters(x[1])))
    compute_w_log("Joined Cache Users:", joined_cands.count)
    return joined_cands

def get_user_based_cf(data):
    """ Get similar users by generating possible
        candidates over MinHash-LSH and computing
        respective pearson correlation
    """
    # Build users rdd -- (user, (biz, stars))
    users_rdd = get_user_ratings(data)
    # Fetch Candidates from MinHash-LSH
    user_candids = minhash_lsh_candidates(users_rdd)
    # Join candidates with features
    user_joined_cand = get_joined_user_candidates(users_rdd, user_candids)
    # filter those without enough co-rated
    user_filt_cand = user_joined_cand\
        .filter(lambda x: x[1][2].__len__() >= MIN_CO_RATED)
    compute_w_log("Filtered Cands", user_filt_cand.count)
    # Compute Pearson Correlation
    user_corrs = user_filt_cand\
            .map(lambda x: (x[0], pearson_correlation(x[1][2])))
    compute_w_log("Candidates Correl", user_corrs.count)
    return user_corrs.collect()

# ------------ End of User Based CF -----------------

if __name__ == "__main__":
    log("Starting Task 3 [Train]- Collaborative Filtering recommentation system")
    st_time = time.time()
    args = parse_args()
    log("Arguments: ", args)
    sc = create_spark()
    # Read data
    train_data = read_json(sc, args['train_file'])
    cf_type = args['cf_type']
    log("Loaded Training data")
    # Redirect to CF-type method
    if cf_type == "item_based":
        model_wgts = get_item_based_cf(train_data)
    else:
        model_wgts = get_user_based_cf(train_data)
    # Save model
    with open(args['model_file'], 'w') as mdl:
        k = ["b1", "b2"] \
                if cf_type == "item_based"\
                else ["u1", "u2"]
        for v in model_wgts:
            mdl.write(json.dumps({
                k[0]: v[0][0],
                k[1]: v[0][1],
                "stars": v[1]
            })+"\n")
    log("Finished Task 3 [TRAIN], Saved Model!")
    log("Job duration: ", time.time()-st_time)