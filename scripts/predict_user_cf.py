""" INF 553 Assignment 3
    Task 3: Collaborative Filtering Recommendation System [PREDICT]

    Execution:
    python task3predict.py ../../data/project/train_review.json ../../data/project/test_review_ratings.json userCFRS.model userCFRS.preds  user_based
    python task3predict.py ../../data/project/train_review.json ../../data/project/test_review_ratings.json itemCFRS.model itemCFRS.preds  item_based

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
import time
import math
from pprint import pprint
import json
from pathlib import Path
from collections import OrderedDict
from pyspark import SparkContext, SparkConf
from utils import pearson_correlation

# Params
N_NEIGHS = 5
MIN_SIM = 0.01
MAX_PART_SIZE = 10 * (1024**2)

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
        "train_file": str,
        "test_file": str, "model_file": str,
        "output_file": str, "cf_type": str
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
        .setAppName("Task1")\
        .setMaster("local")\
        .set("spark.executor.memory","4g")
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

def load_model(sc, mdl_file, cf_):
    """ Parse model file 
        
        Returns:
        ----
        (user_i, biz_i, sim_i)
    """
    mdl_ = read_json(sc, mdl_file)
    if cf_ == "item_based":
        return mdl_.map(lambda x: (x["b1"], x['b2'], x['stars']))
    return mdl_.map(lambda x: (x["u1"], x['u2'], x['stars']))

# ------------ Item Based CF ------------------------
def infer_item_based_cf(model, train, test):
    """ Get nearest neighbors from business,
        and compute rating
    """
    # build access index
    it_idx_one = build_itemb_idxs(model).collectAsMap()
    def _get_neighs(b):
        if b in it_idx_one:
            return sorted(it_idx_one[b].items(), 
                            key=lambda y: y[1], 
                            reverse=True)[:N_NEIGHS]
        else:
            return []
    # get k-NN from business (u, (b, k-ratings))
    d_neighs = test.map(lambda x: (x['business_id'], x['user_id']) )\
                    .map(lambda x: (
                        x[1], (
                            x[0], 
                            _get_neighs(x[0])
                        )
                    ))
    # Get users cache
    test_users = set(d_neighs.map(lambda x: x[0])\
                     .distinct().collect())
    users_ratings = train.map(lambda x: (x['user_id'], (x['business_id'], x['stars'] )) )\
                        .groupByKey()\
                        .filter(lambda x: x[0] in test_users)\
                        .mapValues(dict).collectAsMap()
    # compute prediction score (user, biz, score)
    def get_score(u_rates, neighs):
        num_, den_ = 0, 0
        for n,nv in neighs:
            if n in u_rates:
                num_ += u_rates[n]*nv
                den_ += abs(nv)
        if den_ == 0:
            return 0
        return num_ / den_
    scores = d_neighs\
                .map(lambda x: (
                    x[0], 
                    x[1][0], 
                    get_score(users_ratings.get(x[0],{}), x[1][1] )) 
                )
    _scores = scores.collect()
    log("Scores detected:", len(_scores))
    return _scores

def build_itemb_idxs(data):
    """ Build fast access idx for similarity
    """
    _left = data.map(lambda x: (x[0], (x[1], x[2])) )
    _right = data.map(lambda x: (x[1], (x[0], x[2])) )
    # union
    ib_idx = _left.union(_right)\
                .groupByKey()\
                .mapValues(dict)
    ib_idx.cache()
    return ib_idx

# ------------ End Item Based CF ------------------------

# ------------ User Based CF ------------------------
def infer_user_based_cf(model, train, test):
    """ Infer user CF Vanilla
    """ 
    from statistics import mean
    from collections import namedtuple
    # Users' mean ratings - (user, mean_rating)
    mean_user_ratings = train.map(lambda x: (x['user_id'],x['stars']))\
                            .groupByKey()\
                            .mapValues(mean).collectAsMap()
    # Format Hash table of user weights - ((user1, user2), sim)
    users_weights = model.flatMap(lambda x: [
                            ((x[0], x[1]), x[2]),
                            ((x[1], x[0]), x[2])
                        ]).collectAsMap()
    # User neighbors - (u, {u1,u2})
    user_neighs = model\
            .flatMap(lambda x: [
                    (x[0], x[1]),
                    (x[1], x[0])])\
            .groupByKey()\
            .mapValues(set).collectAsMap()
    # rating's index by business
    biz_ratings = train.map(lambda x: (x['business_id'], (x['user_id'], x['stars'] )) )\
                        .groupByKey().mapValues(dict)\
                        .collectAsMap()
    # get predictions
    def _get_score(u, neighs, rates):
        num_, den_ = [], []
        for n in neighs:
            if n in rates:
                w = users_weights[(u,n)]
                ra_i = rates[n]
                _ra = mean_user_ratings.get(n, 0)
                num_.append(w*(ra_i-_ra))
                den_.append(abs(w))
        if len(den_) == 0 or sum(den_) == 0:
            return 0
        return mean_user_ratings.get(u, 0) + (sum(num_)/sum(den_))

    Rating = namedtuple("Rating", ("user", "biz"))
    preds = test.map(lambda x: Rating(x['user_id'], x['business_id']))\
                .map(lambda x: (
                    x.user, x.biz,
                    _get_score(x.user, user_neighs.get(x.user, set()), biz_ratings.get(x.biz, {}))
                ))
    _preds = preds.collect()
    return _preds

# ------------ End user Based CF ------------------------
    

if __name__ == "__main__":
    log("Starting Task 3 [Predict]- Colaborative Filtering recommentation system")
    args = parse_args()
    log("Arguments: ", args)
    sc = create_spark()
    # Read files
    train_data = read_json(sc, args['train_file'])
    test_data = read_json(sc, args['test_file'])
    model = load_model(sc, args['model_file'], args['cf_type'])
    log("Loaded Model and Test data")
    # compute inference
    if args['cf_type'] == "item_based":
        predictions = infer_item_based_cf(model, train_data, test_data)
    else:
        predictions = infer_user_based_cf(model, train_data, test_data)
    log("Got Predictions")
    # Write predictions
    with open(args['output_file'], 'w') as of:
        for pv in predictions:
            of.write(json.dumps({
                "user_id": pv[0], "business_id": pv[1],
                "stars": pv[2]
            })+"\n")
    log("Finished Task 3 [PREDICT], Saved Predictions!")
    
    