""" INF 553 Assignment 3
    Task 2: Content Based Recommendation System [TRAIN]

    In this task, you will build a content-based recommendation system by generating 
    profiles from review texts for users and businesses in the train review set. 
    Then you will use the system/model to predict if a user prefers to review 
    a given business, i.e., computing the cosine similarity between the user 
    and item profile vectors.

    During the training process, you will construct business and user 
    profiles as the model:
        a. Concatenating all the review texts for the business as 
            the document and parsing the document, such as removing the 
            punctuations, numbers, and stopwords. Also, you can remove 
            extremely rare words to reduce the vocabulary size, 
            i.e., the count is less than 0.0001% of the total words.
        b. Measuring word importance using TF-IDF, 
            i.e., term frequency * inverse doc frequency
        c. Using top 200 words with highest TF-IDF scores to describe the document
        d. Creating a Boolean vector with these significant words 
            as the business profile
        e. Creating a Boolean vector for representing the user profile by 
            aggregating the profiles of the items that the user has reviewed
    
    During the predicting process, you will estimate if a user would prefer 
    to review a business by computing the cosine distance between the profile vectors. 
    The (user, business) pair will be considered as a valid pair if their 
    cosine similarity is >= 0.01. You should only output these valid pairs.
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
from lsh import lsh

# Params
MIN_COS_SIM = 0.01
RARE_WORDS_PERC = 0.0001
TOP_TFIDF = 200
LSH_BANDS = 200
LSH_ROWS = TOP_TFIDF // LSH_BANDS
MAX_PART_SIZE = 10 * (1024**2)

# Aux vars
puncts = [ "(", "[", ",", ".", "!", "?", ":", ";", "]", ")" ,"\n", 
    "*", "/", " ", "$", "'", '"', '-', '\r', '#']
puncts_re = r"(\(|\[|\,|\.|\!|\?|\:|\;|\]|\)|\n|\*|\/|\$|\'|\"|-|\#|\r)"

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
        "stopwords": str
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
        .set("spark.executor.memory","4g")\
        .set("spark.driver.cores", "2")\
        .set("spark.driver.memory", "2g")
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

def parse_biz_reviews(data):
    """ Join all reviews by business
    """
    biz_data = data.map(lambda x: (x[0][0], x[1]))\
                    .mapValues(list)\
                    .groupByKey()
    return biz_data

def parse_text_data(data, stops, puncts):
    """ Parse text, removing stop words,
        punctuation and empy spaces
    """
    def extend_puncts(l):
        return re.sub(puncts_re, lambda x: " "+x.group(0)+" ", l)

    remove_set = set(puncts+stops+[''])
    parsed_data = data.map(lambda x: ((x['business_id'], x['user_id']), x['text']))\
                    .map(lambda x: (x[0], extend_puncts(x[1].lower()).split(' ')))\
                    .zipWithIndex().map(lambda z: ((z[1], z[0][0]), z[0][1]))\
                    .flatMapValues(lambda x: x)\
                    .filter(lambda x: x[1] not in remove_set)\
                    .groupByKey().map(lambda x: (x[0][1], x[1]))
    return parsed_data

def parse_user_reviews(data):
    """ Join reviews done by user
        and parse text, removing stop words,
        punctuation and empty spaces
    """
    user_data = data.map(lambda x: (x[0][1], x[1]))\
                    .mapValues(list)\
                    .groupByKey()
    return user_data

def count_terms_freq(data):
    """ Count terms' frequency
    """
    mapped = data.flatMapValues(lambda x: x)\
                .flatMapValues(lambda x: x)\
                .map(lambda x: (x[1], x[0]))\
                .groupByKey()
    # Document frequency
    doc_freq = mapped.mapValues(set)\
                    .mapValues(len)
    #### Count terms routine
    # print("Overall count")
    # overall_term_count = mapped\
    #                         .mapValues(len)\
    #                         .sortBy(lambda x: x[1], ascending=False)
    # pprint(overall_term_count.take(5))
    return dict(doc_freq.collect())

def compute_tfidf(data,  doc_fq, N):
    """ Compute TF-IDF vectors
    """
    # TF  (biz_id, {t1: 3, t2: 4})
    def normed_tf(x, norm=False):
        k, v  = x
        c = Counter()
        for _k in v:
            c.update(_k)
        if norm:
            _max = c.most_common()[0][1]
            return (k, {i: j/_max for i,j in c.items()})
        return (k, {i: j for i,j in c.items()})
    tfq = data.mapValues(list).map(normed_tf)
    # IDF (biz_id, {t1: (2,3,4.5), t2: (1,3,6.7)}
    def _tfidf(val):
        k, _tf = val
        d = {}
        for term,v in _tf.items():
            _df = doc_fq.value[term]
            d[term] = (v, _df, v * math.log(N/_df, 2))
        return (k, d)
    tfidf = tfq.map(_tfidf)
    # [TODO] ---- change the way most common terms are selected
    # Get most recent elements, top terms {t1: 23.4, t2:6.4} 
    top_terms = tfidf.flatMap(lambda x: [(_k, _j)  for _k, _j in x[1].items()])\
                    .filter(lambda x: x[1][1] > 1)\
                    .map(lambda x: (x[0], x[1][2]))\
                    .groupByKey()\
                    .mapValues(max)
    top_terms = OrderedDict(
        top_terms
            .sortBy(lambda x: x[1], ascending=False)\
            .take(TOP_TFIDF)
    )
    # [TODO] ---- change the way to construct pos index
    top_idx = {_ky:_i for _i,_ky in enumerate(top_terms)}
    log("Got Top Terms")
    # pprint(tfidf
    #     .flatMap(lambda x: x[1].items())
    #     .filter(lambda x: x[0] in top_terms)
    #     .take(50))
    return tfidf, top_terms, top_idx

def get_profile(feats, top_terms, top_idx):
    # One-hot encode
    def one_hot_tdf(x):
        one_hot = [0]*TOP_TFIDF
        for w in x[1]:
            if w in top_terms:
                one_hot[top_idx[w]] = 1
        return (x[0], one_hot)
    feat_onehot = feats.map(one_hot_tdf)
    log("One-hot TF-IDF features")
    return feat_onehot

def build_user_profiles(data, terms_idx):
    """ Build User profiles
    """
    # Parse text documents
    user_data = parse_user_reviews(data)\
                .flatMapValues(lambda x: x)\
                .flatMapValues(lambda x: x)\
                .groupByKey()\
                .mapValues(set)
    user_prof = get_profile(user_data, terms_idx, terms_idx)
    return user_prof


if __name__ == "__main__":
    log("Starting Task 2 [Train]- Content based recommentation system")
    st_time = time.time()
    args = parse_args()
    log("Arguments: ", args)
    sc = create_spark()
    # Read data
    train_data = read_json(sc, args['train_file'])
    stop_ws = read_file(sc, args['stopwords']).collect()
    log("Loaded Training data")
    # Parse text data
    parsed_data = parse_text_data(train_data, stop_ws, puncts)
    parsed_data.cache()
    # Parse business reviews
    biz_data = parse_biz_reviews(parsed_data)
    biz_data.cache(); t1 = time.time()
    N_biz = biz_data.count()
    log("Got Count of businesses: ", N_biz, "in", time.time()-t1)
    # Encode business features with top TF-IDF
    doc_freq = sc.broadcast(count_terms_freq(biz_data))
    log("Got document frequency of {} terms"
        .format(doc_freq.value.__len__()))
    tf_idf, top_terms, top_idx = compute_tfidf(biz_data, doc_freq, N_biz) 
    # Build Business profile
    biz_prof = get_profile(tf_idf, top_terms, top_idx)
    # Build user profiles
    user_prof = build_user_profiles(parsed_data, top_idx)
    #### Train Classifier 
    # LSH and compute Cosine similarity over candidates
    ###### Generate buckets (Not useful right now)
    ###### biz_buckets = lsh(biz_prof, LSH_BANDS, LSH_ROWS)
    
    # -- Store model info (TF-IDF vector, top_terms, lsh_buckets)
    biz_prof_val = biz_prof.collect()
    log("Fetched profiles")
    user_prof_val = user_prof.collect()
    log("Fetched users")
    with open(args['model_file'], 'w') as mdl:
        mdl.write(json.dumps({
            "business_profiles": biz_prof_val,
            "user_profiles": user_prof_val,
            "top_terms": top_terms,
            "terms_pos_idx": top_idx 
        }))
    log("Finished Task 2 [TRAIN], Saved Model!")
    log("Job duration: ", time.time()-st_time)