""" INF 553 Assignment 3
    Task 2: Content Based Recommendation System [PREDICT]

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
import time
from pprint import pprint
import json
from pathlib import Path
from collections import OrderedDict
from pyspark import SparkContext, SparkConf
from utils import cosine_similarity

# Params
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
        "test_file": str, "model_file": str,
        "output_file": str
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

def load_model(mdl_file):
    """ Parse model file 
        
        Returns:
        ----
        (biz_prof, user_prof, top_terms, terms_idx)
    """
    with open(mdl_file, "r") as buff:
        mdl_ = json.loads(buff.read())
    return dict(mdl_['business_profiles']), \
            dict(mdl_['user_profiles']), \
            mdl_['top_terms'], mdl_['terms_pos_idx']

def find_similarity(data, biz, users):
    """ Find cosine similarity over provided data
    """
    def _sim(x,y):
        # similarity with cold start = 0
        if (x and y):
            return cosine_similarity(x, y)
        return 0
    sim_data = data.map(lambda x: (x['user_id'], x['business_id']))\
                .map(lambda x: (x[0], x[1], users.get(x[0], []), biz.get(x[1], [])) )\
                .map(lambda x: (x[0], x[1], 5*_sim(x[2],x[3])) )
    return sim_data 
    

if __name__ == "__main__":
    log("Starting Task 2 [Predict]- Content based recommentation system")
    args = parse_args()
    log("Arguments: ", args)
    sc = create_spark()
    # Read files
    test_data = read_json(sc, args['test_file'])
    biz_prof, user_prof, top_terms, terms_idx = load_model(args['model_file'])
    log("Loaded Model and Test data")
    # compute similarity
    predictions = find_similarity(test_data, biz_prof, user_prof)
    pred_vals = predictions.collect()
    log("Got Predictions")
    # Write predictions
    with open(args['output_file'], 'w') as of:
        for pv in pred_vals:
            of.write(json.dumps({
                "user_id": pv[0], "business_id": pv[1],
                "stars": pv[2]
            })+"\n")
    log("Finished Task 2 [PREDICT], Saved Predictions!")
    
    