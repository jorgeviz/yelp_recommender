""" Evaluation module t
    Testing RSME given GT file and Predicted elements, with JSON formatted
    rows as follows.

    {"user_id": "XXXXX", "business_id": "YYYYY", "stars": 5.0}
    {"user_id": "XXYXX", "business_id": "YZYYY", "stars": 2.0}
    ...

    Execution:

    python evaluate.py <predicted.json> <ground_truth.json>
"""
import sys
import json
import math
import random
from pathlib import Path
from collections import OrderedDict, Counter
from pyspark import SparkContext, SparkConf

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
        "pred_file": str, "gt_file": str
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
        .setAppName("RMSE-Evaluator")\
        .setMaster("local")\
        .set("spark.executor.memory","4g")\
        .set("spark.driver.cores", "4")\
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

def log(*msg, level="INFO"):
    """ Log message with visual help
    """
    print("-"*50)
    print("[{}]".format(level), end=" ")
    for m in msg:
        print("{}".format(m), end=" ")
    print("\n" + "-"*50)

def format_dsets(preds, gts):
    """ Format datasets
    """
    pred_stars = preds.map(lambda x: ((x['user_id'], x['business_id']), x['stars'])).collectAsMap()
    gt_stars = gts.map(lambda x: ((x['user_id'], x['business_id']), x['stars'])).collectAsMap()
    return pred_stars, gt_stars

def compute_rmse(p_stars, gt_stars, log_pred=True):
    """ Compute RMSE
    """
    missing = []
    _mse = 0
    for k, gt in gt_stars.items():
        pred = p_stars.get(k, None)
        if pred is None or str(pred) == 'nan':
            missing.append(k)
            continue
        _mse += (gt - pred)**2
        if (random.uniform(0,1) < 0.001) and log_pred:
            log("Prediction: ", pred, "GT:", gt)
    if (len(gt_stars)-len(missing)) == 0:
        return 'N/A', missing
    return math.sqrt(_mse/(len(gt_stars)-len(missing))), missing

def compute_decision_rmse(pred, gt):
    """ Compute RMSE per decision rule
    """
    decis_map = {
        'biz_avg': 'Business Avg',
        'usr_avg': 'User Avg',
        'cos': 'Cosine Sim',
        'default': 'Default'
    }
    for k, dec in decis_map.items():
        p_stars, gt_stars = format_dsets(
            pred.filter(lambda x: x['decision'] == k), 
            gt
        )
        rmse, missing_preds = compute_rmse(p_stars, gt_stars, log_pred=False)
        log("*"*20, dec, "*"*20)
        log("RMSE:", rmse)
        log("Number of Predictions", len(p_stars))

if __name__ == "__main__":
    log("Starting RMSE Evaluation ...")
    args = parse_args()
    log("Arguments: ", args)
    sc = create_spark()
    # Read data
    preds = read_json(sc, args['pred_file'])
    gts = read_json(sc, args['gt_file'])
    log("Loaded data")
    # Start evaluation
    p_stars, gt_stars = format_dsets(preds, gts)
    rmse, missing_preds = compute_rmse(p_stars, gt_stars)
    log("RMSE:", rmse)
    log("Missing predictions:", len(missing_preds))
    if 'decision' in preds.take(1)[0]:
        compute_decision_rmse(preds, gts)