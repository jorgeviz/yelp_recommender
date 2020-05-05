""" Yelp Recommender predict module
"""
import sys
import time
from pyspark import SparkConf, SparkContext
from models import models
from config.config import *
from utils.misc import parse_predit_args, log, read_json

def create_spark():
    """ Method to create Spark Context

        Returns:
        -----
        sc : pyspark.SparkContext
    """
    conf = SparkConf()\
        .setAppName(APP_NAME)\
        .setMaster("local[4]")\
        .set("spark.executor.memory", "4g")\
        .set("spark.executor.cores", "4")\
        .set("spark.driver.cores",  "2")\
        .set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    return sc

if __name__ == '__main__':
    log(f"Starting {APP_NAME} predicting ...")
    st_time = time.time()
    args = parse_predit_args()
    # load config
    cfg = load_conf()
    log(f"Using {cfg['class']}")
    # create spark
    sc = create_spark()
    # Load testing data
    testing = read_json(sc, args['test_file'])
    # Init model
    model = models[cfg['class']](sc, cfg)
    # Load model  and predict
    model.load_model()
    model.predict(testing, args['output_file'])
    # model.predict_debug(testing, args['output_file'])
    log(f"Finished predicting in {time.time() - st_time}")
