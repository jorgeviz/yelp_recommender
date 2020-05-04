""" Yelp Recommender training module
"""
import sys

from pyspark import SparkConf, SparkContext

from config.config import *
from models import models
from utils.misc import log, read_json

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
    log(f"Starting {APP_NAME} training ...")
    # load config
    cfg = load_conf()
    # create spark
    sc = create_spark()
    # Load training data
    training = read_json(sc, cfg['training_data'])
    # Init model
    model = models[cfg['class']](sc, cfg)
    print(training.take(2))
