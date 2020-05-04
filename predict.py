""" Yelp Recommender predict module
"""
import sys

from pyspark import SparkConf, SparkContext

from config.config import *
from utils.misc import parse_predit_args, log

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
    args = parse_predit_args()
    # load config
    # create spark
    sc = create_spark()
    print(sc.parallelize([1,2]).collect())
