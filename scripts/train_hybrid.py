""" Hybrid Recommendation using ALS and MLP to estimate score
"""
import json
import itertools
from pathlib import Path
import os
import numpy as np
import pandas as pd
import time
from scipy import sparse
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf, SQLContext

st_time = time.time()
MAX_PART_SIZE = 10 * (1024**2)
os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'
train_file = '../../data/project/train_review.json' #'/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/train_review.json'

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

def create_spark():
    """ Method to create Spark Context

        Returns:
        -----
        sc : pyspark.SparkContext
    """
    conf = SparkConf()\
        .setAppName("ALS")\
        .setMaster("local[*]")\
        .set("spark.executor.memory","4g")\
        .set("spark.driver.cores", "2")\
        .set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    return sc

sc = create_spark()
spark =  SQLContext(sc)
print("-"*50, '\n', "ALS CF Hybrid Recommender System\n", "-"*50)
# Data
lines = read_json(sc, train_file)
parts = lines.map(lambda r: (r['user_id'], r['business_id'],r['stars']))
user_map = parts.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
print("Found Users: ", len(user_map))
biz_map = parts.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
print("Found Businesses: ", len(biz_map))
ratingsRDD = parts.map(lambda p: Row(
                                userId=int(user_map[p[0]]), 
                                bizId=int(biz_map[p[1]]),
                                rating=float(p[2])
                                )
            )
ratings = spark.createDataFrame(ratingsRDD).cache()
# (training, val) = ratings.randomSplit([0.9, 0.1])

#############################################
# ALS
#############################################
# hyper parameters
ranks_ = [50]
regs_ = [0.2]
niters = 1

for (rg, r) in itertools.product(regs_, ranks_):
    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=niters, rank=r, regParam=rg, userCol="userId", 
                itemCol="bizId", ratingCol="rating", coldStartStrategy='nan')
    model = als.fit(ratings)
    predictions = model.transform(ratings)
    predictions = predictions.fillna({'prediction': 2.5})
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    # val_rmse = evaluator.evaluate(predictions)
    # print('[VAL]','-'*50,'\nALS Rank:', r, 'Reg:', rg)
    # print("[VAL] Root-mean-square error = " + str(val_rmse))
    model.save(f'als_double_reg{rg}_rank{r}.model')
    print("Saved ALS model!")
#############################################
# MLP
#############################################
avgs_files ={
    'UAVG': '../../data/project/user_avg.json', #/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/user_avg.json
    'BAVG': '../../data/project/business_avg.json' #    'BAVG': '/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/business_avg.json'
}

def train_model(X, Y):
    model = MLPRegressor(hidden_layer_sizes=(30,10,30), 
                        activation='relu',
                        alpha=0.005,
                        learning_rate='adaptive',
                        learning_rate_init=1e-2,
                        max_iter=50, verbose=True)
    model.fit(X,Y)
    np.save("hybridMLP.model", model)
    return model

def read_avgs(data, avgs):
    # averages
    for _a, _af in avgs.items():
        with open(_af, 'r') as _f:
            acache = json.load(_f)
        _dmean = np.mean([ij for ij in acache.values()])
        _col = 'user_id' if _a.startswith('U') else 'business_id'
        data[_a] = data[_col].apply(lambda v: acache.get(v, _dmean))
    return data

# decoding indexes 
inv_idxs = {
    "user": {v:k for k,v in user_map.items()},
    "biz": {v:k for k,v in biz_map.items()}
}
# Formating features 
feats = predictions.toPandas().rename(columns={'prediction': 'ALS'})
feats['user_id'] = feats['userId'].apply(lambda x: inv_idxs['user'][x])
feats['business_id'] = feats['bizId'].apply(lambda x: inv_idxs['biz'][x])
feats = read_avgs(feats, avgs_files)
print("Features:\n", feats[['ALS', 'UAVG', 'BAVG']].head(5))
model = train_model(feats[['ALS', 'UAVG', 'BAVG']], 
                    feats['rating'])
print("Saved MLP model!")

print("Took:", time.time() - st_time)