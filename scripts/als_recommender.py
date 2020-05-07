""" PySpark ALS Recommendation

    Alternate Least Squared matrix representation of Users and Items matrix, 
    not suitable for high ColdStart ratio of users at inference.
"""
import json
import itertools
from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors  
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf, SQLContext

MAX_PART_SIZE = 10 * (1024**2)

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
        .setMaster("local[3]")\
        .set("spark.executor.memory","4g")\
        .set("spark.executor.cores", "4")\
        .set("spark.driver.cores", "2")\
        .set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    return sc


sc = create_spark()
spark =  SQLContext(sc)
print("-"*50, '\nALS CF Recommender System')
# Data
lines = read_json(sc, '../../data/project/train_review.json')
parts = lines.map(lambda r: (r['user_id'], r['business_id'],r['stars']))
user_map = parts.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
print("Found Users: ", len(user_map))
biz_map = parts.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
print("Found Businesses: ", len(user_map))
ratingsRDD = parts.map(lambda p: Row(
                                userId=int(user_map[p[0]]), 
                                bizId=int(biz_map[p[1]]),
                                rating=float(p[2])
                                )
            )
ratings = spark.createDataFrame(ratingsRDD)
(training, val) = ratings.randomSplit([0.9, 0.1])
training.show(5)

# hyper parameters
ranks_ = [2] #[8, 10, 12, 14, 16, 18, 20]
regs_ = [0.01] #[0.001, 0.01, 0.05, 0.1, 0.2]
niters = 1


import os
MODEL_NAME = 'weights/als_18_double_reg0.2_rank50.model'
if os.path.exists(MODEL_NAME):
    print("Loading model ....")
    model = ALSModel.load(MODEL_NAME)
    predictions = model.transform(val)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    val_rmse = evaluator.evaluate(predictions)
    print("[VAL] Root-mean-square error = " + str(val_rmse))
else:
    for (rg, r) in itertools.product(regs_, ranks_):
        # Build the recommendation model using ALS on the training data
        als = ALS(maxIter=niters, rank=r, regParam=rg, userCol="userId", itemCol="bizId", ratingCol="rating", coldStartStrategy='nan')
        model = als.fit(training)
        predictions = model.transform(val)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        val_rmse = evaluator.evaluate(predictions)
        print('-'*50,'\nALS Rank:', r, 'Reg:', rg)
        print("[VAL] Root-mean-square error = " + str(val_rmse))
        model.save(f'als_{niters}_reg{rg}_rank{r}.model')

# print("Business")
#model.itemFactors.show()

# print("Users")
# model.userFactors.show()

# ####### TEST
# Evaluate the model by computing the RMSE on the test data
test = read_json(sc, '../../data/project/test_review_ratings.json')\
        .map(lambda r: (r['user_id'], r['business_id'],r['stars']))
# Update Mappings
miss_biz = set(test.map(lambda x: x[1]).distinct().collect()) - set(biz_map)
for m in miss_biz:
    biz_map.update({m: biz_map.__len__()})
miss_user = set(test.map(lambda x: x[0]).distinct().collect()) - set(user_map)
for m in miss_user:
    user_map.update({m: user_map.__len__()})
testRDD = test.map(lambda p: Row(
                                userId=int(user_map[p[0]]), 
                                bizId=int(biz_map[p[1]]),
                                rating=float(p[2])
                                )
            )
inv_idxs = {
    "user": {v:k for k,v in user_map.items()},
    "biz": {v:k for k,v in biz_map.items()}
}
testDF = spark.createDataFrame(testRDD)
predictions = model.transform(testDF)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
# Coldstart
predictions = predictions.fillna({'prediction': 2.5}).cache()
def wrj(i):
    with open('ALS.preds', 'a') as f:
        f.write(json.dumps(i)+'\n')

predictions.rdd.map(lambda r: wrj({'user_id': inv_idxs['user'][r.userId], 'business_id': inv_idxs['biz'][r.bizId], 'stars':r.prediction})).count()
rmse = evaluator.evaluate(predictions)
print("[TEST] Root-mean-square error = ", rmse)

# k-NN predictions
biz_feats = model.itemFactors.orderBy('id').select('features').toPandas()\
                .features.apply(pd.Series).to_numpy()
knn_mdl = NearestNeighbors(n_neighbors=500, algorithm='brute', metric='cosine')\
            .fit(biz_feats)
testbizs = [b_ix for b,b_ix in biz_map.items() if b_ix < biz_feats.shape[0]]
rates_by_user = ratingsRDD.map(lambda r: (r.userId, (r.bizId, r.rating)))\
                        .groupByKey().mapValues(dict).collectAsMap()
neighs_cache = knn_mdl.kneighbors(biz_feats[testbizs])
neighs_cache = {t: (neighs_cache[0][j], neighs_cache[1][j]) for j, t in enumerate(testbizs)}
kpreds = testDF.rdd.map(lambda r: Row(
                    bizId=r.bizId, userId=r.userId, 
                    rating=r.rating, neighs=neighs_cache.get(r.bizId,([],[])),
                    userRates=rates_by_user.get(r.userId, {})
                    )).cache()
def w_avg(ngs, rts):
    num_, den_ = 0., 0.
    for w_i, n_i in zip(*ngs):
        if n_i in rts:
            num_ += (w_i*rts[n_i])
            den_ += abs(w_i)
    if den_ == 0:
        # Cold start 
        return 2.5
    return num_ / den_

print("Inter stats", kpreds.map(lambda r:  set(r.userRates).intersection(set(r.neighs[0])).__len__()).stats() )
kpredsDF = kpreds.map(lambda r: Row(userId=r.userId, bizId=r.bizId, rating=r.rating, prediction=float(w_avg(r.neighs, r.userRates)))).toDF()
print("KNN- preds")
kpredsDF.show(5)
krmse = evaluator.evaluate(kpredsDF)
print("[TEST] K-NN Root-mean-square error = ", krmse)
print("No Available resp:", kpredsDF.filter(kpredsDF.prediction == 2.5).count())
def wrj(i):
    with open('kNNALS.preds', 'a') as f:
        f.write(json.dumps(i)+'\n')

kpredsDF.rdd.map(lambda r: wrj({'user_id': inv_idxs['user'][r.userId], 'business_id': inv_idxs['biz'][r.bizId], 'stars':r.prediction})).count()
sc.stop()