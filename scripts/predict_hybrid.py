import sys
import json
import time
import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf, SQLContext

st_time= time.time()
MAX_PART_SIZE = 10 * (1024**2)

os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

train_file = '../../data/project/train_review.json'  # '/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/train_review.json'
test_file = sys.argv[1]
out_file = sys.argv[2]

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
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    return sc

sc = create_spark()
spark =  SQLContext(sc)
print("-"*50, '\n', "ALS CF Hybrid Recommender System [Prediction]\n", "-"*50)
# Data
lines = read_json(sc, train_file)
parts = lines.map(lambda r: (r['user_id'], r['business_id'],r['stars']))
user_map = parts.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
print("Found Users: ", len(user_map))
biz_map = parts.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
print("Found Businesses: ", len(biz_map))

# -- TEST
# Evaluate the model by computing the RMSE on the test data
test = read_json(sc, test_file)\
        .map(lambda r: (r['user_id'], r['business_id']))
# Update Mappings
miss_biz = set(test.map(lambda x: x[1]).distinct().collect()) - set(biz_map)
for m in miss_biz:
    biz_map.update({m: biz_map.__len__()})
miss_user = set(test.map(lambda x: x[0]).distinct().collect()) - set(user_map)
for m in miss_user:
    user_map.update({m: user_map.__len__()})
testRDD = test.map(lambda p: Row(
                                userId=int(user_map[p[0]]), 
                                bizId=int(biz_map[p[1]])
                                )
            )
testDF = spark.createDataFrame(testRDD).cache()
print("Test")
testDF.show(5)


# decoding indexes 
inv_idxs = {
    "user": {v:k for k,v in user_map.items()},
    "biz": {v:k for k,v in biz_map.items()}
}

#############################################
# ALS
#############################################
MODEL_NAME = 'als_double_reg0.2_rank50.model'
als_model = ALSModel.load(MODEL_NAME)
predictions = als_model.transform(testDF)
predictions = predictions.fillna({'prediction': 2.5}).cache() # Cold Start
print('Preds')
predictions.show(3)

#############################################
# MLP
#############################################
avgs_files ={
    'UAVG': '../../data/project/user_avg.json', #/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/user_avg.json
    'BAVG': '../../data/project/business_avg.json' #  '/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/business_avg.json'
}

def load_model():
    model = np.load('hybridMLP.model.npy', 
            allow_pickle=True)
    return model.item()

def read_avgs(data, avgs):
    # averages
    for _a, _af in avgs.items():
        with open(_af, 'r') as _f:
            acache = json.load(_f)
        _dmean = np.mean([ij for ij in acache.values()])
        _col = 'user_id' if _a.startswith('U') else 'business_id'
        data[_a] = data[_col].apply(lambda v: acache.get(v, _dmean))
    return data

mlp_model = load_model()
feats = predictions.toPandas()
feats['user_id'] = feats['userId'].apply(lambda x: inv_idxs['user'][x])
feats['business_id'] = feats['bizId'].apply(lambda x: inv_idxs['biz'][x])
feats.rename(columns={'prediction':'ALS'}, inplace=True)
feats = read_avgs(feats, avgs_files)
print("Features:\n", feats[['ALS', 'UAVG', 'BAVG']].head(5))
feats['stars'] = mlp_model.predict(feats[['ALS', 'UAVG', 'BAVG']])

# Save
with open(out_file, 'w') as f:
    for j in feats[['user_id','business_id', 'stars']].to_dict(orient='records'):
        f.write(json.dumps(j)+'\n')

print("Done predictions!")
sc.stop()
print("Took: ", time.time() - st_time)