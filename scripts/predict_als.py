""" PySpark ALS Recommendation

    Alternate Least Squared matrix representation of Users and Items matrix, 
    not suitable for high ColdStart ratio of users at inference.
"""
import sys
import json
import time
import itertools
from pathlib import Path
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf, SQLContext

st_time= time.time()
MAX_PART_SIZE = 10 * (1024**2)

os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

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
print("-"*50, '\n', "ALS CF Recommender System\n", "-"*50)
# Data
lines = read_json(sc, '/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/train_review.json')
parts = lines.map(lambda r: (r['user_id'], r['business_id'],r['stars']))
user_map = parts.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
print("Found Users: ", len(user_map))
biz_map = parts.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
print("Found Businesses: ", len(biz_map))

# ####### TEST
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

MODEL_NAME = 'als_18_double_reg0.2_rank50.model'
model = ALSModel.load(MODEL_NAME)
predictions = model.transform(testDF)
predictions = predictions.fillna({'prediction': 2.5}).cache() # Cold Start
print('Preds')
predictions.show(3)

#predictions.rdd.map(lambda r: wrj({'user_id': inv_idxs['user'][r.userId], 'business_id': inv_idxs['biz'][r.bizId], 'stars':r.prediction})).count()
preddf = predictions.toPandas()
preddf['user_id'] = preddf['userId'].apply(lambda x: inv_idxs['user'][x])
preddf['business_id'] = preddf['bizId'].apply(lambda x: inv_idxs['biz'][x])
preddf.rename(columns={'prediction':'stars'}, inplace=True)
with open(out_file, 'w') as f:
    for j in preddf[['user_id','business_id', 'stars']].to_dict(orient='records'):
        f.write(json.dumps(j)+'\n')

print("Done predictions!")
sc.stop()
print("Took: ", time.time() - st_time)