""" PySpark ALS Recommendation

    Alternate Least Squared matrix representation of Users and Items matrix, 
    not suitable for high ColdStart ratio of users at inference.
"""
import json
import itertools
from pathlib import Path
import os
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf, SQLContext

MAX_PART_SIZE = 10 * (1024**2)
os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

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
print("-"*50, '\n', "ALS CF Recommender System\n", "-"*50)
# Data
lines = read_json(sc, '/home/ccc_v1_s_YppY_173479/asn131942_7/asn131945_1/asnlib.0/publicdata/train_review.json')
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

# ####### TEST
# Evaluate the model by computing the RMSE on the test data
test = read_json(sc, '/home/ccc_v1_w_M2ZhZ_97044/asn131942_7/asn131945_1/work/test_review_ratings.json')\
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
testDF = spark.createDataFrame(testRDD).cache()
print("Test")
testDF.show(5)

# hyper parameters
ranks_ = [40, 50]
regs_ = [0.2, 0.4]

DF = ratings.union(testDF)
(training, val) = DF.randomSplit([0.9, 0.1])
training.show(5)


for (rg, r) in itertools.product(regs_, ranks_):
    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=18, rank=r, regParam=rg, userCol="userId", itemCol="bizId", ratingCol="rating", coldStartStrategy='nan')
    model = als.fit(training)
    predictions = model.transform(val)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    val_rmse = evaluator.evaluate(predictions)
    print('[VAL]','-'*50,'\nALS Rank:', r, 'Reg:', rg)
    print("[VAL] Root-mean-square error = " + str(val_rmse))
    model.save(f'als_18_double_reg{rg}_rank{r}.model')
    predictions = model.transform(testDF)
    # Coldstart
    predictions = predictions.fillna({'prediction': 2.5})
    rmse = evaluator.evaluate(predictions)
    print("[TEST] Root-mean-square error = ", rmse)

sc.stop()