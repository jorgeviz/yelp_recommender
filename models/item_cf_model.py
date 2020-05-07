from itertools import combinations
import json

from pyspark.ml.feature import MinHashLSH
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import  *
from pyspark.sql import functions as F
from pyspark.ml.linalg import SparseVector, Vectors
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score

from models.base_model import BaseModel
from utils.misc import log, debug, read_json
from utils.metrics import mean
from utils.metrics import cosine_similarity



class ItemBasedCFModel(BaseModel):

    def __init__(self, sc, cfg):
        """ Item-based CF model constructor
        """
        super().__init__(sc,cfg)
        self._sql = SQLContext(self._sc)
        # params
        self.min_corrated = self.cfg['hp_params']['MIN_CORRATED']
        self.n_minhashes = self.cfg['hp_params']['N_MIN_HASHES']
        self.icf_metric = self.cfg['hp_params']['METRIC']
        self.k_n = self.cfg['hp_params']['K_NEIGHS']

    def get_ratings_by_business(self, data):
        """ Read and format ratings

            Params:
            -----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
            
            Returns: pyspark.dataframe
                Cols: ['biz', 'user', 'stars']
        """
        rates = self._sql.createDataFrame(
            data.map(lambda x: (x['business_id'], x['user_id'],x['stars'])),
            ['biz', 'user', 'stars']
        )
        rates_ = rates.toPandas()
        rates_by_biz = rates_.groupby('biz')
        log("Got ratings by business!")
        return rates_by_biz['user'].apply(set).to_dict(), \
                rates_.set_index(['biz', 'user']), \
                rates.rdd.map(lambda r: r.biz).distinct().sortBy(lambda x: x)
    
    def prepare(self, data):
        """ Fetch business candidates

            Params:
            -----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
        """
        if 'lsh':
            return self.local_hashing_candidates(data)
        # brute force comparison
        b_rate_set, rates_df, unique_biz = self.get_ratings_by_business(data)
        # Fetch permutations
        _min_corrated = self.min_corrated
        _permuts = unique_biz.cartesian(unique_biz).filter(lambda x: x[0] < x[1])
        _valid = _permuts.map(lambda x: (x, b_rate_set[x[0]], b_rate_set[x[1]]))\
                        .map(lambda x: (x[0], x[1].intersection(x[2]) ))\
                        .map(lambda x: (x[0], (len(x[1]), x[1]) ))\
                        .filter(lambda x: x[1][0] >= _min_corrated)
        valid_df = pd.DataFrame(_valid.collect())
        return valid_df, None, rates_df, None
    
    def local_hashing_candidates(self, data):
        """ MinHashLSH from hash

            Params:
            -----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
            
            Returns:
            -----
            candidates: pyspark.rdd 
                [(busines_i_id, business_j_id), ...]
            business_feats: pyspark.dataframe
                ['biz_id','biz','features']
            rate: pyspark.dataframe
                ['biz','user','stars']
            user_map: dict
                {user_key: user_pos, ...}
        """
        # Convert rates 
        rates = self._sql.createDataFrame(
            data.map(lambda x: (x['business_id'], x['user_id'],x['stars'])),
            ['biz', 'user', 'stars']
        ).cache()
        user_map = rates.rdd.map(lambda r: r.user).distinct().zipWithIndex().collectAsMap()
        num_users = len(user_map)
        business_feats = self._sql.createDataFrame(
            rates.rdd\
                .groupBy(lambda r: r.biz)\
                .mapValues(lambda vs: SparseVector(num_users, 
                                        {user_map[v.user]: v.stars for v in vs}))\
                .zipWithIndex()\
                .map(lambda r: (r[1], r[0][0],r[0][1]) ),
            ['biz_id', 'biz', 'features']
        ).cache()
        # Run MinHash LSH 
        mh_lsh = MinHashLSH(inputCol="features", outputCol="hashes", 
                            numHashTables=self.n_minhashes, seed=12345)\
                .fit(business_feats.select('biz_id','features'))
        preds = mh_lsh.transform(business_feats.select('biz_id','features'))
        # Find candidates -- [TODO double-check candidate generation]
        candidates = preds.rdd.map(lambda r: Row(biz_id=r.biz_id, hashes=tuple(h[0] for h in r.hashes)))\
                        .groupBy(lambda r: r.hashes)\
                        .mapValues(lambda l: [j.biz_id for j in l])\
                        .filter(lambda v: len(v[1]) >= 2)\
                        .map(lambda x: x[1])\
                        .flatMap(lambda x: combinations(x, 2))
        log("Candidates", candidates.take(3))
        return candidates, business_feats, rates, user_map

    def compute_weights(self, candidates, features):
        """ Compute Pearson correlation between vectors

            Params:
            -----
            candidates: pyspark.rdd 
                [(busines_i_id, business_j_id), ...]
            business_feats: pyspark.dataframe
                ['biz_id','biz','features']
        """
        features_ = features.rdd.map(lambda r: (r.biz_id, r.features)).collectAsMap()
        cfeats = candidates.map(lambda c: (
                            c, Row(a=features_[c[0]], b=features_[c[1]])
                            )).cache()
        c_pears = cfeats.mapValues(lambda r: pearsonr(
                                            r.a.toArray(), 
                                            r.b.toArray())[0])\
                        .collectAsMap()
        c_cos = cfeats.mapValues(lambda r: cosine_similarity(
                                            [r.a.toArray()], 
                                            [r.b.toArray()]).item())\
                        .collectAsMap()
        c_jacc = cfeats.mapValues(lambda r: jaccard_score(
                                            r.a.toArray().astype(bool), 
                                            r.b.toArray().astype(bool)))\
                        .collectAsMap()
        c_metrics = pd.DataFrame([c_pears, c_cos, c_jacc]).T\
                        .rename(columns={0:'pears',1:'cos',2:'jacc'})\
                        .reset_index()
        features_ = features.rdd.map(lambda r: (r.biz_id, r.biz)).collectAsMap()
        c_metrics['b1'] = c_metrics['index'].apply(lambda x: features_[x[0]])
        c_metrics['b2'] = c_metrics['index'].apply(lambda x: features_[x[1]])
        return c_metrics.drop('index', axis=1)

    def assign_weights(self, wgts):
        """  Assign weights to `self.icf_weights` based on criteria
        """
        _metrics = ['pears', 'cos', 'jacc']
        if self.icf_metric['active'] == 'mean':
            self.icf_weights = wgts[['b1', 'b2']].copy()
            self.icf_weights['w'] = wgts[_metrics].mean(1)
        elif self.icf_metric['active'] in _metrics:
            self.icf_weights = wgts[['b1','b2', self.icf_metric['active']]]\
                                .rename(columns={self.icf_metric['active']:'w'})
        else:
            raise Exception("Not valid weighting metric!")
        # Filter values
        self.icf_weights = self.icf_weights[self.icf_weights.w >= self.icf_metric['min_value']]
        log("Model similar pairs:", self.icf_weights.w.count())
        
    def save(self, weights, biz_vectors, user_avg, biz_avg, user_map):
        """ Save Model values
        """
        # save weights
        weights.to_csv(self.cfg['mdl_file']+'.weights')
        self.assign_weights(weights)
        # save business vectors
        _fts_file = self.cfg['mdl_file']+'.features'
        def _serialize(v):
            return (v.size, v.indices.tolist(), v.values.tolist())
        def write_fts(x):
            with open(_fts_file, 'a') as f:
                f.write(x+'\n')
        biz_vectors.rdd.map(lambda r: json.dumps({
                                "biz_id": r.biz_id,
                                "biz": r.biz,
                                "features": _serialize(r.features)
                                }))\
                        .map(write_fts).count()
        self.biz_vectors = biz_vectors
        # save averages
        with open(self.cfg['mdl_file']+'.avgs', 'w') as prf:
                prf.write(json.dumps({
                    "business_avg": biz_avg,
                    "user_avg": user_avg,
                    "user_map": user_map
                }))
        self.biz_avg, self.user_avg = biz_avg, user_avg
        self.user_map = user_map

    def load_model(self):
        """ Load model values from config
        """
        self.assign_weights(
            pd.read_csv(self.cfg['mdl_file']+'.weights').drop('Unnamed: 0', axis=1)
        )
        self.biz_vectors = self._sql.createDataFrame(
                read_json(self._sc, self.cfg['mdl_file']+'.features')\
                    .map(lambda r: 
                        Row(biz_id=r['biz_id'], biz=r['biz'],
                            features=SparseVector(*r['features']))
                    )
            )
        # load avgs
        with open(self.cfg['mdl_file']+'.avgs', "r") as buff:
            mdl_ = json.loads(buff.read())
        self.biz_avg = mdl_['business_avg']
        self.user_avg  = mdl_['user_avg']
        self.user_map  = mdl_['user_map']

    def compute_avgs(self, data):
        """ Compute Business and User Avgs

             Params:
            ----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
            
            Returns:
            -----
            (dict, dict)
                (User Avgs, Business Avgs)
        """
        user_avg = data.map(lambda x: (x['user_id'], x['stars']))\
                        .groupByKey().mapValues(mean)\
                        .collectAsMap()
        buss_avg = data.map(lambda x: (x['business_id'], x['stars']))\
                        .groupByKey().mapValues(mean)\
                        .collectAsMap()
        log("Got Business and User rating averages")
        return user_avg, buss_avg

    def train(self, data):
        """ Training method

            Params:
            -----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
        """
        user_avg, biz_avg = self.compute_avgs(data)
        b_candidates, b_vectors, ratings, usr_map = self.prepare(data)
        b_weights = self.compute_weights(b_candidates, b_vectors)
        self.save(b_weights, b_vectors, user_avg, biz_avg, usr_map)

    def get_biz_nn(self, data):
        """ Get Business Nearest Neighbors

            Params:
            -----
            data: pyspark.rdd
                Test Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]

            Returns:
            -----
            pyspark.rdd
                Format: [
                    Row(biz=str,user=str, neighs={b: w, ...} ),
                    ...
                ]
        """
        _b_wgts = pd.concat([
            self.icf_weights.rename(columns={'b2':'b', 'b1':'k'}),
            self.icf_weights.rename(columns={'b1':'b', 'b2':'k'})
            ]).groupby('b')\
            .apply(lambda r: {j:w for j,w in zip(r['k'], r['w'])})\
            .to_dict()
        _k_n = self.k_n
        def _get_neighs(b):
            return sorted(_b_wgts.get(b, {}).items(), 
                        key=lambda i: i[1], 
                        reverse=True)[:_k_n]

        neighs = data.map(lambda x: Row(
                                biz=x['business_id'], 
                                user=x['user_id'])
                        )\
                    .map(lambda r: Row(
                            biz=r.biz,
                            user=r.user,
                            neighs=_get_neighs(r.biz)
                    ))
        return neighs

    def compute_score(self, test, data):
        """ Compute weighted average from neighbors

            Params:
            -----
            data: pyspark.rdd
                Format: [
                    Row(biz=str,user=str, neighs=[(idx, {'b':str, 'w':float}),..] ),
                    ...
                ]
            
            Returns:
            ----
            pyspark.rdd
                Format: [
                    Row(biz=str,user=str, score=float ),
                    ...
                ]
        """
        biz_test = set(test.map(lambda x: x['business_id']).distinct().collect())
        _inv_umap = {v:k for k,v in self.user_map.items()}
        # review if need to filter .rdd.map(lambda r: (r.biz, r.features)).filter(lambda r: r[0] in biz_test)\
        _usr_rates = self.biz_vectors\
                .rdd.map(lambda r: (r.biz, r.features))\
                .mapValues(lambda fts: {_inv_umap[ix]:iv \
                                for ix, iv in zip(fts.indices, fts.values)})\
                .flatMap(lambda e: [(_u, (e[0], _r)) for _u,_r in e[1].items()])\
                .groupByKey().mapValues(dict).collectAsMap()
        log("Computing scores...")
        # Join and compute score
        def _score(n, r):
            num_, den_, cnt = 0., 0., 0
            for n_i, w_i in n:
                if n_i in r:
                    num_ += w_i * r[n_i]
                    den_ += abs(w_i)
                    cnt += 1
            if cnt == 0:
                # Cold Start strategy
                return 2.5
            return num_ / den_
        
        pred_scores = data.map(lambda r: (
                                r.biz, 
                                r.user, 
                                _score(r.neighs, _usr_rates.get(r.user, {})))
                            ).collect()
        return pred_scores
    
    def predict(self, test, outfile):
        """ Prediction method

            test: pyspark.rdd
                Test Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
            outfile: str
                Path to output file
        """
        # Find K-neighbors from business
        neighs = self.get_biz_nn(test)
        # Compute prediction score
        preds_ = self.compute_score(test, neighs)
        with open(outfile, 'w') as of:
            for pv in preds_:
                of.write(json.dumps({
                    "user_id": pv[1], "business_id": pv[0],
                    "stars": pv[2]
                })+"\n")