import re
import os
import json
from pprint import pprint
import math
from collections import OrderedDict, Counter, namedtuple

from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import SparseVector
# from pyspark.sql import SQLContext, Row
import scipy.sparse as sps
import numpy as np

from models.base_model import BaseModel
from utils.misc import log, debug, read_json
from utils.metrics import mean, cosine_similarity


puncts = [ "(", "[", ",", ".", "!", "?", ":", ";", 
        "]", ")" ,"\n", "*", "/", " ", "$", "'", 
        '"', '-', '\r', '#']
puncts_re = r"(\(|\[|\,|\.|\!|\?|\:|\;|\]|\)|\n|\*|\/|\$|\'|\"|-|\#|\r)"


class ContentBasedModel(BaseModel):
    """ Content Based recommendation model 
        based on TF-IDF features
    """

    def __init__(self, sc, cfg):
        """ Content Based constructor
        """
        super().__init__(sc,cfg)
        # self._spark = SQLContext(self._sc)
        self.topk_tfidf = self.cfg['hp_params']['TOP_TFIDF']
        self.feat_type = self.cfg['hp_params']['FEATURES']
        # params for linear decision rule
        self.alpha = self.cfg['hp_params']['DECISION_RULE']['params']['slope']
        self.beta = self.cfg['hp_params']['DECISION_RULE']['params']['bias']
        # aux elements
        with open('utils/stopwords') as f:
            self.stops = f.read().split('\n')
        # model vars
        self.num_revs = 0

    def preprocess(self, rdd):
        """ Preprocess text reviews. Parse text, 
            removing stop words, punctuation and empy spaces

            Params:
            -----
            rdd: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
            Returns:
            -----
            pyspark.rdd
                Format: [
                    ((biz_id, usr_id), [texts])
                ]
        """
        def extend_puncts(l):
            return re.sub(
                puncts_re, 
                lambda x: " "+x.group(0)+" ", 
                l
            )
        remove_set = set(puncts+self.stops+[''])
        parsed_data = rdd.map(lambda x: (
                            (x['business_id'], x['user_id']), 
                            x['text'])
                        ).map(lambda x: (
                            x[0], 
                            extend_puncts(x[1].lower()).split(' '))
                        ).zipWithIndex()\
                        .map(lambda z: ((z[1], z[0][0]), z[0][1]))\
                        .flatMapValues(lambda x: x)\
                        .filter(lambda x: x[1] not in remove_set)\
                        .groupByKey()\
                        .map(lambda x: (x[0][1], x[1]))
        return parsed_data

    @staticmethod
    def get_revs(data, prof):
        """ Join all reviews by business or user to build profile.
            
            Params:
            -----
            data: pyspark.rdd
                Format: [((biz, usr), <texts>), ...]

            Return:
            ----
            pyspark.rdd
                Format: [
                    (b, [[texts],..])
                ]
        """
        _j = 0 if prof == 'biz' else 1
        _revs = data.map(lambda x: (x[0][_j], x[1]))\
                        .mapValues(list)\
                        .groupByKey()
        return _revs

    @staticmethod
    def get_DF(data):
        """ Count Terms' Document Frequency

            Params:
            ----
            data: pyspark.rdd
                Format: [((b,u), [texts]),...]

            Returns:
            -----
            pyspark.rdd
                Format: [(word, freq), ...]
        """
        mapped = data.flatMapValues(lambda x: set(x))\
                    .map(lambda x: (x[1], x[0]))\
                    .groupByKey()
        # Document frequency
        doc_freq = mapped.mapValues(len)
        return doc_freq.collectAsMap()

    def get_tfidf(self, data, doc_fq):
        """ Compute TF-IDF vectors
            
            Params:
            -----
            data: pyspark.rdd
                Format: [(k, [texts]), ...]
            doc_fq: pypsark.broadcast
                Dict with term's doc frequency

            Returns:
            -----
            (tfidf, top_k, top_idx): (pyspark.rdd, OrderedDict, dict)
                tfidf: [(k, {w:(tf,df, tf-idf),..}), ...]
                top_k: {k: tf-idf, k2:tf-idf, ...}
        """ 
        N = self.num_revs
        # TF  (biz_id, {t1: 3, t2: 4})
        def normed_tf(x, norm=False):
            k, v  = x
            c = Counter(v)
            # for _k in v:
            #     c.update(_k)
            if norm:
                _max = c.most_common()[0][1]
                return (k, {i: j/_max for i,j in c.items()})
            return (k, {i: j for i,j in c.items()})
        tfq = data.mapValues(list).map(normed_tf)
        # IDF (biz_id, {t1: (2,3,4.5), t2: (1,3,6.7)}
        def _tfidf(val):
            k, _tf = val
            d = {}
            for term,v in _tf.items():
                _df = doc_fq.value[term]
                d[term] = (v, _df, v * math.log(N/_df, 2))
            return (k, d)
        tfidf = tfq.map(_tfidf)
        # Get most recent elements, top terms {t1: 23.4, t2:6.4} 
        top_terms = tfidf.flatMap(lambda x: [(_k, _j)  for _k, _j in x[1].items()])\
                        .filter(lambda x: x[1][1] > 1)\
                        .map(lambda x: (x[0], x[1][2]))\
                        .groupByKey()\
                        .mapValues(max)
        top_terms = OrderedDict(
            top_terms
                .sortBy(lambda x: x[1], ascending=False)\
                .take(self.topk_tfidf)
        )
        # - pos index
        top_idx = {_ky:_i for _i,_ky in enumerate(top_terms)}
        log("Got Top Terms")
        return tfidf, top_terms, top_idx

    def featurize(self, data, ftype='onehot'):
        """ Generate TF-IDF features
        """
        # format text data
        parsed = self.preprocess(data).cache()
        self.num_revs = parsed.count()
        if ftype in ('onehot', 'continuous'):
            # generate features
            _df = self._sc.broadcast(CBM.get_DF(parsed))
            log("Document term frequecy:")
            pprint(list(_df.value.items())[:10])
            return parsed, self.get_tfidf(parsed.mapValues(list), _df)
        elif ftype == 'sparse':
            tf_hasher = HashingTF(self.topk_tfidf)
            # All document hashes
            _tf = tf_hasher.transform(parsed\
                                   .mapValues(list)\
                                   .map(lambda x: x[1])
                            )
            idfer = IDF(minDocFreq=self.cfg['hp_params']['MIN_DOC_FREQ'])\
                        .fit(_tf)
            TFIDF = namedtuple("TFIDF", ('tfer','idfer'))
            tfidf = TFIDF(tf_hasher, idfer)
            log("Constructed Sparse TFIDF!")
            return parsed, (tfidf, None, None)
        return None, (None, None, None)

    def get_onehot_profile(self, feats, top_idx):
        """ One-hot profile construction

            Params:
            -----
            feats: pyspark.rdd
                [(k, {words}), ...]
            top_idx: dict
                Position Index for top terms
            
            Returns:
            -----
            pyspark.rdd
                [(k, [0,1,0,0,1]), ..]
        """
        _TOP_TFIDF = self.topk_tfidf
        # One-hot encode
        def one_hot_tdf(x):
            one_hot = [0]*_TOP_TFIDF
            for w in x[1]:
                if w in top_idx:
                    one_hot[top_idx[w]] = 1
            return (x[0], one_hot)
        return  feats.map(one_hot_tdf)
    
    def get_continuous_profile(self, feats, top_terms, top_idx):
        """ Continuous profile construction

            Params:
            -----
            feats: pyspark.rdd
                [(k, [words]), ...]
            top_terms: dict
                Fast access TFIDF values of top-k
            top_idx: dict
                Position Index for top terms
            
            Returns:
            -----
            pyspark.rdd
                [(k, [0,2.1,0,4.0,0.1]), ..]
        """
        _TOP_TFIDF = self.topk_tfidf
        def _encode(x):
            vect = [0]*_TOP_TFIDF
            for w in x[1]:
                if w in top_idx:
                    vect[top_idx[w]] = top_terms[w]
            return (x[0], vect)
        return feats.map(_encode)
    
    def get_sparse_profile(self, feats, tfidf):
        """ Sparse profile construction

            Params:
            -----
            feats: pyspark.rdd
                [(k, [words]), ...]
            tfidf: TFIDF (HashingTF, IDF)

            Returns:
            -----
            pyspark.rdd
                #### --- TODO
        """
        data = feats.zipWithIndex()\
                    .map(lambda x: (x[1], x[0])).cache()
        _tf = tfidf.tfer.transform(
            data.map(lambda x: list(x[1][1]))
        )
        _tfidf = tfidf.idfer.transform(_tf)
        embedding = data.map(lambda x: (x[0], x[1][0]))\
                        .join(
                            _tfidf.zipWithIndex()\
                                .map(lambda x: (x[1], x[0]))
                        )
        return embedding.map(lambda x: x[1])

    def _get_profile(self, feats, tfidf, top_terms, top_idx, ftype):
        """ Build profile based on features

            Params:
            -----
            feats: pyspark.rdd
                [(k, {words}), ...]
            tfidf: pyspark.rdd | TFIDF
                TF-IDF [((b,u), {w:(tf,df, tfidf), ...}), ...]
            top_terms: dict
                Fast access TFIDF values of top-k
            top_idx: dict
                Position Index for top terms
            ftype: str
                Feature type

            Returns:
            -----
            pyspark.rdd
                Vector representation: [(k, [3,2.5,6,7,..]), ..]
        """
        if ftype == 'onehot':
            return self.get_onehot_profile(feats, top_idx)
        elif ftype == 'continuous':
            return self.get_continuous_profile(feats, top_terms, top_idx)
        elif ftype == 'sparse':
            return self.get_sparse_profile(feats, tfidf)
        return None

    def build_profiles(self, data, tfidf, top_terms, top_idx, ftype):
        """ Build User and Item profiles

            Params:
            -----
            data: pyspark.rdd
                Parsed data [((b,u), [text]),  ...]
            tfidf: pyspark.rdd | TFIDF
                TF-IDF [((b,u), {w:(tf,df, tfidf), ...}), ...]
            top_terms: dict
                Fast access TFIDF values of top-k
            top_idx: dict
                Position Index for top terms
            ftype: str
                Feature type
        """
        biz_revs = CBM.get_revs(data, 'biz')\
                    .flatMapValues(lambda x: x)\
                    .flatMapValues(lambda x: x)\
                    .groupByKey()
        user_revs = CBM.get_revs(data, 'user')\
                    .flatMapValues(lambda x: x)\
                    .flatMapValues(lambda x: x)\
                    .groupByKey()
        if ftype == 'onehot':
            biz_prof = self._get_profile(biz_revs.mapValues(set),
                                tfidf, top_terms, top_idx, ftype)
            user_prof = self._get_profile(user_revs.mapValues(set),
                            tfidf, top_terms, top_idx, ftype)
        elif ftype in ('continuous', 'sparse'):
            biz_prof = self._get_profile(biz_revs, tfidf,
                                top_terms, top_idx, ftype)
            user_prof = self._get_profile(user_revs, tfidf,
                                top_terms, top_idx, ftype)
        else:
            return None, None
        return biz_prof, user_prof

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

    def save(self, top_terms, top_idx, biz_prof, user_prof, biz_avg, user_avg):
        """ Save Model values
        """
        if self.feat_type in ('continuous', 'onehot'):
            # Save profiles
            _biz_prof = biz_prof.collect()
            _usr_prof = user_prof.collect()
            with open(self.cfg['mdl_file']+'.profiles', 'w') as prf:
                prf.write(json.dumps({
                    "business_profiles": _biz_prof,
                    "user_profiles": _usr_prof,
                    "business_avg": biz_avg,
                    "user_avg": user_avg
                }))
            # Save content 
            with open(self.cfg['mdl_file']+'.features', 'w') as prf:
                prf.write(json.dumps({
                    "top_terms": top_terms,
                    "terms_pos_idx": top_idx 
                }))
            # In memory value assignation
            self.top_terms, self.top_idx = top_terms, top_idx
            self.biz_prof, self.user_prof = dict(_biz_prof), dict(_usr_prof)
            self.biz_avg, self.user_avg = biz_avg, user_avg
        elif self.feat_type == 'sparse':
            # save profiles
            def write_profile(x, pfile):
                with open(pfile, 'a') as bf:
                    bf.write(json.dumps({x[0]: 
                        (x[1].size, 
                        x[1].indices.tolist(), 
                        x[1].values.tolist())
                        })+"\n"
                    )
                return 1
            bfile = self.cfg['mdl_file']+'.biz_profile'
            biz_prof.map(lambda x: write_profile(x, bfile)).count()
            ufile = self.cfg['mdl_file']+'.user_profile'
            user_prof.map(lambda x: write_profile(x, ufile)).count()
            # save avgs
            with open(self.cfg['mdl_file']+'.avgs', 'w') as prf:
                prf.write(json.dumps({
                    "business_avg": biz_avg,
                    "user_avg": user_avg
                }))
            self.top_terms, self.top_idx = None, None
            self.biz_prof, self.user_prof = None, None
            self.biz_avg, self.user_avg = biz_avg, user_avg

    def train(self, data):
        """ Training method

            Params:
            ----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
        """
        user_avg, biz_avg = self.compute_avgs(data)
        parsed, (tfidf, top_terms, top_idx) = self.featurize(data, self.feat_type)
        biz_prof, user_prof = self.build_profiles(parsed, tfidf, top_terms, top_idx, self.feat_type)
        self.save(top_terms, top_idx, biz_prof, user_prof, biz_avg, user_avg)
        log(f"Model correctly saved at {self.cfg['mdl_file']}")
        return parsed

    def load_model(self):
        """ Load model from config defined model file
        """
        if self.feat_type in ('onehot', 'continuous'):
            # load profiles
            with open(self.cfg['mdl_file']+'.profiles', "r") as buff:
                mdl_ = json.loads(buff.read())
            self.biz_prof = dict(mdl_['business_profiles'])
            self.user_prof = dict(mdl_['user_profiles'])
            self.biz_avg = mdl_['business_avg']
            self.user_avg  = mdl_['user_avg']
            # load features
            with open(self.cfg['mdl_file']+'.features', "r") as buff:
                mdl_f = json.loads(buff.read())
            self.top_terms, self.top_idx  = mdl_f['top_terms'], mdl_f['terms_pos_idx']
        elif self.feat_type == 'sparse':
            mdl_b = read_json(self._sc, self.cfg['mdl_file']+'.biz_profile')
            self.biz_prof = mdl_b.map(lambda x: list(x.items())[0])\
                                .mapValues(lambda sv: SparseVector(*sv)).collectAsMap()
            mdl_b = read_json(self._sc, self.cfg['mdl_file']+'.user_profile')
            self.user_prof = mdl_b.map(lambda x: list(x.items())[0])\
                                .mapValues(lambda sv: SparseVector(*sv)).collectAsMap()
            # load avgs
            with open(self.cfg['mdl_file']+'.avgs', "r") as buff:
                mdl_ = json.loads(buff.read())
            self.biz_avg = mdl_['business_avg']
            self.user_avg  = mdl_['user_avg']
        else:
            log("Not valid feature type!", lvl="WARNING")
            return 
        log(f"Model correctly loaded from {self.cfg['mdl_file']}")

    def cold_start(self, test, users, biz):
        """ [NOT USED FOR NOW] Cold Start strategy
        """
        # Average of the rest of the population
        missing_biz = (set(test.map(lambda x: x['business_id']).distinct().collect()) 
                - set(biz))
        missing_usr = (set(test.map(lambda x: x['user_id']).distinct().collect()) 
                - set(users))
        return missing_biz, missing_usr

    def predict(self, test, outfile):
        """ Prediction method

            Params:
            ----
            test: pyspark.rdd
                Test Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
            outfile: str
                Path to output file
        """
        _feat_type = self.feat_type
        users, biz = self.user_prof, self.biz_prof
        user_avg, biz_avg = self.user_avg, self.biz_avg
        _alpha, _beta = self.alpha, self.beta
        _decision = self.cfg['hp_params']['DECISION_RULE']['active']
        def _sim(u_i, b_i, u, b):
            if (u and b):
                if _feat_type == 'sparse':
                    _cos = cosine_similarity(
                        [u.toArray()],[b.toArray()]
                    ).item()
                else:
                    _cos = cosine_similarity([u],[b]).item()
                if _decision == 'linear':
                    return user_avg[u_i] + _alpha*(_cos - _beta)
                elif _decision == 'geometric':
                    return (_cos)*user_avg[u_i]  + (1-_cos)*biz_avg[b_i]
                else:  # constant
                    return 5*(_cos)
            # similarity for cold start, average of the other
            if u:
                # No business info, return avg from user
                return user_avg[u_i]
            elif b:
                # No user info, return avg from business
                return biz_avg[b_i]
            return 2.5 # return constant
        preds_ = test.map(lambda x: (x['user_id'], x['business_id']))\
                    .map(lambda x: (x[0], x[1], users.get(x[0], []), biz.get(x[1], [])) )\
                    .map(lambda x: (x[0], x[1], _sim(*x)) ).collect()
        with open(outfile, 'w') as of:
            for pv in preds_:
                of.write(json.dumps({
                    "user_id": pv[0], "business_id": pv[1],
                    "stars": pv[2]
                })+"\n")


CBM = ContentBasedModel