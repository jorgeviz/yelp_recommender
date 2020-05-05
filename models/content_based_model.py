import re
import os
import json
from pprint import pprint
import math
from models.base_model import BaseModel
from utils.misc import log
from utils.metrics import cosine_similarity, mean
from collections import OrderedDict, Counter

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
        self.topk_tfidf = self.cfg['hp_params']['TOP_TFIDF']
        self.feat_type = self.cfg['hp_params']['FEATURES']
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

    def get_onehot_tfidf(self, data, doc_fq):
        """ Compute One-hot TF-IDF vectors
            
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
                top_k: {}
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
        # [TODO] ---- change the way most common terms are selected
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
        # [TODO] ---- change the way to construct pos index
        top_idx = {_ky:_i for _i,_ky in enumerate(top_terms)}
        log("Got Top Terms")
        return tfidf, top_terms, top_idx

    def featurize(self, data, ftype='onehot'):
        """ Generate TF-IDF features
        """
        if ftype == 'onehot':
            # format text data
            parsed = self.preprocess(data).cache()
            self.num_revs = parsed.count()
            # generate features
            _df = self._sc.broadcast(CBM.get_DF(parsed))
            log("Document term frequecy:")
            pprint(list(_df.value.items())[:10])
            return parsed, self.get_onehot_tfidf(parsed.mapValues(list), _df)
        return None, None, None, None

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

    def _get_profile(self, feats, top_idx, ftype):
        """ Build profile based on features

            Params:
            -----
            feats: pyspark.rdd
                [(k, {words}), ...]
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
        return None

    def build_profiles(self, data, top_idx, ftype):
        """ Build User and Item profiles

            Params:
            -----
            data: pyspark.rdd
                Parsed data [((b,u), [text]),  ...]
            top_idx: dict
                Position Index for top terms
            ftype: str
                Feature type
        """
        biz_revs = CBM.get_revs(data, 'biz')
        user_revs = CBM.get_revs(data, 'user')
        if ftype == 'onehot':
            biz_prof = self._get_profile(
                biz_revs.flatMapValues(lambda x: x)\
                    .flatMapValues(lambda x: x)\
                    .groupByKey()\
                    .mapValues(set),
                top_idx, ftype)
            user_prof = self._get_profile(
                user_revs.flatMapValues(lambda x: x)\
                    .flatMapValues(lambda x: x)\
                    .groupByKey()\
                    .mapValues(set),
                top_idx, ftype)
        else:
            return None, None
        return biz_prof, biz_prof

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
        biz_prof, user_prof = self.build_profiles(parsed, top_idx, self.feat_type)
        self.save(top_terms, top_idx, biz_prof, user_prof, biz_avg, user_avg)
        log(f"Model correctly saved at {self.cfg['mdl_file']}")

    def load_model(self):
        """ Load model from config defined model file
        """
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
        users = self.user_prof
        biz = self.biz_prof
        user_avg = self.user_avg
        biz_avg = self.biz_avg
        def _sim(u_i, b_i, u, b):
            if (u and b):
                return 5*cosine_similarity(u, b)
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