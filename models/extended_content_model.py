import os
import json
from utils.misc import log, debug, read_json
from pyspark.ml.linalg import SparseVector, Vectors
from models.content_based_model import ContentBasedModel

alcohol_values = ['full_bar', 'beer_and_wine', 'none']
noise_values = ['quiet', 'loud', 'average', 'very_loud']

def business_attributes_parser(d):
    if not isinstance(d, dict):
        return {}
    _attrs = {}
    if 'Alcohol' in d:
        _alc = eval(d['Alcohol'])
        if _alc is not None:
            _attrs['Alcohol'] = alcohol_values.index(_alc)
    if 'Ambiance' in d:
        a = eval(d['Ambiance'])
        if a:
            for k,v in a.items():
                _attrs['Ambiance_'+k] = int(v)+1
    if 'GoodForDancing' in d: 
        gfd = eval(d['GoodForDancing'])
        if gfd is not None:
            _attrs['GoodForDancing'] = int(gfd)+1
    if 'GoodForKids' in d: 
        gfk = eval(d['GoodForKids'])
        if gfk is not None:
            _attrs['GoodForKids'] = int(gfk)+1
    if 'GoodForMeal' in d:
        gfm = eval(d['GoodForMeal'])
        if gfm:
            for k,v in gfm.items():
                _attrs['GoodForMeal_'+k] = int(v)+1
    if 'OutdoorSeating' in d: 
        ouds = eval(d['OutdoorSeating'])
        if ouds is not None:
            _attrs['OutdoorSeating'] = int(ouds)+1
    if 'NoiseLevel' in d:
        nsl = eval(d['NoiseLevel'])
        if nsl is not None:
            _attrs['NoiseLevel'] = noise_values.index(nsl)
    if 'Music' in d:
        m = eval(d['Music'])
        if m:
            for k,v in m.items():
                _attrs['Music_'+k] = int(v)+1
    return _attrs


class ContentBasedExtendedModel(ContentBasedModel):
    """ Extended Content Based RS

        Model vars:
        -----
        self.top_terms - Top K terms of TFIDF with value
        self.top_idx  - TFIDF terms position index
        self.biz_prof -  Business Profile (TDIDF + Demographics)
        self.user_prof -  User Profile (TDIDF + Demographics)
        self.biz_avg - Business Average Rating
        self.user_avg  - User average Rating
    """

    def __init__(self, sc, cfg):
        """ Extended Content Based constructor
        """
        super().__init__(sc, cfg)
        self.__cfg = cfg
        # TFIDF cache validation
        self.tfidf_cache = cfg['hp_params']['CACHE_TFIDF']\
            if (os.path.exists(cfg['hp_params']['CACHE_TFIDF']+'.features') 
                and os.path.exists(cfg['hp_params']['CACHE_TFIDF']+'.profiles')  )\
            else ''
        # demographics attributes
        self.biz_attrs = self.cfg['hp_params']['BUSINESS_INFO']
        self.biz_attrs['rules'] = {
            "categories" :  lambda s: {_c:1 for _c in s.strip().split(', ')},
            "attributes" : business_attributes_parser
        }
        self.user_attrs = self.cfg['hp_params']['USERS_INFO']
        self.user_attrs['rules'] = [
            'funny',
            'useful',
            'compliment_cool',
            'compliment_cute',
            'compliment_funny',
            'compliment_hot',
            'compliment_list',
            'compliment_more',
            'compliment_note',
            'compliment_photos',
            'compliment_plain',
            'compliment_profile',
            'compliment_writer',
            'cool'
        ]

    def _get_demographics(self):
        """ Fetch demographics info from business and user

            Returns:
            -----
            user_feats, biz_embedd
        """
        # parse user
        user_info = read_json(self._sc, self.user_attrs['file'])
        u_rules = self.user_attrs['rules']
        user_feats = user_info.map(lambda x: (x['user_id'], [x.get(_r, 0) for _r in u_rules]))
        # parse biz
        biz_info = read_json(self._sc, self.biz_attrs['file'])
        b_rules = self.biz_attrs['rules']
        b_categs = self.biz_attrs['categs']
        b_ats = self.biz_attrs['attrs']
        tfidf_b_prof = self.tfidf_biz_prof
        _offs = len(self.biz_attrs['attrs']) + len(self.biz_attrs['categs'])
        biz_feats = biz_info.map(lambda x: (x['business_id'], 
                                            b_rules['categories'](x['categories']),
                                            b_rules['attributes'](x['attributes']),
                                            tfidf_b_prof.get(x['business_id'], [])
                                            ))
        def join_dict(m,q,n):
            w = n.copy()
            w.update(m)
            w.update(q)
            return w
        
        embedding_size = _offs + self.cfg['hp_params']['TOP_TFIDF']
        biz_embedd = biz_feats.map(lambda x: (
                                        x[0],
                                        {ix: x[1][cc] for ix, cc in enumerate(b_categs) if cc in x[1]},
                                        {jx+len(b_categs): x[2][aa] for jx, aa in enumerate(b_ats) if aa in x[2]},
                                        {kx+_offs : bb for kx, bb in enumerate(x[3]) if bb}
                                        )
                            ).map(lambda r: (r[0], join_dict(r[1], r[2], r[3])))\
                            .map(lambda r: (r[0], SparseVector(embedding_size, r[1])))
        # Embeddings [categories, attributs, tfidf]
        return user_feats, biz_embedd

    def _train(self, data):
        """ Extended training routine

            Params:
            ----
            data: pyspark.rdd
                Format: [(biz_id, usr_id), ...]
        """
        user_dems, biz_dems = self._get_demographics() # [TODO] NOT USED USER YET
        # get business profiles
        log("Computing business embeddings..")
        _biz_prof = biz_dems.collectAsMap()
        del self.tfidf_biz_prof
        _offs = len(self.biz_attrs['attrs']) + len(self.biz_attrs['categs'])
        embedding_size = _offs + self.cfg['hp_params']['TOP_TFIDF']
        # get user profiles
        def average_vects(vecs):
            _v = vecs[0].toArray()
            for rmg in vecs[1:]:
                _v += rmg.toArray()
            _v /= len(vecs)
            _idxs = _v.nonzero()[0].astype(int)
            return SparseVector(embedding_size, 
                                _idxs,
                                _v[_idxs]
                            )
        log("Computing user profiles from vector averages..")
        _user_prof = data.map(lambda x: (x[1], x[0])).groupByKey()\
                        .mapValues(lambda bs: [_biz_prof[b] for b in bs])\
                        .mapValues(average_vects).collectAsMap()
        del self.tfidf_user_prof
        self.biz_prof = _biz_prof
        self.user_prof = _user_prof
        self._save(self.top_terms, self.top_idx, self.biz_prof, self.user_prof, self.biz_avg, self.user_avg)
    
    def _save(self, top_terms, top_idx, biz_prof, user_prof, biz_avg, user_avg):
        """ Save Model values
        """
        def _serialize(v):
            return (v.size, v.indices.tolist(), v.values.tolist())
        _biz_prof = {k:_serialize(v) for k,v in biz_prof.items()}
        _usr_prof = {k:_serialize(v) for k,v in user_prof.items()}
        # Save profiles
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

    def train(self, data):
        """ Main Training method

            Params:
            ----
            data: pyspark.rdd
                Review JSON RDD 
                Format: [
                    {'business_id':x, 'user_id':x, 'text':x}, 
                    ...
                ]
        """
        if self.tfidf_cache:
            # load model from cache 
            self.cfg['mdl_file'] = self.tfidf_cache
            log("Using cache for TFIDF features...")
            self.load_model()
            parsed = data.map(lambda x: (
                            (x['business_id'], x['user_id']), "1")
                        ).cache()
            self.cfg['mdl_file'] = self.__cfg['mdl_file']
        else:
            # Train TFIDF module
            parsed = super().train(data)
        # Rename profiles TFIDF
        self.tfidf_biz_prof = self.biz_prof
        self.tfidf_user_prof = self.user_prof
        # Train extended routine
        self._train(parsed.map(lambda x: x[0]))

    def predict(self, test, outfile):
        """ Prediction method with extension for sparse computation
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
        self.feat_type = 'sparse'
        _is_sparse = False
        for k in self.biz_prof.keys():
            _is_sparse =  isinstance(k, SparseVector)
            break
        if not _is_sparse:
            self.biz_prof = {k: SparseVector(*v) for k,v in self.biz_prof.items()}
            self.user_prof = {k: SparseVector(*v) for k,v in self.user_prof.items()}
        super().predict(test, outfile)
        self.feat_type = self.cfg['hp_params']['FEATURES']