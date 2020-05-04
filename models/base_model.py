
class BaseModel(object):

    def __init__(self, sc, cfg):
        """ Base model constructor
        """
        self._sc = sc
        self.cfg = cfg

    def train(self, data):
        """ Training method

            Params:
            -----
            data: pyspark.rdd
                Input Data
        """
        pass
    
    def predict(self, data):
        """ Prediction method
        """
        pass
