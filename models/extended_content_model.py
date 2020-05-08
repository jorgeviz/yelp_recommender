from models.content_based_model import ContentBasedModel

class ContentBasedExtendedModel(ContentBasedModel):
    def __init__(self, sc, cfg):
        """ Extended Content Based constructor
        """
        super()__init__(sc, cfg)
        self.__cfg = cfg