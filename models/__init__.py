from models.base_model import BaseModel
from models.content_based_model import ContentBasedModel
from models.item_cf_model import ItemBasedCFModel

models = {
    "BaseModel": BaseModel,
    "ContentBasedModel": ContentBasedModel,
    "ItemBasedCFModel": ItemBasedCFModel
}