from models.base_model import BaseModel
from models.content_based_model import ContentBasedModel
from models.item_cf_model import ItemBasedCFModel
from models.extended_content_model import ContentBasedExtendedModel

models = {
    "BaseModel": BaseModel,
    "ContentBasedModel": ContentBasedModel,
    "ItemBasedCFModel": ItemBasedCFModel,
    "ContentBasedExtendedModel": ContentBasedExtendedModel
}