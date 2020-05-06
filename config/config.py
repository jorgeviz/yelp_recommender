import os

os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

# -- Application vars
APP_NAME = "YelpRecomender"

# -- Model configurations
# model_conf = "config/config_base.json"
# model_conf = "config/config_content.json"
# model_conf = "config/config_content_sparse.json"
model_conf = "config/config_item_cf.json"


def validate_dirs(mcf):
    from pathlib import Path
    mdl_f = Path(os.path.abspath(mcf))
    if not os.path.exists(mdl_f.parent):
        os.makedirs(mdl_f.parent)

def load_conf():
    import json
    with open(model_conf, 'r') as mf:
        mdl_cnf = json.load(mf)
    assert 'class' in mdl_cnf, "No Model Class!"
    assert 'mdl_file' in mdl_cnf, "No Model output file!"
    assert 'hp_params' in mdl_cnf, "No Hyperparameters!"
    assert 'training_data' in mdl_cnf, "No Training data!"
    validate_dirs(mdl_cnf['mdl_file'])
    return mdl_cnf
