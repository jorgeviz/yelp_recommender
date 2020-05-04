# -- Application vars
APP_NAME = "YelpRecomender"

# -- Model configurations
# model_conf = "config_base.json"
# model_conf = "config_content.json"
# model_conf = "config_item_cf.json"
model_conf = "config/config_base.json"

def validate_dirs(mcf):
    import os
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
