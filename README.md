# Yelp Recommender system 

```
Hybrid recommender implementation for a subset of the Yelp open dataset in California, US. 
```

The `yelp_recommender` project implements Content-based recommmendations, Item-based Colaborative Filtering, a Matrix-Factorization and a Hybrid method comparing the best out of the Test dataset RMSE obtained values. 

The **Item-based CF**, it uses MinHashLSH to find the possible businesses candidates that may be similar to later implement a Pearson correlation to determined the respective weights. The **Matrix-Factorization** utilized is an Alternated Least Squares (ALS) matrix method (from PySpark Mllib), a direct extension of this method is the **Hybrid recommender** that utilizes users and business averages, altogether with the ALS output to feed a Multilayer Perceptron trained to minimize the RMSE with the given ratings.

Finally, the model with the best perfomance in the given dataset is the modified version of the **Content-based** recommender. Using a continuous TF-IDF representation of 1000-dimensions that computes the similarity of the User and Business profiles using the cosine similarity, and uses a geometric weighted mean (Eq. 1) to predict the respective score/rating to the user-business pair.

$$ \hat{r_{u,b}} = \beta_{u,b} \bar{u} + (1-\beta_{u,b}) \bar{b}  $$

where,

$$\beta_{u,b} = Cos(V_u, V_b)$$

_Eq. 1. Geometric Weighted mean for Rating estimation using respective profile vectors ($V_u, V_b$)_

For implementation details: [Github Yelp recommender](https:/github.com/jorgeviz/yelp_recommender) 

## Execution

Prerequisites:

|        | Version |
| ------ | ------- |
| Python | >=3.6   |
| Spark | 2.4.5  |
| Hadoop | 2.7  |
| Java | 1.8  |
| Scala | 2.11  |


### Training

First, validate the configuration setup in your `config/config.py` and point to the correspondent training file within the selected JSON config. (Current training data format are EOF separated JSON strings with "user_id","business_id", "stars" and "text". )
The training script is runs as follows:

```bash
python train.py
```

### Predicting

For the prediction,  only "user_id","business_id" keys in the input JSON are required.

```bash
python predict.py <test_data> <output_file>
```

### Evaluation

```python
python scripts/evaluate.py <predicted_output_file> <ground_truth_file>
```



## Project Structure

```
├── README.md
├── config
│   ├── config.py
│   ├── config_base.json
│   ├── config_content.json
│   ├── config_content_extended.json
│   └── ...
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── content_based_model.py
│   ├── extended_content_model.py
│   └── item_cf_model.py
├── predict.py
├── scripts
│   ├── als_recommender.py
│   ├── evaluate.py
│   └── ...
├── train.py
└── utils
    ├── lsh.py
    ├── metrics.py
    ├── minhash.py
    ├── misc.py
    └── stopwords
```

Each model has 3 main methods `train`, `predict` and  `load_model`, and most inherit from the `BaseModel` class or ensure to handle within the class the configuration dict and PySpark Context.  

```python
# models/base_model.py

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
```



For model's addition to the pipeline, it has to be registered in the `__init__.py` file from the models directory, and define a configuration as the one in the `config`folder to point to the key class of the registered model, the trining data file, the model output file and its respective hyper parameters.

```json
{
    "class": "BaseModel",
    "training_data": "../../data/project/train_review.json",
    "mdl_file": "weights/base.model",
    "hp_params": {
        "TOP_TFIDF": 200,
        "RARE_WORDS_PERC": 0.0001
    }
}
```

### License 

[MIT License](./LICENSE)
Jorge Viz © 2020
