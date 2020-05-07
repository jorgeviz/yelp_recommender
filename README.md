# Yelp Recommender system 

## --  Jorge Vizcayno García  (INF553)

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

For implementation details: [Github Yelp recomender](https:/github.com/jorgeviz/yelp_recommender) 
