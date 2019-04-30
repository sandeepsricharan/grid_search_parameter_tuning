# Grid Search for Parameter Tuning
Grid-search is used to find the optimal hyperparameters of a model which results in the most ‘accurate’ predictions.

### Dataset
To dwell into the grid search we built a classification model on [*Breast Cancer dataset*](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)

The hyperparameters we tuned are:
* [Penalty](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c): l1 or l2 which species the norm used in the penalization.
* [C](https://stackoverflow.com/questions/22851316/what-is-the-inverse-of-regularization-strength-in-logistic-regression-how-shoul): Inverse of regularization strength- smaller values of C specify stronger regularization.

Also, in Grid-search function, we have the scoring parameter where we can specify the metric to evaluate the model on.

#### Reference
[Original blog](https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e) on Towards Data Science