from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing, svr
from hyperopt import tpe, hp
import math as m

X, y = load_boston().data, load_boston().target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
model = HyperoptEstimator(regressor=any_regressor('test1_reg'), preprocessing=
                          any_preprocessing('test1_preprocessing'), 
                          algo=tpe.suggest, verbose=True, max_evals=100)
model.fit(X_train, y_train, n_folds=3, cv_shuffle=True)
print(model.score(X_test, y_test))
print(model.best_model())
