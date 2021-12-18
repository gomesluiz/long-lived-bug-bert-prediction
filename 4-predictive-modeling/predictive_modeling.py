
# Third-party packages.
import numpy as np
import torch
import xgboost as xgb

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import log_loss

def load_tensors_data_fn(tensors_path):
    """Load pytorch tensor data from file

    Args:
        tensors_path (str): a full filename path.

    Returns:
        X (array): a numpy array of features.
        y (array): a numpy array of labels.
    """
    tensors = torch.load(tensors_path)
    X = tensors[:, :-1].copy()
    y = tensors[:, -1].copy().astype(int)

    return (X, y)


def optimize_model_fn(X_train, X_valid, y_train, y_valid, max_evals=10):
    """Optimize the predicting model.

    Args:
        trials:
        max_evals:
    
    Returns:
        best_hyperparameters:

    """

    def score_model_fn(params):
        """Score the predicting model with hyperparameters.

        Args:
            X_train, y_train:
            X_valid, y_valid:
            params:

        Returns:
            loss:
            status:
        """
        
        num_round = int(params['n_estimators'])
        del params['n_estimators']
    
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        model  = xgb.train(params, dtrain, num_round)
        
        predictions = model.predict(dvalid).reshape((X_valid.shape[0], 5))
        
        score = log_loss(y_valid, predictions)

        return {'loss': score, 'status': STATUS_OK}

    params_space = {'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
            'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
            'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
            'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'num_class' : 5,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'nthread' : 6,
            'verbosity' : 1}

    best_params = fmin(score_model_fn, 
                params_space, 
                algo=tpe.suggest, 
                trials=Trials(), 
                max_evals=max_evals)

    return best_params