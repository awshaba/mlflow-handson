import xgboost as xgb
from load_dataset import load_dataset
from sklearn.metrics import accuracy_score, log_loss

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset()
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.3,
        "eval_metric": "mlogloss",
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "seed": 42,
    }
    
    model = xgb.train(params, dtrain, evals=[(dtrain, "train")])
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
        