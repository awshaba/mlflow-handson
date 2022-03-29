from catboost import CatBoostClassifier
from load_dataset import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset()
    params = {
    "iterations": 2,
    "learning_rate": 1,
    "depth": 2,
    "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Accuracy {acc}")
