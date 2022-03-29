import pandas as pd
from load_dataset import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load_dataset()
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Accuracy {acc}, precision {precision}")
    pd.DataFrame(y_pred).to_csv("predictions.csv")
