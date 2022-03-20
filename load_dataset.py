import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_dataset(test_size=0.2, random_state=123, download_path='.'):
    # Load dataset
    iris = datasets.load_iris()
    # Process dataset
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset()
    print(x_train)
    print(y_train)
