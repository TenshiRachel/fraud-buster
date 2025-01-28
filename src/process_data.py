import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_train_data(test_size, random_state):
    file_path = os.path.abspath('../../data/Base.csv')
    df = pd.read_csv(file_path)
    df = df.drop(['device_fraud_count'], axis=1, errors='ignore')

    # Features (Contributing factors to fraud)
    X = df.drop(['fraud_bool'], axis=1)
    # Target (Fraud or not fraud)
    y = df['fraud_bool']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                        random_state=random_state)

    s = (X_train.dtypes == 'object')  # list of column-names and whether they contain categorical features
    object_cols = list(s[s].index)  # All the columns containing these features

    # Encode categorical columns
    ohe = OneHotEncoder(sparse_output=False,
                        handle_unknown='ignore')

    # Get one-hot-encoded columns
    ohe_cols_train = pd.DataFrame(ohe.fit_transform(X_train[object_cols]))
    ohe_cols_test = pd.DataFrame(ohe.transform(X_test[object_cols]))

    # Set the index of the transformed data to match the original data
    ohe_cols_train.index = X_train.index
    ohe_cols_test.index = X_test.index

    # Remove the object columns from the training and test data
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_test = X_test.drop(object_cols, axis=1)

    # Concatenate the numerical data with the transformed categorical data
    X_train = pd.concat([num_X_train, ohe_cols_train], axis=1)
    X_test = pd.concat([num_X_test, ohe_cols_test], axis=1)

    # Newer versions of sklearn require the column names to be strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    scaler = StandardScaler()
    # Computes statistics and transforms the data
    X_train = scaler.fit_transform(X_train)
    # Transforms data with statistics from above
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
