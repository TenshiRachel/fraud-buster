import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_train_data(test_size, random_state, feature_engineering=False):
    file_path = os.path.abspath('./data/Base.csv')
    df = pd.read_csv(file_path)
    df.replace(-1, np.nan)
    df = df.drop(['device_fraud_count'], axis=1, errors='ignore')

    epsilon = 1e-6

    if feature_engineering:
        # Longer current address duration = more stability
        df['address_stability_ratio'] = df['current_address_months_count'] / (df['prev_address_months_count'] + epsilon)
        # Flag for short stay at current address
        df['recent_address_change'] = (df['current_address_months_count'] < 6).astype(int)

        # Credit limit as percentage of income (riskier if high)
        df['credit_utilization_ratio'] = df['proposed_credit_limit'] / (df['income'] + epsilon)
        # High-risk flag
        df['high_risk_credit'] = (df['credit_risk_score'] < 0).astype(int)

        # Velocity of short vs long term transactions
        df['velocity_ratio_6h_24h'] = df['velocity_6h'] / (df['velocity_24h'] + epsilon)
        df['velocity_ratio_24h_4w'] = df['velocity_24h'] / (df['velocity_4w'] + epsilon)
        # High velocity
        df['high_velocity'] = (df['velocity_6h'] > 500).astype(int)

        # Detect bots
        df['short_session'] = (df['session_length_in_minutes'] < 2).astype(int)
        df['unusual_session_length'] = (((df['session_length_in_minutes'] < 2) | (df['session_length_in_minutes'] < 60))
                                        .astype(int))

    # Features (Contributing factors to fraud)
    X = df.drop(['fraud_bool'], axis=1)
    # Target (Fraud or not fraud)
    y = df['fraud_bool']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                        random_state=random_state)

    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # OneHot Encoding only for categorical columns
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    if categorical_cols:
        # Transform categorical columns
        ohe_train = pd.DataFrame(ohe.fit_transform(X_train[categorical_cols]))
        ohe_test = pd.DataFrame(ohe.transform(X_test[categorical_cols]))

        # Set the index of the transformed data to match the original data
        ohe_train.index = X_train.index
        ohe_test.index = X_test.index

        # Remove categorical columns from X_train and X_test
        X_train = X_train.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)

        # Concatenate one-hot-encoded features
        X_train = pd.concat([X_train, ohe_train], axis=1)
        X_test = pd.concat([X_test, ohe_test], axis=1)

    # Newer versions of sklearn require the column names to be strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    scaler = StandardScaler()
    # Computes statistics and transforms the data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transforms data with statistics from above
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to retain feature names
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test
