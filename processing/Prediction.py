import pandas as pd
import numpy as np
from memory_profiler import profile
from line_profiler import LineProfiler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def replace_na(X_train, X_test, vars_cat):
    """_summary_
    Remplace les valeurs manquantes
    Args:
        X_train (_type_): données de test
        X_test (_type_): données d'entrainement
        vars_cat (_type_): variable catégorielle
    """
    X_train[vars_cat] = X_train[vars_cat].fillna('Missing')
    X_test[vars_cat] = X_test[vars_cat].fillna('Missing')

def find_frequent_labels(df, var, rare_perc):
    """
    Find frequent labels for categorical variables.

    Args:
    - df (pd.DataFrame): the dataset.
    - var (str): the categorical variable name.
    - rare_perc (float): threshold percentage for rare labels.

    Returns:
    - pd.Index: frequent labels.
    """
    df = df.copy()
    
    tmp = df.groupby(var)[var].count() / len(df)
    
    return tmp[tmp > rare_perc].index


def add_rare_status(X_train, X_test, vars_cat: list):
    for var in vars_cat:
        frequent_ls = find_frequent_labels(X_train, var, 0.05)
        X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
        X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')

def process_categorical_data(X_train, X_test, vars_cat: list):
    """
    Process categorical data by handling rare labels and encoding them.

    Args:
    - X_train (pd.DataFrame): the training dataset.
    - X_test (pd.DataFrame): the test dataset.
    - vars_cat (list): list of categorical variable names.

    Returns:
    - pd.DataFrame: processed datasets (X_train, X_test).
    """



    for var in vars_cat:
        X_train = pd.concat([X_train, pd.get_dummies(X_train[var], prefix=var, drop_first=True)], axis=1)
        X_test = pd.concat([X_test, pd.get_dummies(X_test[var], prefix=var, drop_first=True)], axis=1)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, variables: list) -> tuple:
    """
    Scale the features using StandardScaler.

    Args:
    - X_train (pd.DataFrame): the training dataset.
    - X_test (pd.DataFrame): the test dataset.
    - variables (list): list of feature names.

    Returns:
    - tuple: scaled datasets (X_train, X_test).
    """
    scaler = StandardScaler()
    scaler.fit(X_train[variables])

    X_train = scaler.transform(X_train[variables])
    X_test = scaler.transform(X_test[variables])

    return X_train, X_test, scaler


@profile
@LineProfiler
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Args:
    - X_train (pd.DataFrame): the training features.
    - y_train (pd.Series): the target variable.

    Returns:
    - LogisticRegression: trained logistic regression model.
    """
    model = LogisticRegression(C=0.0005, random_state=0)
    model.fit(X_train, y_train)
    return model

 

def evaluate_model(model: LogisticRegression, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the model using accuracy and roc-auc.

    Args:
    - model (LogisticRegression): trained model.
    - X_train (pd.DataFrame): training features.
    - y_train (pd.Series): training target variable.
    - X_test (pd.DataFrame): test features.
    - y_test (pd.Series): test target variable.
    """
    # Train set evaluation
    train_pred = model.predict_proba(X_train)[:, 1]
    train_class = model.predict(X_train)
    print(f'train roc-auc: {roc_auc_score(y_train, train_pred)}')
    print(f'train accuracy: {accuracy_score(y_train, train_class)}')

    # Test set evaluation
    test_pred = model.predict_proba(X_test)[:, 1]
    test_class = model.predict(X_test)
    print(f'test roc-auc: {roc_auc_score(y_test, test_pred)}')
    print(f'test accuracy: {accuracy_score(y_test, test_class)}')
    

