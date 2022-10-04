from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# This file contains 2 functions for data preprocessing and 2 for evaluation
# Just to make Classification.ipynb file clear
def LabelEncoderforClassification(stream_quality_df, stream_quality_test):
    """
    Function get datasets and return modified datasets
    :param stream_quality_df: train set
    :param stream_quality_test: test set
    :return: both modified datasets
    """
    le = LabelEncoder()
    # labeling for train
    stream_quality_df.iloc[:, 8] = le.fit_transform(stream_quality_df.iloc[:, 8])
    stream_quality_df.iloc[:, 9] = le.fit_transform(stream_quality_df.iloc[:, 9])
    # labeling for test
    stream_quality_test.iloc[:, 8] = le.fit_transform(stream_quality_test.iloc[:, 8])
    stream_quality_test.iloc[:, 9] = le.fit_transform(stream_quality_test.iloc[:, 9])
    return stream_quality_df, stream_quality_test


def NumericalEncoderforClassification(stream_quality_df, stream_quality_test):
    """
    Function get datasets and return modified datasets
    :param stream_quality_df: train set
    :param stream_quality_test: test set
    :return: both modified datasets
    """
    # scaling for train
    scaler = MinMaxScaler()
    stream_quality_df.iloc[:, :8] = scaler.fit_transform(stream_quality_df.iloc[:, :8])
    stream_quality_df.iloc[:, 10] = scaler.fit_transform(np.array(stream_quality_df.iloc[:, 10]).reshape((-1, 1)))
    # scaling for test
    scaler = MinMaxScaler()
    stream_quality_test.iloc[:, :8] = scaler.fit_transform(stream_quality_test.iloc[:, :8])
    stream_quality_test.iloc[:, 10] = scaler.fit_transform(
        np.array(stream_quality_test.iloc[:, 10]).reshape((-1, 1)))
    return stream_quality_df, stream_quality_test


def LogisticRegressionFunction(X_train, Y_train, X_test, Y_test):
    logistic = LogisticRegression(penalty='l2').fit(X_train, Y_train)
    y_pred = logistic.predict(X_test)
    y_pred_train = logistic.predict(X_train)
    print('\nLogisticRegression')
    print('Accuracy on testset', accuracy_score(Y_test, y_pred))
    print('Precision on testset', precision_score(Y_test, y_pred))
    print('Recall on testset', recall_score(Y_test, y_pred))
    print('\nAccuracy on trainset', accuracy_score(Y_train, y_pred_train))
    print('Precision on trainset', precision_score(Y_train, y_pred_train))
    print('Recall on trainset', recall_score(Y_train, y_pred_train))


def RidgeClassifierFunction(X_train, Y_train, X_test, Y_test):
    ridge = RidgeClassifier().fit(X_train, Y_train)
    y_pred = ridge.predict(X_test)
    y_pred_train = ridge.predict(X_train)
    print('\nRidgeClassifier')
    print('Accuracy on testset', accuracy_score(Y_test, y_pred))
    print('Precision on testset', precision_score(Y_test, y_pred))
    print('Recall on testset', recall_score(Y_test, y_pred))
    print('\nAccuracy on trainset', accuracy_score(Y_train, y_pred_train))
    print('Precision on trainset', precision_score(Y_train, y_pred_train))
    print('Recall on trainset', recall_score(Y_train, y_pred_train))
