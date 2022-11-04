'''
Data Mining Course Project
Imported models are:
Decision Tree, Random Forest, Naive Bayes, KNN, Perceptron, and Logistic Regression
The model ensemble comprises the following models:
Random Forest, Naive Bayes, KNN, and Logistic Regression
'''

#import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter(action='ignore', category=ConvergenceWarning)

def _replace_str(df):
    '''
    replaces the values of a column in a dataframe whose dtype is string (object)
    it is a helper function and will only be used in preprocess()
    '''

    for c in df.columns:
        if df[c].dtype == object:
            unq = sorted(list(df[c].unique()))
            mapping = dict()
            for u in unq:
                mapping.update({u: unq.index(u)})
            df[c].replace(mapping, inplace=True)

def preprocess(df):
    '''
    applies the necessary preprocessings to the Customer_Segmentation dataframe
    '''

    df.drop('ID', axis=1, inplace=True)
    df['Work_Experience'].fillna(value=np.mean(df['Work_Experience']), inplace=True)
    df['Family_Size'].fillna(value=np.mean(df['Family_Size']), inplace=True)
    df['Ever_Married'].fillna(df['Ever_Married'].value_counts().index[0], inplace=True)
    df['Graduated'].fillna(df['Graduated'].value_counts().index[0], inplace=True)
    df.dropna(inplace=True)
    _replace_str(df)

def _is_categorical(col):
    '''
    checks whether a column is categorical (not numerical)
    True: categorical, False: otherwise
    '''

    if set(col) == set(range(col.unique().shape[0])):
        return True
    if col.dtype in (object, bool):
        return True
    return False

def min_max_normalization(df):
    '''
    applies min-max normalization to the columns of a dataframe that are categorical
    '''

    for c in df.columns:
        if not _is_categorical(df[c]): 
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

def split(df):
    '''
    splits the dataframe into X and y
    '''

    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]
    min_max_normalization(X)
    return X, y

def majority_voting(*args):
    '''
    takes the majority voting of predictions made by different models
    implemented from scratch
    currently out-of-use
    '''

    predictions = np.array([], dtype=np.uint8)
    for i in range(len(args[0])):
        vote_counts = [0, 0, 0, 0]
        for j in range(len(args)):
            vote_counts[args[j][i]] += 1
        predictions = np.append(predictions, vote_counts.index(max(vote_counts)))
    return predictions


def main():
    train = pd.read_csv('Train.csv')
    test = pd.merge(pd.read_csv('Test.csv'), pd.read_csv('sample_submission.csv'), on='ID')
    preprocess(train)
    preprocess(test)
    X_train, y_train = split(train)
    X_test, y_test = split(test)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    pr = Perceptron()
    pr.fit(X_train, y_train)
    pr_pred = pr.predict(X_test)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)

    v = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('gnb', gnb), ('knn', knn)], voting='hard')
    v.fit(X_train, y_train)
    v_pred = v.predict(X_test)
    
    print("RandomForest Accuracy: %.2f%%" %(accuracy_score(y_test, rf_pred)*100))
    print("DecisionTree Accuracy: %.2f%%" %(accuracy_score(y_test, dt_pred)*100))
    print("KNN Accuracy: %.2f%%" %(accuracy_score(y_test, knn_pred)*100))
    print("NaiveBayes Accuracy: %.2f%%" %(accuracy_score(y_test, gnb_pred)*100))
    print("Perceptron Accuracy: %.2f%%" %(accuracy_score(y_test, pr_pred)*100))
    print("LogisticRegression Accuracy: %.2f%%" %(accuracy_score(y_test, lr_pred)*100))
    print("---------After Voting---------")
    print("Overall Accuracy: %.2f%%" %(accuracy_score(y_test, v_pred)*100))

if __name__ == '__main__':
    main()