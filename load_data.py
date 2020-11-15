# from pandas import DataFrame, read_csv
import pandas as pd
import labels
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np


def load_data(csv_file):
    data = pd.read_csv(csv_file, sep=';')
    data.columns = labels.FEATURES
    classes = data[data.columns[-1]]
    data.drop(data.columns[-1], axis=1, inplace=True),
    return (data, classes)
