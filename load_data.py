# from pandas import DataFrame, read_csv
import pandas as pd
import labels


def load_data(csv_file):
    data = pd.read_csv(csv_file, sep=';')
    data.columns = labels.FEATURES
    classes = data[data.columns[-1]]
    data.drop(data.columns[-1], axis=1, inplace=True),
    return (data, classes)

# x, y = load_data('./data_csv.csv')
# print(y)
# print(len(labels.FEATURES))
