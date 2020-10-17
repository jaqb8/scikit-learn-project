from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from load_data import load_data


def perform_ranking(data, classes):
    selector = SelectKBest(score_func=chi2)
    selector.fit(data, classes)
    res = selector.scores_
    output = dict(zip(data.columns, res))
    return sorted(output.items(), key=lambda x: x[1], reverse=True)


data, classes = load_data('./data_csv.csv')
# print(data.dtypes)
ranking = perform_ranking(data, classes)
print(ranking)
print(len(ranking))
