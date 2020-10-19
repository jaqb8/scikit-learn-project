from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# from load_data import load_data


def perform_ranking(data, classes, k=10):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(data, classes)
    new_features = [(feature, score)
                    for mask_value, feature, score in
                    zip(selector.get_support(), data.columns, selector.scores_)
                    if mask_value]
    return sorted(new_features, key=lambda x: x[1], reverse=True)


def print_ranking(ranking):
    for idx, rank in enumerate(ranking):
        print(f'{idx+1}. {rank[0]}: {rank[1]}')

# data, classes = load_data('./data_csv.csv')
# print(data.dtypes)
# ranking = perform_ranking(data, classes)
# print(ranking)
# print(len(ranking))
