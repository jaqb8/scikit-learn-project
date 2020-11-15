from load_data import load_data
from ranking import perform_ranking
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np


x, y = load_data('./data_csv.csv')
ranking = perform_ranking(x, y)

clfs = {
    'kNN_euc_1': KNeighborsClassifier(n_neighbors=1),
    'kNN_euc_2': KNeighborsClassifier(n_neighbors=5),
    'kNN_euc_3': KNeighborsClassifier(n_neighbors=10),
    'kNN_man_1': KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
    'kNN_man_2': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'kNN_man_3': KNeighborsClassifier(n_neighbors=10, metric='manhattan')
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(clfs), n_splits * n_repeats))

for count in range(len(ranking)):
    features = [ranking[feature_index][0] for feature_index in range(count + 1)]
    print(features)
    selected_features = x[features]
    for fold_id, (train, test) in enumerate(rskf.split(selected_features, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(selected_features.iloc[train], y[train])
            y_pred = clf.predict(selected_features.iloc[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(clfs):
        print('{}: {} ({})'.format(clf_name, mean[clf_id], std[clf_id]))

    np.save(f'results/results_{count}', scores)
