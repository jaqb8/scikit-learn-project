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

clfs = dict()
clfs_names = ['euc_n_1', 'euc_n_5', 'euc_n_10', 'man_n_1', 'man_n_5', 'man_n_10']
neighbors = [1, 5, 10, 1, 5, 10]
no_of_features = range(1, 11)
clfs_keys = ['clf', 'features']
for idx, (n, clf_name) in enumerate(zip(neighbors, clfs_names)):
    clf = KNeighborsClassifier(n_neighbors=n) if idx < 3 else KNeighborsClassifier(n_neighbors=n, metric='manhattan')
    for feat_count in range(len(ranking)):
        clfs[f'{clf_name}_feat_{feat_count+1}'] = dict(clf=clf, features=[ranking[feature_index][0]
                                                                          for feature_index in range(feat_count + 1)])

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(clfs), n_splits * n_repeats))
print(scores)

for clf_id, clf_name in enumerate(clfs):
    selected_features = x[clfs[clf_name]['features']]
    for fold_id, (train, test) in enumerate(rskf.split(selected_features, y)):
        clf = clone(clfs[clf_name]['clf'])
        clf.fit(selected_features.iloc[train], y[train])
        y_pred = clf.predict(selected_features.iloc[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

print(scores)

mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print('{}: {} ({})'.format(clf_name, mean[clf_id], std[clf_id]))

np.save('results/results_matrix', scores)
