import numpy as np
from scipy.stats import ttest_ind
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate


class Stats:

    def __init__(self, file_path):
        self.scores = np.load(file_path)
        self.clfs = {
            'kNN_euc_1': KNeighborsClassifier(n_neighbors=1),
            'kNN_euc_2': KNeighborsClassifier(n_neighbors=5),
            'kNN_euc_3': KNeighborsClassifier(n_neighbors=10),
            'kNN_man_1': KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
            'kNN_man_2': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
            'kNN_man_3': KNeighborsClassifier(n_neighbors=10, metric='manhattan')
        }
        self.headers = list(self.clfs.keys())
        self.names_column = np.array([[key] for key in self.clfs.keys()])
        self.alfa = .05
        self.t_statistic = np.zeros((len(self.clfs), len(self.clfs)))
        self.p_value = np.zeros((len(self.clfs), len(self.clfs)))
        for i in range(len(self.clfs)):
            for j in range(len(self.clfs)):
                self.t_statistic[i, j], self.p_value[i, j] = ttest_ind(self.scores[i],
                                                                       self.scores[j])

    def print_t_statistic(self):
        t_statistic_table = np.concatenate((self.names_column, self.t_statistic), axis=1)
        t_statistic_table = tabulate(t_statistic_table, self.headers, floatfmt=".2f")
        print("t-statistic:\n", t_statistic_table, "\n")

    def print_p_value(self):
        p_value_table = np.concatenate((self.names_column, self.p_value), axis=1)
        p_value_table = tabulate(p_value_table, self.headers, floatfmt=".2f")
        print("p-value:\n", p_value_table, "\n")

    def get_advantage(self):
        advantage = np.zeros((len(self.clfs), len(self.clfs)))
        advantage[self.t_statistic > 0] = 1
        return advantage

    def print_advantage_table(self):
        advantage = self.get_advantage()
        advantage_table = tabulate(np.concatenate(
            (self.names_column, advantage), axis=1), self.headers)
        print("Advantage:\n", advantage_table)

    def get_significance(self):
        significance = np.zeros((len(self.clfs), len(self.clfs)))
        significance[self.p_value <= self.alfa] = 1
        return significance

    def print_significance_table(self):
        significance = self.get_significance()
        significance_table = tabulate(np.concatenate(
            (self.names_column, significance), axis=1), self.headers)
        print("Statistical significance (alpha = 0.05):\n", significance_table)

    def print_stat_better(self):
        advantage = self.get_advantage()
        significance = self.get_significance()
        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (self.names_column, stat_better), axis=1), self.headers)
        print("Statistically significantly better:\n", stat_better_table)


for i in range(10):
    print('-' * 100)
    print(f'FILE: results_{i}\n\n')
    stats = Stats(f'results/results_{i}.npy')
    stats.print_t_statistic()
    stats.print_p_value()
    stats.print_advantage_table()
    stats.print_significance_table()
    stats.print_stat_better()
    print('-' * 100)
    print('\n\n')
