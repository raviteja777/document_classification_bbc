from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class ModelGenerator:

    def __init__(self, data_train, data_test=None):
        self.data_train = data_train
        self.data_test = data_test

    def naive_bayes(self):
        model = MNB()
        params = {}
        return self.cross_validation(model,params)

    def knn(self):
        model = KNN()
        params = {'n_neighbors': [3, 4, 5, 6], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        return self.cross_validation(model, params)

    def svm(self):
        model = SVC()
        params = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
        return self.cross_validation(model, params)

    def random_forest(self):
        model = RFC()
        params = {'n_estimators': [100, 200, 400],
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'sqrt', 'log2']}
        return self.cross_validation(model, params)

    def cross_validation(self, model, params):
        cols = self.data_train.columns.difference(['labels', 'sentences'])
        gcv = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', n_jobs=-1, cv=10)
        gcv.fit(self.data_train[cols], self.data_train['labels'])
        return gcv
