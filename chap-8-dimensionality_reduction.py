from sklearn import datasets
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import time

mnist = datasets.fetch_openml("mnist_784", version=1)

class chapter():
    def method(self):
        features = mnist['data']
        target = mnist['target']

        print(features.shape)

        pca = PCA()
        pca.fit(features)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1

        pca = PCA(n_components=154)
        pca.fit_transform(features)

        print(features.shape)

        clf = Pipeline([
            ("Kernal PCA", KernelPCA(n_components=d)),
            ("log_reg", LogisticRegression())
        ])

        param_grid = [{
            'kpca_gamma':np.linspace(0.03, 0.05, 10),
            'kpca_kernel': ['rbf', 'linear', 'sigmoid']
        }]

        grid_search = GridSearchCV(clf, param_grid, cv=3)
        grid_search.fit(features)

class exer():
    X = mnist["data"]
    y = mnist["target"]

    def rnd_clf(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=10000)

        rnd_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        start_time = time.time()
        rnd_forest_clf.fit(x_train, y_train)
        end_time = time.time()

        y_pred = rnd_forest_clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)

        print("Training took {} time".format(end_time-start_time))
        print("Accuracy score for plain Random classifier {}".format(acc_score))

    def pca_rnd_clf(self):

        pca = PCA()
        pca.fit(self.X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        best_n_components = np.argmax(cumsum >= 0.95) + 1

        pca = PCA(n_components=best_n_components)
        self.X = pca.fit_transform(self.X)

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=10000)

        rnd_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        start_time = time.time()
        rnd_forest_clf.fit(x_train, y_train)
        end_time = time.time()

        y_pred = rnd_forest_clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)

        print("Training took {} time".format(end_time - start_time))
        print("Accuracy score for PCA Random classifier {}".format(acc_score))

    def log_reg_clf(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=10000)

        log_reg_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)

        start_time = time.time()
        log_reg_clf.fit(x_train, y_train)
        end_time = time.time()

        y_pred = log_reg_clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)

        print("Training took {} time".format(end_time-start_time))
        print("Accuracy score for plain Random classifier {}".format(acc_score))

    def pca_log_reg_clf(self):

        pca = PCA()
        pca.fit(self.X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        best_n_components = np.argmax(cumsum >= 0.95) + 1

        pca = PCA(n_components=best_n_components)
        self.X = pca.fit_transform(self.X)

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=10000)

        log_reg_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)

        start_time = time.time()
        log_reg_clf.fit(x_train, y_train)
        end_time = time.time()

        y_pred = log_reg_clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)

        print("Training took {} time".format(end_time - start_time))
        print("Accuracy score for PCA Random classifier {}".format(acc_score))



object = exer()
object.log_reg_clf()
object.pca_log_reg_clf()