import xgboost
import numpy as np
from sklearn import datasets
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier,\
    GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class chapter():
    x, y = datasets.make_moons(n_samples=10000, noise=0.4)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    def plain_ensemble(self):
        log_clf = LogisticRegression()
        svc_clf = SVC(probability=True)
        rnd_clf = RandomForestClassifier()

        voting_clf = VotingClassifier(estimators=[('lr',log_clf), ('rf',rnd_clf), ('svc', svc_clf)], voting='soft')

        for clf in [log_clf, svc_clf, rnd_clf, voting_clf]:
            clf.fit(self.x_train, self.y_train)
            y_pred = clf.predict(self.x_test)
            print(clf.__class__.__name__, accuracy_score(self.y_test, y_pred))

    def bagging_tree_ensemble(self, bootstrap=True):
        if bootstrap:
            print("Bagging Decision tree ensemble")
        else:
            print("Pasting Decision tree ensemble")

        bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=500, n_jobs=-1,
                          max_samples=100, bootstrap=bootstrap,oob_score=bootstrap)
        bag_clf.fit(self.x_train, self.y_train)
        y_pred = bag_clf.predict(self.x_test)
        if(bootstrap):
            print("oob score: {0}".format(bag_clf.oob_score_))
        print(accuracy_score(self.y_test, y_pred))

    def extra_trees(self):
        
        extra_tree_clf = ExtraTreesClassifier(max_depth=16)
        extra_tree_clf.fit(self.x_train, self.self.y_train)
        y_pred = extra_tree_clf.predict(self.self.x_test)
        print("Accuracy: {0}".format(accuracy_score(self.self.y_test, y_pred)))

    def feature_importance(self):
        iris_data = datasets.load_iris()
        rnd_for_clf = RandomForestClassifier()
        rnd_for_clf.fit(iris_data["data"], iris_data["target"])

        for feature, score in zip(iris_data["feature_names"], rnd_for_clf.feature_importances_):
            print(feature, score)

    def grad_boost_reg(self):
        grbt = GradientBoostingRegressor(max_depth=2, n_estimators=100)
        grbt.fit(self.x_train, self.y_train)

        errors = [mean_squared_error(self.y_test, y_pred) for y_pred in grbt.staged_predict(self.x_test)]
        n_estimator_best = np.argmin(errors) + 1

        grbt_best = GradientBoostingRegressor(max_depth=2, n_estimators=n_estimator_best)
        grbt_best.fit(self.x_train, self.y_train)

        y_pred = [1 if x > 0.5 else 0 for x in grbt_best.predict(self.x_test)]
        print(accuracy_score(self.y_test, y_pred))

    def xgboost_reg(self):
        xgb = xgboost.XGBRegressor()
        xgb.fit(self.x_train, self.y_train, eval_set=[(self.x_test, self.y_test)], early_stopping_rounds=2)

        y_pred = [1 if x > 0.5 else 0 for x in xgb.predict(self.x_test)]
        print(accuracy_score(self.y_test, y_pred))

class exer():
    def que8(self):
        mnist_data = datasets.fetch_openml('mnist_784', version=1)
        x, y = mnist_data["data"], mnist_data["target"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)

        random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        extra_tree_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
        svc_clf = SVC(random_state=42, probability=True)
        estimators = [random_forest_clf, extra_tree_clf, svc_clf]

        for estimator in estimators:
            estimator.fit(x_train, y_train)

        named_estimators = [("random forest", random_forest_clf),
                            ("extra tree", extra_tree_clf),
                            ("svc", svc_clf)]

        voting_clf = VotingClassifier(estimators=named_estimators, voting='hard')

        voting_clf.fit(x_train, y_train)

        print(voting_clf.score(x_val, y_val))

        for estimator in voting_clf.estimators_:
            print(estimator.score(x_val, y_val))

        voting_clf.set_params(svc_clf=None)

        x_val_predictions = np.empty((len(x_val), len(estimators)), dtype=np.float32)

        for index, estimator in enumerate(estimators):
            x_val_predictions[:, index] = estimator.predict(x_val)

        rnd_forest_blender = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True)
        rnd_forest_blender.fit(x_val_predictions, y_val)

        x_test_predictions = np.empty((len(x_test), len(estimators)), dtype=np.float32)

        for index, estimator in enumerate(estimators):
            x_test_predictions[:, index] = estimator.predict(x_test)

        y_pred = rnd_forest_blender.predict(x_test_predictions)

        print(accuracy_score(y_test, y_pred))


e = exer()
e.que8()
