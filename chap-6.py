import numpy as np
from scipy.stats import mode
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def chap():
    iris = load_iris()
    X = iris.data[:,2:]
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=4)
    tree_clf.fit(X, y)

    export_graphviz(
            tree_clf,
            out_file="iris_tree.dot",
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

    tree_clf.predict_proba([[5, 1.5]])

def exercise():
    x, y = datasets.make_moons(n_samples=10000, noise=0.4)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    parameters = {'max_leaf_nodes':[3,4,5,6,8,7,9]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters, cv=3, return_train_score=True)
    grid_search.fit(x_train, y_train)

    y_pred = grid_search.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    print("Building a forest.")

    n_trees = 1000
    n_instances = 100

    mini_sets = []

    ss = ShuffleSplit(n_splits=n_trees, test_size=len(x_train) - n_instances, random_state=42)
    for mini_train_index, mini_test_index in ss.split(x_train):
        x_mini_train = x_train[mini_train_index]
        y_mini_train = y_train[mini_train_index]
        mini_sets.append((x_mini_train,y_mini_train))

    forest = [clone(grid_search.best_estimator_) for _ in range(n_trees)]
    accuracy_scores = []

    for tree, (x_mini_train, y_mini_train) in zip(forest, mini_sets):
        tree.fit(x_mini_train, y_mini_train)
        y_pred = tree.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    print(np.mean(accuracy_scores))

    print("Magic coming up")

    y_pred = np.empty([n_trees, len(x_test)], dtype=np.uint8)
    for tree_index, tree in enumerate(forest):
        y_pred[tree_index] = tree.predict(x_test)

    y_pred_majority_votes, n_votes = mode(y_pred, axis=0)
    print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))

exercise()