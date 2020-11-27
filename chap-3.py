from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml(name='mnist_784', version=1)

# separating features and target
x, y = mnist["data"], mnist["target"].astype(np.uint8)

some_digit = x[0]

# separating training and test data
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

def binary_classification():
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train_5)

    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    for train_index, test_index in skfolds.split(x_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = x_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = x_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

    cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")

    y_pred_5 = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
    confusion_matrix(y_train_5, y_pred_5)

    prec_score = precision_score(y_train_5, y_pred_5)
    rec_score = recall_score(y_train_5, y_pred_5)
    f1_score(y_train_5, y_pred_5)

    y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")
    precision, recall, thresholds = precision_recall_curve(y_train_5, y_scores)

    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    roc_auc_score(y_train_5, y_scores)

    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    fpr_forest, tpr_forest, thresholds = roc_curve(y_train_5, y_scores_forest)

def multiclass_classification():

    svc_clf = SVC(max_iter=1)
    svc_clf.fit(x_train, y_train)
    print(svc_clf.predict([some_digit]))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))

    cross_val_score(svc_clf, x_train, y_train, cv=3, scoring="accuracy")
    cross_val_score(svc_clf, x_train_scaled, y_train, cv=3, scoring="accuracy")

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train)
    cross_val_score(sgd_clf, x_train, y_train, scoring="accuracy")

def multilabel_classification():
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train %2 == 0)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train, y_multilabel)

    knn_clf.predict([some_digit])

    y_pred_knn = cross_val_predict(knn_clf, x_train, y_multilabel, cv=3)
    f1_score(y_train, y_pred_knn, average="weighted")
