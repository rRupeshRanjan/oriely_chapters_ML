import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, SVR, LinearSVR

def iris_dataset():
    iris = load_iris()

    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("SVC model", SVC(C=1, kernel="linear"))
    ])

    svm_clf.fit(X,y)

    print(svm_clf.predict([[5.5, 1.7]]))

def moon_dataset():
    X, y = make_moons(n_samples=100, noise=0.15)

    poly_linear_svc_clf = Pipeline([
        ("Poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svc_clf", LinearSVC(C=10, loss="hinge"))
    ])

    poly_kernel_svc_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc_clf", SVC(kernel="poly", degree=3, C=5, coef0=1))
    ])

    rbf_kernel_svc_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc_clf", SVC(kernel="rbf", gamma=5, C=0.0001))
    ])

    svr_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svr_clf", SVR(kernel="poly", degree=3, gamma=0.1, C=100))
    ])

    linear_svr_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svr_clf", SVR(epsilon=1.5))
    ])

    poly_linear_svc_clf.fit(X, y)
    poly_kernel_svc_clf.fit(X, y)
    rbf_kernel_svc_clf.fit(X, y)
    svr_clf.fit(X,y)
    linear_svr_clf.fit(X,y)

    print(poly_linear_svc_clf.predict([[1,1]]))
    print(poly_kernel_svc_clf.predict([[1,1]]))
    print(rbf_kernel_svc_clf.predict([[1,1]]))
    print(svr_clf.predict([[1,1]]))
    print(linear_svr_clf.predict([[1,1]]))

moon_dataset()