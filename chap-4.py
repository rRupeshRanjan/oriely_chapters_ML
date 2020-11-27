import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.base import clone
from sklearn import datasets

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.ylim(0,3)
    plt.show()

def linear_regression():
    X = 2 * np.random.rand(100,1)
    y = 4 + 3 * X + np.random.rand(100,1)

    X_b = np.c_[np.ones((100,1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    print(theta_best)

    eta = 0.1
    n_iterations = 1000
    m = 100
    theta_new_best = np.random.rand(2,1)

    for iter in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta_new_best) - y)
        theta_new_best = theta_new_best - eta * gradients

    print(theta_new_best)

    lin_reg = LinearRegression()
    lin_reg.fit(X_b, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.1, penalty=None)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.intercept_, sgd_reg.coef_)

    plot_learning_curves(lin_reg, X, y)

def polynomial_regression():
    m = 100
    X = 6 * np.random.rand(m,1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    # print(lin_reg.intercept_, lin_reg.coef_)

    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

    plot_learning_curves(polynomial_regression, X, y)

def ridge_regression(X, y):
    print("Ridge regression")
    ridge_reg = Ridge(alpha=0.1, solver="cholesky")
    ridge_reg.fit(X, y)
    print(ridge_reg.predict([[1.5]]))

    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X,y)
    print(sgd_reg.predict([[1.5]]))

def lasso_regression(X, y):
    print("Lasso regression")
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    print(lasso_reg.predict([[1.5]]))

    sgd_reg = SGDRegressor(penalty="l1")
    sgd_reg.fit(X,y)
    print(sgd_reg.predict([[1.5]]))

def elasticnet_regression(X,y):
    print("Elastic net")
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X, y)
    print(elastic_net.predict([[1.5]]))

def early_stopping():
    poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.fit_transform(X_val)

    sgd_reg = SGDRegressor(max_iter=1, penalty=None, tol=-np.inf, eta0=0.005, learning_rate="constant", warm_start=True)

    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None

    for epoch in range(100):
        sgd_reg.fit(X_train_poly_scaled, y_train)
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)

        if val_error < minimum_val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)

    print("Done")

def decision_boundaries():
    iris = datasets.load_iris()

    X = iris["data"]
    y = iris["target"]

    lin_reg_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    lin_reg_model.fit(X, y)

    y_predict = lin_reg_model.predict(X[1:2,:])
    y_precict_proba = lin_reg_model.predict_proba(X[1:2, :])

    print(y_predict)
    print(y_precict_proba)

linear_regression()