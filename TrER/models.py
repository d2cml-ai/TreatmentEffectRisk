from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np


def tau_predict(X, y, weights):
    model = LinearRegression()

    # Fit the model with weights
    model.fit(X, y, sample_weight=weights)

    # Make predictions
    tau_pred = model.predict(X)
    return tau_pred


def mu_calculate(data, cvgroup, y_ref, X_ref, A_col="A", sw_col="sw", K=5):
    mu0_pred = np.zeros(len(data))
    mu1_pred = np.zeros(len(data))

    for k in range(1, K + 1):
        train_indices = cvgroup != k
        test_indices = cvgroup == k

        clf0 = GradientBoostingClassifier(tol=0.1)
        clf1 = GradientBoostingClassifier(tol=0.1)

        train_A0 = train_indices & (data[A_col] == 0)
        train_A1 = train_indices & (data[A_col] == 1)

        clf0.fit(X_ref[train_A0], y_ref[train_A0], sample_weight=data[sw_col][train_A0])
        clf1.fit(X_ref[train_A1], y_ref[train_A1], sample_weight=data[sw_col][train_A1])

        mu0_pred[test_indices] = clf0.predict_proba(X_ref[test_indices])[:, 1]
        mu1_pred[test_indices] = clf1.predict_proba(X_ref[test_indices])[:, 1]

    return mu0_pred, mu1_pred


def var_calculate(
    data,
    X_ref,
    cvgroup,
    K=5,
    y_col="Y",
    mu0_col="mu0",
    sw_col="sw",
    A_col="A",
    rearrangement=False,
):
    if X_ref is None:
        raise ValueError("X_ref must be provided as a pandas DataFrame.")

    var0_pred = np.zeros(len(data))
    var1_pred = np.zeros(len(data))
    X_ref[A_col] = data[A_col]

    for k in range(1, K + 1):
        train_indices = cvgroup != k
        test_indices = cvgroup == k

        fvar = GradientBoostingRegressor()

        y_ref0 = (data[y_col] - data[mu0_col]) ** 2
        sw = data[sw_col]

        # Ajustar el modelo para var0
        fvar.fit(
            X_ref[train_indices], y_ref0[train_indices], sample_weight=sw[train_indices]
        )

        # Predecir var0 y var1
        X_test0 = X_ref[test_indices].copy()
        X_test1 = X_ref[test_indices].copy()

        X_test0[A_col] = 0
        X_test1[A_col] = 1

        var0_pred[test_indices] = fvar.predict(X_test0)
        var1_pred[test_indices] = fvar.predict(X_test1)
    if rearrangement:
        var0_pred = var0_pred * (var0_pred > 0)
        var1_pred = var1_pred * (var1_pred > 0)
    return var0_pred, var1_pred
