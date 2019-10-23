import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

class LinearRegressor:
    def __init__(self, estimator='linear', alpha=1, mode='state'):
        if estimator not in ['linear', 'lasso', 'ridge']:
            raise ValueError('Please pass linear, lasso, or ridge as estimator name')
        self.estimator = estimator
        self.alpha = alpha
        self.mode = mode

    def fit_state(self, X, y_data, y_state):
        if self.estimator == 'linear':
            self.clf_free = LinearRegression()
            self.clf_queue = LinearRegression()
        elif self.estimator == 'lasso':
            self.clf_free = Lasso(alpha=self.alpha)
            self.clf_queue = Lasso(alpha=self.alpha)
        else:
            self.clf_free = Ridge(alpha=self.alpha)
            self.clf_queue = Ridge(alpha=self.alpha)

        f_indices = y_state == 0
        q_indices = y_state == 1

        X_f = X[f_indices]
        y_f = y_data[f_indices]
        self.clf_free.fit(X_f, y_f)

        X_q = X[q_indices]
        y_q = y_data[q_indices]
        self.clf_queue.fit(X_q, y_q)

    def fit_full(self, X, y):
        if self.estimator == 'linear':
            self.clf_full = LinearRegression()
        elif self.estimator == 'lasso':
            self.clf_full = Lasso(alpha=self.alpha)
        else:
            self.clf_full = Ridge(alpha=self.alpha)
        self.clf_full.fit(X, y)

    def predict(self, X, y_state):
        predictions = np.zeros(len(y_state))
        if self.mode == 'state':
            f_indices = y_state == 0
            q_indices = y_state == 1

            X_f = X[f_indices]
            X_q = X[q_indices]
            if len(X_f.shape) > 1:
                predictions[f_indices] = self.clf_free.predict(X_f)
            else:
                predictions[f_indices] = self.clf_free.predict(np.array([X_f]))
            if len(X_q.shape) > 1:
                predictions[q_indices] = self.clf_queue.predict(X_q)
            else:
                predictions[q_indices] = self.clf_queue.predict([X_q])

        else:
            predictions = self.clf_full.predict(X)
        
        return predictions
