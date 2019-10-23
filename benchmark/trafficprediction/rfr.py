import numpy as np

from sklearn.ensemble import RandomForestRegressor as RFR

class RandomForestRegressor:
    def __init__(self, n_estimators=50, criterion='mae',mode='state'):
        self.mode = mode
        self.n_estimators = n_estimators
        self.criterion = criterion

    def fit_state(self, X, y_data, y_state):
        self.clf_free = RFR(n_estimators = self.n_estimators, criterion = self.criterion)
        self.clf_queue = RFR(n_estimators = self.n_estimators, criterion = self.criterion)

        f_indices = y_state == 0
        q_indices = y_state == 1

        X_f = X[f_indices]
        y_f = y_data[f_indices]
        self.clf_free.fit(X_f, y_f)

        X_q = X[q_indices]
        y_q = y_data[q_indices]
        self.clf_queue.fit(X_q, y_q)

    def fit_full(self, X, y):
        self.clf_full = RFR(n_estimators = self.n_estimators, criterion = self.criterion)
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
