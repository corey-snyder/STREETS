import numpy as np

from sklearn.svm import SVR

class SupportVectorRegressor:
    def __init__(self, kernel='rbf', mode='state'):
        self.mode = mode
        self.kernel = kernel

    def fit_state(self, X, y_data, y_state):
        self.clf_free = SVR(kernel=self.kernel, gamma='scale')
        self.clf_queue = SVR(kernel=self.kernel, gamma='scale')

        f_indices = y_state == 0
        q_indices = y_state == 1

        X_f = X[f_indices]
        y_f = y_data[f_indices]
        self.clf_free.fit(X_f, y_f)
        #print('Free-flow Training MAE = {}'.format(np.mean(np.abs(self.clf_free.predict(X_f)-y_f))))

        X_q = X[q_indices]
        y_q = y_data[q_indices]
        self.clf_queue.fit(X_q, y_q)
        #print('Queue Training MAE = {}'.format(np.mean(np.abs(self.clf_queue.predict(X_q)-y_q))))

    def fit_full(self, X, y):
        self.clf_full = SVR(kernel=self.kernel, gamma='scale')
        self.clf_full.fit(X, y)
        #print('Full Training MAE = {}'.format(np.mean(np.abs(self.clf_full.predict(X)-y))))

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
