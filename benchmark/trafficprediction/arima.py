import numpy as np

from statsmodels.tsa.arima_model import ARIMA as arima
from tqdm import tqdm

class ARIMA:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def fit_forecast(self, y, h):
        clf = arima(y, order=(self.p, self.d, self.q))
        clf_fit = clf.fit(disp=0)
        return clf_fit.forecast(steps = h)[0][-1]

    def predict(self, y_train, y_test, h):
        predictions = np.zeros(len(y_test))
        y_avail = y_train.copy()
        for n in range(len(y_test)):
            predictions[n] = self.fit_forecast(y_avail, h)
            y_avail = np.concatenate((y_avail, np.array([y_test[n]])))
        return predictions
