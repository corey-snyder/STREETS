import numpy as np
import os
import pickle
import argparse
import statsmodels
import warnings

from benchmarkutils import *
from ha import HistoricalAverage
from rfr import RandomForestRegressor
from svr import SupportVectorRegressor
from ann import TrafficANN
from arima import ARIMA
from linear import LinearRegressor
from tqdm import tqdm

warnings.simplefilter("ignore", statsmodels.tools.sm_exceptions.ConvergenceWarning)
warnings.simplefilter("ignore", statsmodels.tools.sm_exceptions.HessianInversionWarning)

def load_model(model_str, sensor, feature_dimension, history):
    if model_str == 'ha':
        model_state = HistoricalAverage(sensor)
        model_full = HistoricalAverage(sensor)
    elif model_str == 'rfr':
        model_state = RandomForestRegressor(n_estimators=50, mode='state')
        model_full = RandomForestRegressor(n_estimators=50, mode='full')
    elif model_str == 'svr':
        model_state = SupportVectorRegressor(kernel='rbf', mode='state')
        model_full = SupportVectorRegressor(kernel='rbf', mode='full')
    elif model_str == 'ann':
        model_state = TrafficANN(feature_dimension, 200, 100, mode='state')
        model_full = TrafficANN(feature_dimension, 200, 100, mode='full')
    elif model_str == 'arima':
        model_state = None
        model_full = ARIMA(3, 1, history)
    else:
        model_state = LinearRegressor(estimator='ridge', alpha=1, mode='state')
        model_full = LinearRegressor(estimator='ridge', alpha=1, mode='full')
    
    return model_state, model_full

def main():
    #parse community, sensors, model
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, help='string to identify model')
    args = parser.parse_args()

    community = 'gurnee'
    sensors = [28, 56, 84]

    model_str = args.model
    if model_str not in ['ha', 'rfr', 'svr', 'ann', 'arima', 'linear']:
        raise ValueError('Select ha, rfr, svr, ann, arima, or linear')

    graph_path = os.path.join('graphs', community, '{}-graph.json'.format(community))
    traffic_data_path = os.path.join('trafficdata','{}-traffic-dictionary.json'.format(community))
    training_dates = ['2019-6-5', '2019-6-6', '2019-6-7',
                      '2019-6-10', '2019-6-11', '2019-6-12', '2019-6-13', '2019-6-14',
                      '2019-6-17', '2019-6-18', '2019-6-19', '2019-6-20', '2019-6-21',
                      '2019-6-24', '2019-6-25', '2019-6-26', '2019-6-27', '2019-6-28',
                      '2019-7-1', '2019-7-2', '2019-7-3', '2019-7-4', '2019-7-5',
                      '2019-7-8', '2019-7-9', '2019-7-10', '2019-7-11', '2019-7-12']
    testing_dates = ['2019-7-15', '2019-7-16', '2019-7-17', '2019-7-18']
    A, D, S = load_traffic_graph(graph_path)
    T = 5
    n_trials = 10
    results_dict = {}
    for s in sensors:
        for test_date in tqdm(testing_dates):
            for K in [0, 1, 2]:
                neighborhood_dict, neighbors = load_K_hop_neighborhood(A, s, S, K, skip_outbound=False)
                traffic_data = load_traffic_data(traffic_data_path, training_dates+testing_dates)
                neighborhood_data = load_neighborhood_data(traffic_data, neighbors)
                interpolated_data, n_samples = interpolate_traffic_data(neighborhood_data, T)
                for history in [3, 6]:
                    for horizon in [3, 6]:
                        for use_state in [True, False]:
                            results_dict[(s, test_date, K, history, horizon, use_state)] = {'state': [], 'full': []}
                            X_train, y_data_train, y_state_train, max_dict = load_train_data(training_dates,
                                                                                             interpolated_data,
                                                                                             neighborhood_dict,
                                                                                             neighbors,
                                                                                             history,
                                                                                             horizon,
                                                                                             n_samples,
                                                                                             use_state=use_state,
                                                                                             normalize_max=False)
                            X_test, y_data_test, y_state_test, sample_indices = load_test_data(test_date,
                                                                                               interpolated_data,
                                                                                               neighborhood_dict,
                                                                                               neighbors,
                                                                                               history,
                                                                                               horizon,
                                                                                               n_samples,
                                                                                               max_dict,
                                                                                               use_state=use_state)
                            if model_str == 'arima':
                                y_arima_train, y_arima_test = load_arima_data(training_dates,
                                                                              testing_dates,
                                                                              test_date,
                                                                              interpolated_data,
                                                                              neighborhood_dict,
                                                                              neighbors,
                                                                              history,
                                                                              horizon,
                                                                              n_samples)
                            model_state, model_full = load_model(model_str, s, X_train.shape[1], history)
                            for n in range(n_trials):
                                if model_str == 'ha':
                                    model_state.fit(interpolated_data, training_dates)
                                    model_full.fit(interpolated_data, training_dates)
                                    preds_state = model_state.predict(y_state_test, sample_indices)
                                    preds_full = model_full.predict(y_state_test, sample_indices, mode='full')
                                elif model_str == 'arima':
                                    preds_state = None
                                    preds_full = model_full.predict(y_arima_train, y_arima_test, horizon)
                                else:
                                    model_state.fit_state(X_train, y_data_train, y_state_train)
                                    model_full.fit_full(X_train, y_data_train)
                                    preds_state = model_state.predict(X_test, y_state_test)
                                    preds_full = model_full.predict(X_test, y_state_test)
                                if model_str != 'arima':
                                    state_result = mae(preds_state, y_data_test)
                                    full_result = mae(preds_full, y_data_test)
                                else:
                                    state_result = -1
                                    full_result = mae(preds_full, y_data_test)
                                results_dict[(s, test_date, K, history, horizon, use_state)]['state'].append(state_result)
                                results_dict[(s, test_date, K, history, horizon, use_state)]['full'].append(full_result)

    with open('{}-results.pkl'.format(model_str), 'wb') as f:
        pickle.dump(results_dict, f)

if __name__ == '__main__':
    main()
