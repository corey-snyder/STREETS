import numpy as np
import os
import json
"""
This file contains utility functions for loading and preprocessing data
from STREETS.
"""

def load_traffic_graph(path):
    #load JSON for provided path
    with open(path, 'r') as f:
        graph = json.load(f)
    #return adjacency matrix, distance matrix, and sensor dictionary
    adjacency_matrix = np.array(graph['adjacency-matrix'])
    distance_matrix = np.array(graph['distance-matrix'])
    sensor_dict = graph['sensor-dictionary']
    return adjacency_matrix, distance_matrix, sensor_dict

def get_bound(sensor_dict, sensor_id):
    name = sensor_dict[str(sensor_id)][1]
    return 'inbound' if 'inbound' in name else 'outbound'

def load_K_hop_neighborhood(adjacency_matrix,
                            sensor_id,
                            sensor_dict,
                            K,
                            skip_outbound=True):
    #initialize neighborhood dictionary (nd)
    #keys in [0,K], nd[k] = {inbound: [list of inbound sensor IDs], outbound: [list of outbound sensors]}
    home_sensor_bound = get_bound(sensor_dict, sensor_id)
    neighborhood_dict = {0: {'inbound': [], 'outbound': []}}
    neighborhood_dict[0][home_sensor_bound].append(sensor_id)
    neighbors = [sensor_id]
    for k in range(1, K+1):
        #extract previous hop's sensor IDs
        if skip_outbound:
            neighborhood_dict, neighbors = grow_neighborhood_skip(neighborhood_dict,
                                                                  adjacency_matrix,
                                                                  sensor_dict,
                                                                  neighbors,
                                                                  k)
        else:
            neighborhood_dict, neighbors = grow_neighborhood(neighborhood_dict,
                                                             adjacency_matrix,
                                                             sensor_dict,
                                                             neighbors,
                                                             k)
    return neighborhood_dict, neighbors

def grow_neighborhood(neighborhood_dict, adjacency_matrix, sensor_dict, neighbors, k):
    previous_sensors = np.unique(neighborhood_dict[k-1]['inbound']+neighborhood_dict[k-1]['outbound'])
    neighborhood_dict[k] = {'inbound': [], 'outbound': []}
    for sensor in previous_sensors:
        #upstream sensors
        upstream = np.where(adjacency_matrix[:, sensor] > 0)[0]
        #downstream sensors
        downstream = np.where(adjacency_matrix[sensor] > 0)[0]
        #separate into inbound and outbound sensors
        all_sensors = np.concatenate((upstream, downstream))
        for s in all_sensors:
            if s not in neighbors:
                bound = get_bound(sensor_dict, s)
                neighborhood_dict[k][bound].append(s)
                neighbors.append(s)
    return neighborhood_dict, neighbors
        
def grow_neighborhood_skip(neighborhood_dict, adjacency_matrix, sensor_dict, neighbors, k):
    #edge of neighborhood should only be inbound sensors

    #handle case the home sensor is an outbound sensor and this is first growth
    if k == 1 and len(neighborhood_dict[0]['outbound']):
        previous_sensors = neighborhood_dict[0]['outbound']
    #all other cases
    else:
        previous_sensors = neighborhood_dict[k-1]['inbound']
    neighborhood_dict[k] = {'inbound': [], 'outbound': []}
    for sensor in previous_sensors:
        #upstream sensors
        upstream = np.where(adjacency_matrix[:, sensor] > 0)[0]
        #skip another edge for outbound sensors
        for u in upstream:
            if get_bound(sensor_dict, u) is 'outbound':
                upstream = np.concatenate((upstream, np.where(adjacency_matrix[:, u] > 0)[0]))
        #downstream sensors
        downstream = np.where(adjacency_matrix[sensor] > 0)[0]
        #skip edge for outbound sensors
        for d in downstream:
            if get_bound(sensor_dict, d) is 'outbound':
                downstream = np.concatenate((downstream, np.where(adjacency_matrix[:, d] > 0)[0]))
        all_sensors = np.concatenate((upstream, downstream))
        for s in all_sensors:
            if s not in neighbors:
                bound = get_bound(sensor_dict, s)
                neighborhood_dict[k][bound].append(s)
                neighbors.append(s)
    return neighborhood_dict, neighbors

def load_traffic_data(traffic_data_path, dates='all'):
    #load JSON for provided path
    with open(traffic_data_path, 'r') as f:
        data = json.load(f)
    #extract only provided dates
    traffic_data = {} #td[date][sensor_id][image_name] = {count: int, timestamp = [int, int], state: int}
    if dates == 'all':
        traffic_data = data
    else:
        for d in dates:
            traffic_data[d] = data[d]
    return traffic_data

def interpolate_traffic_data(traffic_data, T):
    #apply nearest neighbor interpolation at sampling interval T to each sensor's data
    #inerpolated_data[date][sensor_id][sample_number] = {count: int, state: int}
    interpolated_data = {}
    min_hour = 5
    max_hour = 23
    n_samples = int(60*(max_hour-min_hour)/T)
    #iterate through each date
    for d in traffic_data:
        date_data = traffic_data[d]
        interpolated_data[d] = {}
    #iterate through each sensor
        for s in date_data:
            sensor_data = date_data[s]
            interpolated_data[d][s] = {}
            timestamps, image_names = load_timestamps(sensor_data, min_hour)
    #iterate through each sample number
            for n in range(n_samples):
                time_value = n*T
                if len(image_names):
                    nearest_image_name = find_nearest_image(time_value, timestamps, image_names)
                    data_of_interest = sensor_data[nearest_image_name]
                    interpolated_data[d][s][n] = {'count': data_of_interest['count'],
                                                  'state': data_of_interest['state']}
                else:
                    interpolated_data[d][s][n] = {'count': 0, 'state': 0}
    return interpolated_data, n_samples

def load_timestamps(sensor_data, min_hour):
    timestamps = []
    names = []
    for name in sensor_data:
        timestamp = sensor_data[name]['timestamp']
        value = (timestamp[0]-min_hour)*60 + timestamp[1] #minutes since minimum time
        timestamps.append(value)
        names.append(name)
    return np.array(timestamps), np.array(names)

def find_nearest_image(time_value, timestamps, image_names):
    nearest_idx = np.argmin(np.abs(timestamps-time_value))
    image_name = image_names[nearest_idx]
    return image_name

def load_neighborhood_data(traffic_data, neighbors):
    #extract data only from neighborhood sensors
    neighborhood_data = {}
    #extract one date at a time, checking if sensor is in the neighborhood
    for date in traffic_data:
        neighborhood_data[date] = {}
        for sensor in neighbors:
            neighborhood_data[date][sensor] = traffic_data[date][str(sensor)]
    return neighborhood_data

def build_feature_vector(idx, n_samples, date, neighborhood_data, neighborhood, history, use_state, use_time_of_day, max_dict):
    #build feature vector for a particular sample at the home sensor
    features = []
    #build list of lists for each set of features
    if use_time_of_day:
        features.append([idx])
    for k in neighborhood:
        for bound in neighborhood[k]:
            for sensor in neighborhood[k][bound]:
                sensor_data = neighborhood_data[date][sensor]
                features.append(build_feature_vector_chunk(idx, sensor, sensor_data, history, use_state, max_dict))
    ret_features = [f for feature_list in features for f in feature_list]
    return ret_features

def build_feature_vector_chunk(idx, sensor_id, sensor_data, history, use_state, max_dict):
    #assemble the contribution to the feature vector for a particular sensor in the neighborhood
    chunk = []
    for h in range(history):
        if len(max_dict):
            chunk.append(sensor_data[idx-h]['count']/max_dict[sensor_id])
        else:
            chunk.append(sensor_data[idx-h]['count'])
        if use_state:
            chunk.append(sensor_data[idx-h]['state'])
    return chunk

def convert_time_of_day(idx, n_samples):
    #divide into 2-hr long windows; 18 hours/2 (hrs/window) = 9 windows
    time_of_day = [0 for i in range(9)]
    window_width = int(n_samples/9)
    time_idx = int(idx/window_width)
    time_of_day[time_idx] = 1
    return time_of_day

def compute_max_dictionary(dates, neighborhood_data, neighbors):
    max_dict = {}
    for sensor in neighbors:
        values = []
        for date in dates:
            neighbor_dict = neighborhood_data[date][sensor]
            for idx in neighbor_dict:
                values.append(neighbor_dict[idx]['count'])
        max_dict[sensor] = np.max(values)
    return max_dict

def get_home_sensor(neighborhood):
    zero_hop = neighborhood[0]
    sensors = zero_hop['inbound'] + zero_hop['outbound']
    if len(sensors) != 1:
        raise ValueError('Neighborhood incorrectly constructed')
    return sensors[0]

def get_gt(idx, date, sensor_id, neighborhood_data, max_dict):
    if len(max_dict):
        return neighborhood_data[date][sensor_id][idx]['count']/max_dict[sensor_id]
    else:
        return neighborhood_data[date][sensor_id][idx]['count']

def get_state(idx, date, sensor_id, neighborhood_data):
    return neighborhood_data[date][sensor_id][idx]['state']

def load_train_data(dates,
                    neighborhood_data,
                    neighborhood,
                    neighbors,
                    history,
                    horizon,
                    n_samples,
                    use_state=True,
                    use_time_of_day=True,
                    normalize_max=False):

    X = [] #input features
    y_data = [] #output target
    y_state = [] #associated traffic state for home sensor
    valid_sample_indices = np.arange(history, n_samples-horizon)

    #initialize home sensor and max dictionary (it being left empty indicates normalize_max=False)
    home_sensor = get_home_sensor(neighborhood)
    max_dict = {}
    #compute maximum count dictionary for each sensor
    if normalize_max:
        max_dict = compute_max_dictionary(dates, neighborhood_data, neighbors)
    #iterate through each date
    for date in dates:
    #iterate through each valid sample index
        for idx in valid_sample_indices:
    #load feature vector, target, and traffic state
            X.append(build_feature_vector(idx, n_samples, date, neighborhood_data, neighborhood, history, use_state, use_time_of_day, max_dict))
            y_data.append(get_gt(idx+horizon, date, home_sensor, neighborhood_data, max_dict))
            y_state.append(get_state(idx+horizon, date, home_sensor, neighborhood_data))

    return np.array(X), np.array(y_data), np.array(y_state), max_dict

def load_test_data(date,
                   neighborhood_data,
                   neighborhood,
                   neighbors,
                   history,
                   horizon,
                   n_samples,
                   max_dict,
                   use_state=True,
                   use_time_of_day=True):

    X = [] #input features
    y_data = [] #output target
    y_state = [] #associated traffic state for home sensor
    valid_sample_indices = np.arange(history, n_samples-horizon)

    #initialize home sensor
    home_sensor = get_home_sensor(neighborhood)

    #iterate through each valid sample index
    for idx in valid_sample_indices:
    #load feature vector, target, and traffic state
        X.append(build_feature_vector(idx, n_samples, date, neighborhood_data, neighborhood, history, use_state, use_time_of_day, max_dict))
        y_data.append(get_gt(idx+horizon, date, home_sensor, neighborhood_data, max_dict))
        y_state.append(get_state(idx+horizon, date, home_sensor, neighborhood_data))

    return np.array(X), np.array(y_data), np.array(y_state), valid_sample_indices

def separate_data_by_state(X, y_data, y_state):
    f_indices = y_state == 0
    q_indices = y_state == 1

    X_f = X[f_indices]
    y_f = y_data[f_indices]

    X_q = X[q_indices]
    y_q = y_data[q_indices]

    return X_f, y_f, X_q, y_q

def load_arima_data(training_dates,
                    testing_dates,
                    test_date,
                    interpolated_data,
                    neighborhood_dict,
                    neighbors,
                    history,
                    horizon,
                    n_samples):
    #identify three days of warm start data
    test_date_idx = np.nonzero(np.array(testing_dates)==test_date)[0][0]
    if test_date_idx < 2:
        training_train_dates = training_dates[test_date_idx-2:]
    else:
        train_dates = []
    training_test_dates = testing_dates[:test_date_idx]
    training_data_dates = training_train_dates+training_test_dates
    _, y_data_train_arima, _, _ = load_train_data(training_data_dates,
                                                  interpolated_data,
                                                  neighborhood_dict,
                                                  neighbors,
                                                  0,
                                                  horizon,
                                                  n_samples)
    _, y_data_test_arima, _, _ = load_test_data(test_date,
                                                interpolated_data,
                                                neighborhood_dict,
                                                neighbors,
                                                0,
                                                horizon,
                                                n_samples,
                                                {})
    y_train = np.concatenate((y_data_train_arima, y_data_test_arima[:history])).astype(float)
    y_test = y_data_test_arima[history:].astype(float)
    y_train[y_train == 0] += 1e-2
    y_test[y_test == 0] += 1e-2
    return y_train, y_test

def mae(x, y):
    return np.mean(np.abs(x-y))

def mse(x, y):
    return np.mean((x-y)**2)

def rmse(x, y):
    return np.sqrt(mse(x, y))
