import numpy as np

class HistoricalAverage:
    def __init__(self, sensor_id, normalize_max=False):
        self.sensor = sensor_id
        self.normalize_max = normalize_max

    def fit(self, neighborhood_data, training_dates):
        self.ha_free = {}
        self.ha_queue = {}
        self.ha_full = {}
        values = []
        for date in training_dates:
            date_sensor_dict = neighborhood_data[date][self.sensor]
            for sample in date_sensor_dict:
                entry = date_sensor_dict[sample]
                state = entry['state']
                count = entry['count']
                values.append(count)
                #add to full dictionary
                if sample not in self.ha_full:
                    self.ha_full[sample] = []
                self.ha_full[sample].append(count)
                #add to free dictionary
                if state == 0:
                    if sample not in self.ha_free:
                        self.ha_free[sample] = []
                    self.ha_free[sample].append(count)
                #add to queue dictionary
                else:
                    if sample not in self.ha_queue:
                        self.ha_queue[sample] = []
                    self.ha_queue[sample].append(count)
        #replace lists with means
        max_value = np.max(values)
        #note that each sample is not guaranteed to be in each dictionary
        dictionaries = [self.ha_full, self.ha_free, self.ha_queue]
        for dictionary in dictionaries:
            for sample in dictionary:
                if self.normalize_max:
                    dictionary[sample] = np.mean(np.array(dictionary[sample])/max_value)
                else:
                    dictionary[sample] = np.mean(dictionary[sample])
        return

    def predict(self, y_state, indices, mode='state'):
        predictions = []
        for i in range(len(indices)):
            state = y_state[i]
            idx = indices[i]
            if mode == 'state':
                if state == 0:
                    if idx in self.ha_free:
                        predictions.append(self.ha_free[idx])
                    else:
                        predictions.append(0)
                else:
                    if idx in self.ha_queue:
                        predictions.append(self.ha_queue[idx])
                    else:
                        predictions.append(0)
            else:
                predictions.append(self.ha_full[idx])
        return np.array(predictions)

