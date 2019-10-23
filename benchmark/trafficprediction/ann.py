import numpy as np
import torch

from benchmarkutils import separate_data_by_state
from torch import nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

class TrafficANN(nn.Module):
    def __init__(self, d_in, h_1, h_2, mode='state'):
        super(TrafficANN, self).__init__()
        self.fc1 = nn.Linear(d_in, h_1)
        self.fc2 = nn.Linear(h_1, h_2)
        self.fc_f = nn.Linear(h_2, 1)
        self.fc_q = nn.Linear(h_2, 1)
        self.mode = mode

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        f = self.fc_f(x)
        q = self.fc_q(x)
        return f, q

    def fit_state(self,
              X,
              y_data,
              y_state,
              criterion=nn.L1Loss(),
              n_steps=10000,
              batch_size=32,
              lr=1e-3,
              momentum=0.9,
              weight_decay=0,
              verbose=False):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        X_f, y_f, X_q, y_q = separate_data_by_state(X, y_data, y_state)
        f_indices = np.arange(X_f.shape[0])
        q_indices = np.arange(X_q.shape[0])
        loss_val = 0
        for n in range(n_steps):
            optimizer.zero_grad()
            if n % 2:
                batch_idx = np.random.choice(q_indices, batch_size)
                X_batch = torch.from_numpy(X_q[batch_idx]).float()
                y_batch = torch.from_numpy(y_q[batch_idx]).unsqueeze(1).float()
                _, y_pred = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            else:
                batch_idx = np.random.choice(f_indices, batch_size)
                X_batch = torch.from_numpy(X_f[batch_idx]).float()
                y_batch = torch.from_numpy(y_f[batch_idx]).unsqueeze(1).float()
                y_pred, _ = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            loss_val += loss.item()
            if verbose and (n+1) % 1000 == 0:
                print('MAE for steps {}-{}: {}'.format(n-999, n+1, loss_val/1000))
                loss_val = 0
            if n == int(n_steps/2):
                optimizer = optim.SGD(self.parameters(), lr=lr/10, momentum=momentum, weight_decay=weight_decay)
        return

    def fit_full(self,
                 X,
                 y_data,
                 criterion=nn.L1Loss(),
                 n_steps=10000,
                 batch_size=32,
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=0,
                 verbose=False):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        indices = np.arange(X.shape[0])
        loss_val = 0
        for n in range(n_steps):
            optimizer.zero_grad()
            batch_idx = np.random.choice(indices, batch_size)
            X_batch = torch.from_numpy(X[batch_idx]).float()
            y_batch = torch.from_numpy(y_data[batch_idx]).unsqueeze(1).float()
            y_pred, _ = self.forward(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            if verbose and (n+1) % 1000 == 0:
                print('MAE for steps {}-{}: {}'.format(n-99, n+1, loss_val/1000))
                loss_val = 0
            if n == int(n_steps/2):
                optimizer = optim.SGD(self.parameters(), lr=lr/10, momentum=momentum, weight_decay=weight_decay)

    def predict(self, X, y_state):
        predictions = np.zeros(len(y_state))
        if self.mode == 'state':
            with torch.no_grad():
                f_preds, q_preds = self.forward(torch.from_numpy(X).float())
            f_indices = y_state == 0
            q_indices = y_state == 1
            predictions[f_indices] = (f_preds.squeeze(1).numpy())[f_indices]
            predictions[q_indices] = (q_preds.squeeze(1).numpy())[q_indices]
        else:
            with torch.no_grad():
                preds, _ = self.forward(torch.from_numpy(X).float())
            predictions = preds.squeeze(1).numpy()
        return predictions

