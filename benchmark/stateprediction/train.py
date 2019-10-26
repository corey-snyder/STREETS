import sys
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch import optim
from torchvision import models
from torch.utils.data import DataLoader
from dataloader import FVQDataset
from tqdm import tqdm

MODEL_PATH = 'logs'
P = 0.8
def train_clf(clf, n_epochs, batch_size, lr):
    path = '../../data/trafficstate'
    dataset = FVQDataset(path)
    n_samples = len(dataset)
    n_f = 0
    n_q = 0
    for _, label in dataset:
        if label==0:
            n_f += 1
        elif label==1:
            n_q += 1
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_indices = indices[:int(n_samples*P)]
    test_indices = indices[int(n_samples*P):]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2)
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Testing size: {}
        Number of free-flow images: {}
        Number of queue images: {}
    '''.format(n_epochs, batch_size, lr, len(train_indices), len(test_indices), n_f, n_q))


    optimizer = optim.SGD(clf.parameters(),
                          lr=lr,
                          momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, n_epochs))
        if (epoch+1) % 5 == 0:
            lr = lr/10
            optimizer = optim.SGD(clf.parameters(),
                                  lr = lr,
                                  momentum=0.9)
            print('New learning rate: {}'.format(lr))
        clf.train()
        epoch_loss = 0
        epoch_acc = 0
        n_total = 0
        for images, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = clf(images)

            loss = criterion(preds, labels)
            epoch_loss += loss.item()

            epoch_acc += torch.sum(torch.max(preds, dim=1)[1] == labels).item()
            n_total += images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(clf.state_dict(), os.path.join(MODEL_PATH, 'CP-{}.pth'.format(epoch + 1)))

        test_loss, test_acc, test_total, q_acc, f_acc = eval_clf(clf, test_loader, criterion)
        print('Training Loss: {:.4e}'.format(epoch_loss/len(train_loader)))
        print('Training Accuracy: {:.4f}%'.format(epoch_acc/n_total*100))

        print('Testing Loss: {:.4e}'.format(test_loss/len(test_loader)))
        print('Testing Accuracy: {:.4f}%'.format(test_acc/test_total*100))
        print('Testing Free-flow Accuracy: {:.4f}%'.format(f_acc*100))
        print('Testing Queue Accuracy: {:.4f}%'.format(q_acc*100))
 
def eval_clf(clf, dataloader, criterion):
    clf.eval()    
    eval_loss = 0
    eval_acc = 0
    eval_total = 0
    f_acc = 0
    f_total = 0
    q_acc = 0
    q_total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = clf(images)

            loss = criterion(preds, labels)
            eval_loss += loss.item()
            pred_classes = torch.max(preds, dim=1)[1]
            eval_acc += torch.sum(pred_classes == labels).item()
            for i, l in enumerate(labels):
                if l:
                    if pred_classes[i] == l:
                        q_acc += 1
                    q_total += 1
                else:
                    if pred_classes[i] == l:
                        f_acc += 1
                    f_total += 1
            eval_total += images.size(0)

    return eval_loss, eval_acc, eval_total, q_acc/q_total, f_acc/f_total

class TrafficStateClassifier(nn.Module):
    def __init__(self, n_channels, n_classes=2):
        super(TrafficStateClassifier, self).__init__()
        self.clf = models.resnet50(pretrained=True)
        self.clf.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        n_features = self.clf.fc.in_features
        self.clf.fc = nn.Linear(n_features, n_classes)
    
    def forward(self, x):
        x = self.clf(x)
        return x

if __name__ == '__main__':
    n_channels = 3
    clf = TrafficStateClassifier(n_channels, n_classes=2)
    clf.to(DEVICE)
    n_epochs = 20
    batch_size = 8
    lr = 1e-3
    train_clf(clf, n_epochs, batch_size, lr)
