"""
import packages
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter

"""
define function, class
"""
def make_vocab(tokenizer):
    counter = Counter(sum(tokenizer, []))
    vocab_sorted = sorted(counter.items(), key = lambda x:x[1], reverse = True)

    vocab = {}
    for i, (word, frequency) in enumerate(vocab_sorted):
        vocab[word] = i+1
    vocab['[UNK]'] = len(vocab) + 1
    return vocab

def encoding(data, vocab):
    encoded_data = []
    print("====================encoding====================")
    for sentence in tqdm(data):
        encoded_sentence = []
        for word in sentence:
            try:
                encoded_sentence.append(vocab[word])
            except KeyError:
                encoded_sentence.append(vocab['[UNK]'])
        encoded_data.append(encoded_sentence)

    max_len = max(len(item) for item in encoded_data)
    for sentence in encoded_data:
        while len(sentence) < max_len:
            sentence.append(0)
    for sentence in encoded_data:
        while len(sentence) < max_len:
            sentence.append(0)
    return torch.LongTensor(encoded_data)

def device_n_seed(seed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return device

def y_one_hot(y_data):
    y_counter = Counter(y_data)

    y_vocab = {}
    for i, word in enumerate(y_counter):
        y_vocab[word] = i+1

    one_hot_y = []
    for y_label in list(y_data):
        one_hot_vector = [0]*(len(y_vocab))
        one_hot_vector[y_vocab[y_label]-1] = 1
        one_hot_y.append(one_hot_vector)

    return torch.Tensor(one_hot_y)

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(CustomDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

# CNN-rand model
class CNN_rand(nn.Module):
    def __init__(self, 
                embedding_size, 
                num_filters, 
                dropout_rate, 
                y_dim,
                filters,
                vocab_size):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv1d_0 = nn.Conv1d(embedding_size, num_filters, filters[0])
        self.conv1d_1 = nn.Conv1d(embedding_size, num_filters, filters[1])
        self.conv1d_2 = nn.Conv1d(embedding_size, num_filters, filters[2])
        self.fc = nn.Linear(len(filters) * num_filters, y_dim)

    def forward(self, x):
        embedding_x = self.embedding(x).transpose(1,2)
        
        out_0 = self.dropout(F.relu(self.conv1d_0(embedding_x)))
        out_1 = self.dropout(F.relu(self.conv1d_1(embedding_x)))
        out_2 = self.dropout(F.relu(self.conv1d_2(embedding_x)))

        out_0 = F.max_pool1d(out_0, out_0.shape[2])
        out_1 = F.max_pool1d(out_1, out_1.shape[2])
        out_2 = F.max_pool1d(out_2, out_2.shape[2])

        out = self.dropout(torch.cat((out_0, out_1, out_2), dim = 1))
        out = out.view(out.size(0), -1)
        return F.softmax(self.fc(out))

"""
load data
"""
dataset = np.array([['title', 'label','train']])
with open('ynat-v1.1_train.json') as f: # 45678 samples
    print("====================train load====================")
    for line in tqdm(json.load(f)):
        data = np.array([[line['title'], line['label'],'1']])
        dataset = np.concatenate((dataset, data), axis=0)

with open('ynat-v1.1_dev.json') as f: # 9107 samples
    print("====================test load====================")
    for line in tqdm(json.load(f)):
        data = np.array([[line['title'], line['label'],'0']])
        dataset = np.concatenate((dataset, data), axis=0)

dataframe = pd.DataFrame(dataset[1:], columns=dataset[0]) 
if dataframe.isnull().values.any():
    dataframe.dropna_()

"""
model train
"""
# hyper parameters
embedding_size = 300
num_filters = 100
dropout_rate = 0.5
filters = [3,4,5]
batch_size = 50
n_epochs = 25
lr = 0.01

# data preprocessing
tokenizer = [word.split() for word in dataframe[dataframe['train']=='1']['title']] # tokenizer: 띄어쓰기 기준
vocab = make_vocab(tokenizer)
x_train = encoding(dataframe[dataframe['train']=='1']['title'], vocab)
x_test = encoding(dataframe[dataframe['train']=='0']['title'], vocab)
y_train = y_one_hot(dataframe[dataframe['train']=='1']['label'])
y_test = y_one_hot(dataframe[dataframe['train']=='0']['label'])
custom_train = CustomDataset(x_train, y_train)
data_loader = DataLoader(dataset=custom_train, batch_size=batch_size, shuffle=True, num_workers=2)

# define model
vocab_size = len(vocab)+1
y_dim = y_train.shape[1]
device = device_n_seed(7)
model = CNN_rand(embedding_size, num_filters, dropout_rate, y_dim, filters, vocab_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adadelta(model.parameters(), lr = lr)

total_batch = len(data_loader)
model.train()
for epoch in range(n_epochs):
    avg_cost = 0
    for num, data in tqdm(enumerate(data_loader)):
        avg_cost = 0
        x, y = data
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)

        cost = criterion(pred, y)
        cost.backward()
        optimizer.step()

        avg_cost = avg_cost + (cost / total_batch)
    print(f'epoch = {epoch}, cost = {avg_cost}')

"""
evaluation
"""
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in data_loader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)

        predicts = torch.argmax(outputs, 1)
        total += y.size(0)
        correct += (predicts == torch.argmax(y, 1)).sum().item()
print(correct / total * 100)