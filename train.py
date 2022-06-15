import time
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.interpolate import make_interp_spline, BSpline

encode_dic = {'0': 0}
co = 0
x = []
y = []
y_label = []
non_feature_train = 40
features = ['smoke', 'smoker', 'smokes', 'smoked', 'smoking', 'cigarette', 'tobacco']
status = ['CURRENT_SMOKER', 'NON-SMOKER', 'PAST_SMOKER', 'UNKNOWN']

def prepcs(adr, adrc, co, non_feature, txt_type):
    with open(adr, 'r') as f:
        with open(adrc, 'w') as f2:
            data_lst = []
            in_feature_count = 0
            for word in (line.split() for line in f):
                word_lst = []
                in_feature = False
                for i in word:
                    if i.lower() in features:
                        in_feature = True
                        if in_feature_count == 0:
                            non_feature -= 1
                        in_feature_count += 1
                        f2.write('Feature: ' + i + '\n')
                        f2.write(str(non_feature) + '\n') 
                    if i not in encode_dic:
                        if txt_type == 'train':
                            encode_dic[i] = co
                            co += 1
                        else:
                            encode_dic[i] = 0
                if in_feature:   
                    for j in word:                   
                        word_lst.append(encode_dic[j])
                        f2.write(j + ' ')
                    f2.write('\n' + str(word_lst) + '\n')
                    data_lst += word_lst 
    return data_lst, co, non_feature, len(data_lst)

for ismoke in status:
    for id in range(1000):
        adr_train = './data/training/{}_ID_{}.txt'.format(ismoke, id)
        adrc_train = './data/training/{}_ID_{}_encoded.txt'.format(ismoke, id)
        try:
            data_lst, co, non_feature_train, lst_len = prepcs(adr_train, adrc_train, co, non_feature_train, 'train')
            x.append(data_lst)
            if lst_len > 0:
                if ismoke == 'CURRENT_SMOKER':
                    y_label = 0
                elif ismoke == 'NON-SMOKER':
                    y_label = 1
                elif ismoke == 'PAST_SMOKER':
                    y_label = 2
            else:
                y_label = 3
            y.append(y_label)           
        except:
            continue

#print('No features:', non_feature)
len_lst = [len(data) for data in x if len(data) > 0]
#print(len_lst)
mean = np.mean(len_lst)
std = np.std(len_lst)
data_length = int(mean + std)
#print('Data length:', data_length)

def data_len_pcs(x, data_length):
    for i in range(len(x)):
        if (len(x[i]) < data_length):
            for it in range(data_length - len(x[i])):
                x[i].insert(0, 0)
        x[i] = x[i][:data_length]

data_len_pcs(x, data_length)

x_test = []
num_test = 40
non_feature_test = num_test
for num in range(num_test):
    text = ('TEST_' if num >= 9 else 'TEST_0')
    adr_test = './data/test/' + text + '{}.txt'.format(num+1)
    adrc_test = './data/test/' + text + '{}_c.txt'.format(num+1)
    data_lst, co, non_feature_test, lst_len = prepcs(adr_test, adrc_test, co, non_feature_test, 'test')
    x_test.append(data_lst)
 
data_len_pcs(x_test, data_length)

x_tensor = torch.LongTensor(x) 
y_tensor = torch.LongTensor(y)
x_test_tensor = torch.LongTensor(x_test) 
y_test_tensor = torch.LongTensor(np.zeros(40)) 

dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [30, 10], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset = train_dataset, batch_size = 30, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = 10, shuffle = False)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset = test_dataset, batch_size = 10, shuffle = False)

# LSTM
class RNN(nn.Module):
    def __init__(self, num_word, input_size, hidden_size, num_layers, bidirectional):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = self.hidden_size,
            num_layers = num_layers,
            bidirectional = self.bidirectional
        )

        embedding_dim = input_size
        self.embeddings = nn.Embedding(num_word + 1, embedding_dim)
    
        if self.bidirectional:
            self.fully_connected = nn.Linear(40, 4)
        else:
            self.fully_connected = nn.Linear(40, 4)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        states, hidden = self.lstm(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim = 1)
        outputs = self.fully_connected(encoding)
        return outputs

lr = 0.01
num_epochs = 300
input_size = 10
num_hidden = 10
num_layers = 2
bidirectional = True

device = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')

model = RNN(len(encode_dic), input_size, num_hidden, num_layers, bidirectional)

model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

plot_x = [i + 1 for i in range(num_epochs)]
plot_x = np.array(plot_x)
plot_y_loss = []
plot_y_acc = []

def train(model, num_epochs, loss_function, optimizer, train_iter, val_iter):
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        model.train()
        for x, y in train_iter:
            n += 1
            optimizer.zero_grad()
            x = Variable(x.to(device))
            y = Variable(y.to(device))

            yhat = model(x)
            
            loss = loss_function(yhat, y)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(yhat.cpu().data, dim = 1), y.cpu())
            train_loss = loss
        
        with torch.no_grad():
            model.eval()
            for x_val, y_val in val_iter:
                m += 1
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                yhat_val = model(x_val)
                val_loss = loss_function(yhat_val, y_val)
                val_acc += accuracy_score(torch.argmax(yhat_val.cpu().data, dim = 1), y_val.cpu())
                val_losses += val_loss
        
        runtime = time.time() - start
        print('epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f' %
            (epoch + 1, train_loss.data/n, train_acc/n, val_losses.data/n, val_acc/n, runtime))
        plot_y_loss.append([train_loss.data/n, val_losses.data/n])
        plot_y_acc.append([train_acc/n, val_acc/n])
        
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join('./', 'last_model.pt'))

def predict(model, test_iter):
    pred_lst = []
    softmax = nn.Softmax(dim = 1)
    with torch.no_grad():
        model.eval()
        for batch, label in test_iter:
            batch = batch.to(device)
            output = model(batch)
            pred_lst.extend(torch.argmax(softmax(output), dim = 1).cpu().numpy())
    return pred_lst

train(model, num_epochs, loss_function, optimizer, train_loader, val_loader)

def draw(x, y, types):   
    xnew = np.linspace(x.min(), x.max(), 300)  
    spl = make_interp_spline(x, y, k = 3)
    y_smooth = spl(xnew)
    a = plt.plot(xnew, y_smooth)

    plt.title(types.capitalize())
    plt.xlabel('epoch')
    plt.ylabel(types.capitalize())
    plt.legend(a, ('train', 'val'))

    plt.savefig(types + '_graph.png')
    plt.show()

draw(plot_x, plot_y_loss, 'loss')
draw(plot_x, plot_y_acc, 'acc')

pred = predict(model, test_loader)

pred_class = []
for result in pred:
    pred_class.append(status[result])

adr_result = 'case1_5.txt'
with open (adr_result, 'w') as f3:
    for i in range(data_length):
        f3.write(pred_class[i] + '\n')