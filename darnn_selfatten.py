import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable
import csv

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import warnings
warnings.filterwarnings('ignore')

class trainset(Dataset):
    def __init__(self,df,T,test=False,test_end=20200101,test_begin=20190101,train_begin=20101231):
        atom=['vol','high','low','open','amount','close']
        if test == True:
            self.df = df[df['trade_date']>=test_begin][df['trade_date']<test_end][atom]
        else:
            self.df = df[df['trade_date']<test_begin][df['trade_date']>train_begin][atom]
        
        self.T=T
        
    def __getitem__(self,index):
        target=int(np.sign(self.df.iloc[index+self.T-1,5]-self.df.iloc[index+self.T-2,5]))+1
        target=(target>1)+0
        
        x = np.array(self.df.iloc[index:index+self.T - 1, :5])
        x = (x - np.mean(x))/np.std(x)
        
        y = np.array(self.df.iloc[index:index+self.T - 1, 5])
        y = (y - np.mean(y))/np.std(y)
        
        return x, y, target#返回的目标是0 ,1
    
    def __len__(self):
        return len(self.df)-self.T+1

class randomdata(Dataset):
    def __init__(self,x, y, t):
        
        self.x = x
        self.y = y
        self.t = t
        
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.t[index]#返回的目标是0 ,1
    
    def __len__(self):
        return len(self.t)
    

    


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers = 1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        #self.encoder_attn = nn.Linear(
        #    in_features=2 * self.encoder_num_hidden + self.T - 1,
        #    out_features=1
        #)
        self.encoder_attn = nn.Sequential(
            nn.Linear(2 * encoder_num_hidden + self.T-1, self.T-1),
            nn.Tanh(),
            nn.Linear(self.T-1, 1)
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())
        #print('X',X.size())#[128,9,81]
        #print('X_tilde',X_tilde.size())#[128,9,81]
        #print('X_encoded',X_encoded.size())#[128,9,128]

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        #print('h_n',h_n.size())#[1,128,128]
        #print('s_n',s_n.size())#[1,128,128]
        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            #h和s在t时刻是被81个特征共享的，所以复制81层 然后拼接上特征X且要把时间列拼接
            #即把X(1)放到X(2)然后拼接2
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)
            #print('拼接',x.shape)
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))
            #print('2拼接',x.shape)
            # get weights by softmax
            #是不是忘了取tanh？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
            #alpha展开成128，81
            alpha = F.softmax(x.view(-1, self.input_size),1)
            #print('alpha',alpha.shape)
            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])
            #128,81
            #print('x_tilde',x_tilde.shape)
            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_encoded[:, t, :] = h_n
        #X_encoded   [128,9,128]
        #print('X_encoded',X_encoded.shape)
        #print(X_encoded)
        return X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X

        Returns:
            initial_hidden_states
        """
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())

class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 3)
        self.softmax=nn.Softmax()
        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T - 1):
            #128，9，128*3
            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)
            #128,9
            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1),1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            #128,1,9 x 128,9,128=128,1,128       
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            #print('context',context.shape)
            #context (128,128)
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                #print("y_prev[:,t]",y_prev[:, t].shape)
                #y_prev (128,9)   y_prev[:, t](128)
                #y_prev[:, t].unsqueeze(1)   (128,1)
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                #print("y_tilde",y_tilde.shape) (128,129)
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden
        #print(d_n.squeeze(0))
        return d_n.squeeze(0)

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())

class SelfAttention(nn.Module):
    def __init__(self, last_hidden_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.last_hidden_size = last_hidden_size
        self.hidden_size = hidden_size

        self.wq = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wk = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wv = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)

        # 输出
        self.ln = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        # h: batch_size * last_hidden_size


        #q k v  batch_size * hidden_size
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        dk = q.size(-1)
        z = torch.mm(q, k.t()) / math.sqrt(dk)  # (b, hidden_size) * (hidden_size, b) ==> (b, b)
        beta = F.softmax(z, dim=1)
        st = torch.mm(beta, v)  # (b, b) * (b, hidden_size) ==> (b, hidden_size)
        
        # b * 1
        y_res = self.ln(st)
        y_res = self.sigmoid(y_res.squeeze(1))

        return y_res

class Darnn_selfattention(nn.Module):
    def __init__(self,input_size, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 learning_rate):
        super(Darnn_selfattention,self).__init__()
        self.input_size = input_size#81
        self.T = T
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.Encoder = Encoder(input_size=self.input_size,
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)
        self.attention = SelfAttention(last_hidden_size=encoder_num_hidden,hidden_size=decoder_num_hidden)
        
        if torch.cuda.is_available():
            self.Encoder = self.Encoder.cuda()
            self.Decoder = self.Decoder.cuda()
            self.attention = self.attention.cuda()
            
        self.Encoder_optim = optim.Adam(params=filter(lambda p: p.requires_grad,self.Encoder.parameters()),
                                        lr=self.learning_rate)
        self.Decoder_optim = optim.Adam(params=filter(lambda p: p.requires_grad,self.Decoder.parameters()),
                                        lr= self.learning_rate)
        self.attention_optim = optim.Adam(params=filter(lambda p: p.requires_grad,self.attention.parameters()),
                                          lr=self.learning_rate)
        
        self.loss_func = nn.BCELoss()
        #self.loss_func = nn.CrossEntropyLoss()
    def forward(self,x,y):
        out1 = self.Encoder(x)
        out2 = self.Decoder(out1, y)
        

        out3 = self.attention(out2)
        return out3

def metrics(results, ori_y):
    '''
        predict: 预测y(需要转化成证整数种类 比如 0  1  2  3 ；若使用二分类 sigmod激活函数，则需要转化一下 比如(out > 0.5)+0)
        ori_y: 原始的01标签
    '''
    accuracy = accuracy_score(ori_y, results)
    precision = precision_score(ori_y, results, labels=[1], average=None)[0]
    recall = recall_score(ori_y, results, labels=[1], average=None)[0]
    f1 = f1_score(ori_y, results, labels=[1], average=None)[0]
    return accuracy, precision, recall, f1

def rand_acc(t_ori):
    return max([np.sum(np.array(t_ori) == r) for r in [0, 1]]) * 1. / len(t_ori)



input_size=5
T = 21
nhidden_encoder = 32
nhidden_decoder = 32
lr = 0.001

print("==> Initialize DA-RNN model ...")
model = Darnn_selfattention(
    input_size,
    T,
    nhidden_encoder,
    nhidden_decoder,
    lr)

#打乱数据集提高GPU利用率

datapath='HScode/'
filelist=os.listdir(datapath)
test_end=20200101
test_begin = 20190101
train_begin =20100101
N=300
valid_ratio = 0.3
epochs =3000


# Read dataset

result = {}
n_iter = 0

acc_train_max_diff = 0
acc_val_max_diff = 0
acc_test_max_diff = 0

x_all = []
y_all = []
t_all = []

x_test = []
y_test = []
t_test = []

#储存训练数据 
print("===Train and Val data is on the way!===>")
for filename in filelist[:N]:
    df=pd.read_csv(os.path.join(datapath,filename))
    if len(df[df['trade_date']<test_begin]) <T:
        continue
    trainloader=DataLoader(trainset(df,T,False,test_end=test_end,
                                    test_begin=test_begin,train_begin=train_begin),batch_size=512,shuffle=True)
    for _,sample in enumerate(trainloader):

        '''x=Variable(torch.from_numpy(np.array(sample[0])).type(torch.FloatTensor).to(model.device))
        x.requires_grad=True
        y=Variable(torch.from_numpy(np.array(sample[1])).type(torch.FloatTensor).to(model.device))
        y.requires_grad=True
        t = Variable(torch.from_numpy(
            np.array(sample[2])).type(torch.FloatTensor).to(model.device))'''
            
        x_all.extend(sample[0])
        y_all.extend(sample[1])
        t_all.extend(sample[2])
print("===Train and Val data are ready!===>")
#储存测试集数据 
print("===Test data is on the way!===>")
for filename in filelist[:N]:
    df=pd.read_csv(os.path.join(datapath,filename))
    if len(df[df['trade_date']>test_begin][df['trade_date']<test_end]) <T:
        continue
    testloader=DataLoader(trainset(df,T,test=True,test_end=test_end,
                                    test_begin=test_begin,train_begin=train_begin),batch_size=512,shuffle=False)
    for _,sample in enumerate(testloader):
        '''x = Variable(torch.from_numpy(np.array(sample[0])).type(torch.FloatTensor).to(model.device))
        y = Variable(torch.from_numpy(np.array(sample[1])).type(torch.FloatTensor).to(model.device))
        t = Variable(torch.from_numpy(
            np.array(sample[2])).type(torch.int64).to(model.device))'''
        x_test.extend(sample[0])
        y_test.extend(sample[1])
        t_test.extend(sample[2])


print("===Test data is ready!===>")

#定义好数据生成器
traindata = DataLoader(randomdata(x_all, y_all, t_all), batch_size=256, shuffle=True)
testdata =  DataLoader(randomdata(x_test, y_test, t_test), batch_size=256, shuffle=False)

print("==> Start training ...")
for epoch in range(epochs):
    print('====epoch:'+str(epoch+1)+' ====>正在训练')
    
    model.train()
    torch.save(model, 'v1.pth')
    
    iter_losses=[]
    t_pred = []
    t_ori = []
    
    x_valid = []
    y_valid = []
    t_valid = []
    
    print('\033[1;34m Train: \033[0m')
    for _,sample in enumerate(traindata):            
            x=Variable(torch.from_numpy(np.array(sample[0])).type(torch.FloatTensor).to(model.device))
            x.requires_grad=True
            y=Variable(torch.from_numpy(np.array(sample[1])).type(torch.FloatTensor).to(model.device))
            y.requires_grad=True
            t = Variable(torch.from_numpy(
                np.array(sample[2])).type(torch.FloatTensor).to(model.device))
            
            if np.random.uniform() < valid_ratio:
                x_valid.append(x)
                y_valid.append(y)
                t_valid.append(t)
                continue
                
            model.Encoder_optim.zero_grad()
            model.Decoder_optim.zero_grad()
            model.attention_optim.zero_grad()

            t_out=model(x,y)
            loss_all = model.loss_func(t_out, t)
            loss_all.backward()
            
            t_out = (t_out >= 0.5) + 0
            t_pred.extend(t_out.data.cpu().numpy())
            t_ori.extend(t.data.cpu().numpy())
                    
            model.Encoder_optim.step()
            model.Decoder_optim.step()
            model.attention_optim.step()

            loss=loss_all.item()
            iter_losses.append(loss)
            n_iter += 1 #记录总共的批次数 比如epoch=0 1 后 就会成128*2+1（初始化在epoch的循环外）

            '''if n_iter % 2000 == 0 and n_iter != 0:#每一千次 更新一下学习率为0.9倍
                for param_group in model.Encoder_optim.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in model.Decoder_optim.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in model.attention_optim.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9'''
            
    train_accuracy, precision, recall, f1 = metrics(t_pred,t_ori)
    train_random = rand_acc(t_ori)
    acc_train_max_diff = max(acc_train_max_diff, train_accuracy-train_random)
    print('第 \033[1;34m %d \033[0m 轮的训练集正确率为:\033[1;32m %.4f \033[0m epoch_Loss 为: \033[1;32m %.4f \033[0m' %
          (epoch+1,train_accuracy,np.mean(iter_losses)))
    print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (train_accuracy, precision, recall, f1))
    print('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' % (train_random, acc_train_max_diff))
            

    
    print('\033[1;34m Valid: \033[0m')
    model.eval()
    with torch.no_grad():
        t_pred = []
        t_ori = []
        for i in range(len(t_valid)):   
 
            t_out = model(x_valid[i],y_valid[i])

            t_out = (t_out >= 0.5) + 0
            t_pred.extend(t_out.data.cpu().numpy())
            t_ori.extend(t_valid[i].data.cpu().numpy())
            
        validation_accuracy, precision, recall, f1 = metrics(t_pred,t_ori)
        validation_random = rand_acc(t_ori)
        acc_val_max_diff = max(acc_val_max_diff, validation_accuracy-validation_random)
        print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (validation_accuracy, precision, recall, f1))
        print('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' % (validation_random, acc_val_max_diff))
    
    
    

    if (epoch+1) % 5 == 0:
        print('\033[1;34m Test: \033[0m')
        model.eval()
        with torch.no_grad():
            t_pred = []
            t_ori = []
            for _,sample in enumerate(testdata):
                x = Variable(torch.from_numpy(np.array(sample[0])).type(torch.FloatTensor).to(model.device))
                y = Variable(torch.from_numpy(np.array(sample[1])).type(torch.FloatTensor).to(model.device))
                t = Variable(torch.from_numpy(
                    np.array(sample[2])).type(torch.int64).to(model.device))
                t_out = model(x,y)
                t_out = (t_out >= 0.5) + 0
                t_pred.extend(t_out.data.cpu().numpy())
                t_ori.extend(t.data.cpu().numpy())
            test_accuracy, precision, recall, f1 = metrics(t_pred,t_ori)
            test_random = rand_acc(t_ori)
            acc_test_max_diff = max(acc_test_max_diff, test_accuracy-test_random)
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (test_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\ttestMaxAccDiff:%.6f \033[0m' % (test_random, acc_test_max_diff))
            
            
            
            
    ''' result[epoch+1] = {
            'loss': np.mean(iter_losses),
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy,
            'validation_maxdiff': acc_val_max_diff,
            'test_accuarcy': test_accuracy,
            'acc_test_maxdiff':acc_test_max_diff,
            'epoch': epoch+1
        }
    else:
        result[epoch+1] = {
            'loss': np.mean(iter_losses),
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy,
            'validation_maxdiff': acc_val_max_diff,
            'epoch':epoch+1
        }'''

    print('\n')