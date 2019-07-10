#import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
from HV_net import hvCNN
from CrossLoss import MyLoss
import datetime
import math
np.set_printoptions(threshold=np.NaN)  #for print
#  load matlab v7.3 files for training data
import h5py
with h5py.File('./data/train_data.mat', 'r') as f:
    f.keys() #
data_x = h5py.File('./data/train_data.mat')['train_data'].value
with h5py.File('./data/train_label.mat', 'r') as f:
    f.keys()
data_y = h5py.File('./data/train_label.mat')['train_label'].value
#

use_cuda=True
# torch.device object used throughout this script
device = torch.device("cuda: 0" if use_cuda else "cpu")
train_epoch=16
train_lr=0.0001
train_momentum=0.5
batchsize=200
featuremaps=64
dime=19
################################################
#cross data
data_x=data_x/255
#data_x=data_x.transpose(0,1,3,2)
data_x=torch.from_numpy(data_x)
data_y=torch.from_numpy(data_y)
data_x=data_x.type(torch.FloatTensor)
data_y=data_y.type(torch.LongTensor)
data_y=data_y.view(data_y.size(0))#
dataset=dataf.TensorDataset(data_x,data_y)
loader=dataf.DataLoader(dataset,batch_size=batchsize,shuffle=True, drop_last=True)
starttime=datetime.datetime.now()
###################################################
def Preloss_special(losstemp, afa):
    rows=int(losstemp.shape[2]/afa)
    special_1=losstemp[:,:,0:rows]
    special_2=losstemp[:,:,dime-rows:dime]
    special= torch.cat([special_1, special_2],1)
    return special
def Preloss_common(losstemp, afa):
    mid=int(losstemp.shape[2]/2)
    rows=round(afa/2)
    common=losstemp[:,:,mid-rows:mid+rows]
    return common
vhnet = hvCNN().to(device)

#nets=[vnet, hnet]
# print(cnn)  # net architecture
optimizer = torch.optim.Adam(vhnet.parameters(), lr=train_lr)   # optimize all cnn parameters
# loss_func =nn.CrossEntropyLoss().to(device)#
my_loss = MyLoss().to(device)
losses_history=[]

for epoch in range(train_epoch):
    epoch_loss = 0
    if epoch > 5 and epoch < 20:
        train_lr=0.00005
    elif epoch >=20:
        train_lr=0.00001
    for step, (d_x, d_y) in enumerate(loader):
        # v_x=d_x[:, :, :, 0: 20]
        # h_x=d_x[:, :, :, 20: 40]
        d_x, d_y = Variable(d_x), Variable(d_y)
        d_x, d_y=d_x.to(device), d_y.to(device)
        vhout=vhnet(d_x[:, :, :, 0: 20],d_x[:, :, :, 20: 40])
        v_loss=vhout[0].t()
        v_loss=v_loss.view(batchsize, featuremaps, dime)
        vloss_special=Preloss_special(v_loss, 5).to(device)
        vloss_common=Preloss_common(v_loss, 5).to(device)
        h_loss = vhout[1].t()
        h_loss = h_loss.view(batchsize, featuremaps, dime)
        hloss_special = Preloss_special(h_loss, 5).to(device)
        hloss_common = Preloss_common(h_loss, 5).to(device)
        loss= my_loss(vloss_common, vloss_special, hloss_common, hloss_special, vhout[2], d_y)  #loss_func(vhout,d_y)+
        #Expected object of type torch.cuda.FloatTensor but found type torch.FloatTensor for argument #3 'other'
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  #vertical
        epoch_loss += loss.item()
        if step % 5 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, step, len(loader), loss.item()))
    losses_history.append(epoch_loss/ len(loader))  # get Python number from 1-element Tensor
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(loader)))
    # 2 ways to save the net
torch.save(vhnet.state_dict(), 'vh_params.pkl')
endtime=datetime.datetime.now()
print((starttime-endtime).seconds)
# plt.plot(losses_history, label='cross')
# plt.legend(loc='best')
# plt.xlabel('epoches')
# plt.ylabel('Loss')
# plt.ylim((-0.1, 1))
# plt.show()