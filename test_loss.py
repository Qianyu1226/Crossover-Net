#import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from modules.inbreast_net import hvCNN
from scipy import io
np.set_printoptions(threshold=np.NaN)  #for print
import h5py
with h5py.File('./test_inbreast/testdata.mat', 'r') as f:
    f.keys() # matlabdata.mat 
test_x = h5py.File('./test_inbreast/testdata.mat')['testdata'].value
with h5py.File('./test_inbreast/testlabel.mat', 'r') as f:
    f.keys() # matlabdata.mat 中的变量名
test_y = h5py.File('./test_inbreast/testlabel.mat')['testlabel'].value
# test_y = np.zeros((2000, 1))
use_cuda=True
device = torch.device("cuda: 0" if use_cuda else "cpu")
batchsize=100

test_x=test_x/255
test_x=torch.from_numpy(test_x).type(torch.FloatTensor)
test_y=torch.from_numpy(test_y)
test_y=test_y.type(torch.LongTensor)
test_y=test_y.view(test_y.size(0))
dataset=dataf.TensorDataset(test_x,test_y)
loader=dataf.DataLoader(dataset,batch_size=batchsize,shuffle=False, drop_last=True)
vhnet = hvCNN().to(device)
vhnet.load_state_dict(torch.load('vh_params_2.pkl'))  #, map_location='cpu'
pred_pro=[]
pred_y=[]
#print(test_x[:, :, :, 0: 20].shape)

for step, (d_x, d_y) in enumerate(loader):
    print(step)
    d_x, d_y = d_x.to(device), d_y.to(device)
    vhout = vhnet.forward(d_x[:, :, :, 0: 20], d_x[:, :, :, 20: 40])
    pred_pro.append(F.softmax(vhout[2], 1).data)  #
    pred_y.append(torch.max(vhout[2], 1)[1].data.cpu())#.squeeze()
row=len(pred_y)
col=batchsize
itc = torch.arange(row).int()
p=np.ones([row,col])
for i in itc:
    p[i, :] = np.mat(pred_y[i])
io.savemat('pred_result.mat', {'result': p})