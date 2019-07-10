import torch.nn as nn
import torch
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self._cross_loss=nn.CrossEntropyLoss(reduce=True, size_average=True)
        # self._vhs_dis=nn.ReLU()
        #self._vhs_dis=nn.Sigmoid()
    def forward(self, vc, vs, hc, hs, vhout, dy): #
        vhc=(vc-hc).mul(vc-hc)
        print(vhc)
        c_temp=torch.sqrt(vhc.sum(0))
        print(c_temp)
        vh_c=torch.mean((c_temp-torch.min(c_temp))/(torch.max(c_temp)-torch.min(c_temp)))
        #print("vh_c ", vh_c)
        vhs = (vs - hs).mul(vs - hs)
        s_temp = torch.sqrt(vhs.sum(0))
        vh_s = -torch.mean((s_temp - torch.min(s_temp)) / (torch.max(s_temp) - torch.min(s_temp)))
        #print("vh_s ", vh_s)
        # vhc=(vc-hc).mul(vc-hc)
        # vh_c=torch.mean(self._vhs_dis(torch.sqrt(vhc.sum(0))))
        # vhs = (vs - hs).mul(vs - hs)
        # vh_s = torch.mean(self._vhs_dis(1 / (torch.sqrt(vhs.sum(0)) + 0.01)))
        return  self._cross_loss(vhout, dy) + vh_c + vh_s