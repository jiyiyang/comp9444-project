# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.l1 = nn.Linear(2,num_hid)
        self.l2 = nn.Linear(num_hid,1)
        self.t = nn.Tanh()
        self.s = nn.Sigmoid()
        # INSERT CODE HERE

    def forward(self, input):
        r = torch.norm(input,2,dim = -1).unsqueeze(-1)
        a = torch.atan2(input[:,1],input[:,0]).unsqueeze(-1)
        x = torch.cat((r,a),-1)
        x_1 = self.l1(x)
        x_2 = self.t(x_1)
        self.hid1 = x_2
        x_3 = self.l2(x_2)
        output= self.s(x_3)

        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.l1 = nn.Linear(2,num_hid)
        self.l2 = nn.Linear(num_hid,num_hid)
        self.l3 = nn.Linear(num_hid,1)
        self.t = nn.Tanh()
        self.r = nn.ReLU()
        self.s = nn.Sigmoid()

    def forward(self, input):
        x= self.l1(input)
        x_1 = self.t(x)
        self.hid1 = x_1
        x_2 = self.l2(x_1)
        x_3 = self.t(x_2)
        self.hid2 = x_3
        x_4 = self.l3(x_3)
        output = self.s(x_4)# CHANGE CODE HERE
        return output


class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        self.l1_2 = nn.Linear(2,num_hid)
        self.l1_3 = nn.Linear(2,num_hid)
        self.l1_4 = nn.Linear(2,1)
        self.l2_3 = nn.Linear(num_hid,num_hid)
        self.l3_4 = nn.Linear(num_hid,1)
        self.l2_4 = nn.Linear(num_hid,1)
        self.s = nn.Sigmoid()
        self.t = nn.Tanh()


    def forward(self, input):
        x1_2 = self.l1_2(input)
        x1_3 = self.l1_3(input)
        x1_4 = self.l1_4(input)
        hid1 = self.t(x1_2)
        self.hid1 = hid1
        x2_3 = self.l2_3(self.hid1)
        x2_4 = self.l2_4(self.hid1)
        x_total = x1_3+x2_3
        hid2 = self.t(x_total)
        self.hid2 = hid2
        x3_4 = self.l3_4(self.hid2)
        x2total = x1_4+x2_4+x3_4
        output = self.s(x2total)
         # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():
        net.eval()
        net(grid)
        if layer ==1:
            pred = (net.hid1[:,node]>=0).float()
        elif layer ==2:
            pred = (net.hid2[:,node]>=0).float()

        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]),cmap='Wistia')

    # INSERT CODE HERE
