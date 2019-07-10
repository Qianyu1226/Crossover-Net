import torch.nn as nn
import torch


class hvCNN(nn.Module):
    def __init__(self):
        super(hvCNN, self).__init__()         #必须调用父类的构造函数，传入类名和self
        self.conv1 = nn.Sequential(         # input shape (1, 100, 20)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=(5,3),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 96, 18)
            nn.ReLU(),
            nn.Conv2d(16, 36, (5,3), 1, 0), # output shape (36, 92, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (36, 46, 8)
        )
        self.conv2 = nn.Sequential(         # input shape (36, 46, 8)
            nn.Conv2d(36, 64, (5,3), 1, 0),     # output shape (64, 42, 6)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 21, 3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 1, 0),  # output shape (64, 19, 1)
            nn.ReLU(),  # activation
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (7, 1), 1, 0),  # output shape (64, 13, 1)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, (7, 1), 1, 0),  # output shape (64, 7, 1)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 500, (7, 1), 1, 0),  # output shape (500, 1, 1)
            #nn.ReLU(),  # activation
        )
        self.out = nn.Linear(1000*1*1, 2)  # fully connected layer, output 2维
    def forward(self, vx, hx):
        vx = self.conv1(vx)
        vx = self.conv2(vx)
        vx = self.conv3(vx)
        vy = self.conv4(vx)
        vy = vy.view(vy.size(0), -1)  # 变成一行，flatten the output of conv2 to (batch_size, 500)
        vx = vx.view(-1, vx.size(0))
        #print("the shape of hx: ", hx.shape)
        hx = self.conv1(hx)
        hx = self.conv2(hx)
        hx = self.conv3(hx)
        hy = self.conv4(hx)
        hy = hy.view(hy.size(0), -1)
        hx = hx.view(-1, hx.size(0))
        vh = torch.cat([vy, hy], 1)
        vh = nn.functional.relu(vh)
        vhout = self.out(vh)
        return vx, hx, vhout    # X作为计算Loss时使用