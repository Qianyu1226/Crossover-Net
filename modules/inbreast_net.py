import torch.nn as nn
import torch
class hvCNN(nn.Module):
    def __init__(self):
        super(hvCNN, self).__init__()         #
        self.conv1 = nn.Sequential(         # input shape (1, 340, 68)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=(5,5),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 336, 62)
            nn.ReLU(),
            nn.Conv2d(16, 36, (5,5), 1, 0), # output shape (36, 332, 58)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (36, 166, 29)
        )
        self.conv2 = nn.Sequential(         # input shape (36, 166, 29)
            nn.Conv2d(36, 64, (5,5), 1, 0),     # output shape (64, 161, 25)
            nn.ReLU(),                      # activation
            nn.Conv2d(64, 64, (6, 6), 1, 0),  # output shape (64, 156, 20)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),                # output shape (64, 78, 10)
        )
        self.conv3 = nn.Sequential(  # input shape (64, 78, 10)
            nn.Conv2d(64, 64, (3, 3), 1, 0),  # output shape (64, 76, 8)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, (3, 3), 1, 0),  # output shape (64, 74, 6)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, (3, 3), 1, 0),  # output shape (64, 72, 4)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 36, 2)
        )
        self.conv4 = nn.Sequential(  #iutput shape (64, 36, 2)
            nn.Conv2d(64, 64, (2, 2), 1, 0),  # output shape (64, 35, 1)
            nn.ReLU(),  # activation
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, (8, 1), 1, 0),  # output shape (64, 28, 1)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, (8, 1), 1, 0),  # output shape (64, 21, 1)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, (8, 1), 1, 0),  # output shape (64, 14, 1)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, (8, 1), 1, 0),  # output shape (64, 7, 1)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 500, (7, 1), 1, 0),  # output shape (500, 1, 1)
            #nn.ReLU(),  # activation
        )
        self.out = nn.Linear(1000*1*1, 2)  # fully connected layer, output 2D
    def forward(self, vx, hx):
        vx = self.conv1(vx)
        vx = self.conv2(vx)
        vx = self.conv3(vx)
        vx = self.conv4(vx)
        vy = self.conv5(vx)
        vy = vy.view(vy.size(0), -1)  # flatten the output of conv2 to (batch_size, 500)
        vx = vx.view(-1, vx.size(0))
        hx = self.conv1(hx)
        hx = self.conv2(hx)
        hx = self.conv3(hx)
        hx = self.conv4(hx)
        hy = self.conv5(hx)
        hy = hy.view(hy.size(0), -1)
        hx = hx.view(-1, hx.size(0))
        vh = torch.cat([vy, hy], 1)
        vh = nn.functional.relu(vh)
        vhout = self.out(vh)
        return vx, hx, vhout    # 