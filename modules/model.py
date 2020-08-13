import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMTagger(nn.Module):
    def __init__(self, HRRP_input_size, pos_input_size, dropout):
        super(LSTMTagger, self).__init__()
        self.hrrp_input = HRRP_input_size
        self.pos_input = pos_input_size
        self.lstm_out = 32
        self.fc_out = 8

        self.target_size = 1
        self.dropout = dropout

        self.lstm_layer = nn.LSTM(
            input_size=self.hrrp_input,
            hidden_size=self.lstm_out
        )
        self.pos_fc = nn.Sequential(
            nn.Linear(self.pos_input, self.fc_out),
            nn.ReLU()
        )
        self.final_fc = nn.Sequential(
            nn.Linear(self.lstm_out+self.fc_out, 16),
            nn.ReLU()
        )
        self.output = nn.Linear(16, 2)

    def forward(self, HRRP_in, Pos_in):
        HRRP_feature, _ = self.lstm_layer(HRRP_in)
        pos_feature = self.pos_fc(Pos_in)

        feature = torch.cat((HRRP_feature[:, -1, :], pos_feature), 1)

        output = self.output(self.final_fc(feature))
        return F.softmax(output, dim=1)
